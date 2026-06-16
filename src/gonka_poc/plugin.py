"""``vllm.general_plugins`` entry point for the gonka-poc package.

Registration is declared in ``pyproject.toml``:

    [project.entry-points."vllm.general_plugins"]
    gonka_poc = "gonka_poc.plugin:register"

vLLM 0.23.0 calls :func:`register` in every process that touches the model
class: process0, the V1 engine-core process, every worker process, and the
registry inspection subprocess. The function MUST therefore be:

  * re-entrant (idempotent across multiple calls in one process),
  * cheap (no CUDA init, no big imports at module scope),
  * exception-safe (vllm.plugins.load_general_plugins swallows exceptions
    but logs them at exception level -- we still want to avoid crashing).

Verified call sites (vllm 0.23.0):
  - vllm/engine/arg_utils.py:747   (sync engine startup)
  - vllm/v1/engine/core.py:109     (V1 engine-core process)
  - vllm/v1/worker/worker_base.py:247   (every worker process)
  - vllm/model_executor/models/registry.py:1411   (inspection subprocess)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("gonka_poc.plugin")

_registered: bool = False


def register() -> None:
    """vllm.general_plugins entry point. Idempotent.

    Tasks performed (each guarded for repeated calls):
      1. Set the ``gonka_poc.PLUGIN_LOADED`` process-local flag so the
         FastAPI startup hook can detect "plugin loaded but no gate
         attached" (operator likely ran plain ``vllm serve`` instead of
         ``gonka-vllm-serve``).
      2. Install a one-shot warning wrapper around
         ``vllm.entrypoints.openai.api_server.build_app`` so the same
         gate-presence check fires for both ``vllm serve`` and our own
         ``gonka-vllm-serve`` path.

    NOTE: we do NOT install the worker extension here -- that lives behind
    the ``--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`` CLI
    flag, which vLLM consumes during ParallelConfig parsing.
    """
    global _registered
    if _registered:
        return

    # Expose a process-local flag so the FastAPI startup-event hook in
    # ``gonka_poc.entrypoint.api_router`` can detect "plugin loaded but no
    # gate attached" (operator likely ran plain ``vllm serve`` instead of
    # ``gonka-vllm-serve``). Set BEFORE the inner registration steps so
    # the flag survives even if a sub-step raises.
    import gonka_poc as _pkg  # local import: avoid cycles at module import

    _pkg.PLUGIN_LOADED = True

    try:
        _install_build_app_warning_wrapper()
    except Exception:  # pragma: no cover
        logger.exception(
            "gonka_poc.plugin.register: build_app warning wrapper install failed"
        )

    _registered = True


def _register_custom_ops() -> None:
    """Import side-effect modules whose decorators register OOT custom ops.

    Each module under ``gonka_poc.custom_ops.*`` is expected to apply
    ``@RotaryEmbedding.register_oot`` / ``@CustomOp.register_oot(name=...)``
    decorators at import time.

    Until layer-1 PoC custom_ops land, this is a no-op so plugin
    registration stays cheap.
    """
    # NOTE(layer-1): create gonka_poc/custom_ops/__init__.py with the OOT
    # decorators and `import gonka_poc.custom_ops  # noqa: F401` here.
    return None


_build_app_wrapped: bool = False


def _install_build_app_warning_wrapper() -> None:
    """Wrap ``vllm.entrypoints.openai.api_server.build_app`` with a one-shot
    startup-event registrar.

    Purpose: catch the operator footgun where ``vllm serve`` runs instead of
    ``gonka-vllm-serve``. The plugin's :func:`register` still executes (so
    ``PLUGIN_LOADED`` is set) but no ``app.state.gonka_gate`` is ever
    attached -- the chat endpoint would NOT be 503-gated while PoC seizes
    the GPU.

    We mutate the upstream module so the wrapper is in effect for both the
    bare ``vllm serve`` invocation and our own ``gonka-vllm-serve`` path
    (where the wrapper just no-ops on top of an already-registered hook).

    Best-effort: if vllm or its api_server module is not importable in this
    process (e.g. a worker subprocess where the entrypoint isn't even
    referenced), we silently skip.
    """
    global _build_app_wrapped
    if _build_app_wrapped:
        return

    try:
        from vllm.entrypoints.openai import api_server as _api_server  # type: ignore
    except Exception:
        # No api_server in this process -- engine-core / worker / inspection
        # subprocesses don't need this hook. Not an error.
        return

    original_build_app = getattr(_api_server, "build_app", None)
    if original_build_app is None:
        return

    def _wrapped_build_app(*args, **kwargs):  # type: ignore[no-untyped-def]
        app = original_build_app(*args, **kwargs)
        try:
            from gonka_poc.entrypoint.api_router import (
                register_gate_presence_warning,
            )

            register_gate_presence_warning(app)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "gonka_poc: failed to attach gate-presence warning to FastAPI app"
            )
        return app

    _api_server.build_app = _wrapped_build_app  # type: ignore[assignment]
    _build_app_wrapped = True


__all__ = ["register"]
