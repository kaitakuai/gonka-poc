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
      1. Register PoC-aware model classes via ``vllm.ModelRegistry``.
      2. Import side-effect modules that decorate CustomOp / PluggableLayer
         classes with ``@<Base>.register_oot`` -- registration happens at
         import time.
      3. Optionally hook ``transformers.AutoConfig.register(...)`` for any
         custom HF config class we ship.

    NOTE: we do NOT install the worker extension here -- that lives behind
    the ``--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`` CLI
    flag, which vLLM consumes during ParallelConfig parsing.
    """
    global _registered
    if _registered:
        return

    try:
        _register_models()
    except Exception:  # pragma: no cover - defensive
        # Per the vllm plugin contract, swallow and log. The host will also
        # log at exception level.
        logger.exception("gonka_poc.plugin.register: model registration failed")

    try:
        _register_custom_ops()
    except Exception:  # pragma: no cover
        logger.exception("gonka_poc.plugin.register: custom_ops import failed")

    _registered = True


def _register_models() -> None:
    """Register / override model classes in ``vllm.ModelRegistry``.

    For the PoC v2 port we currently only seed Qwen3MoeForCausalLM custom_ops
    defaults -- we do NOT swap the model class itself (the in-tree
    Qwen3MoeForCausalLM is still the forward we want).

    If a future PoC variant needs a custom subclass, register it lazily:

        from vllm import ModelRegistry
        ModelRegistry.register_model(
            "Qwen3MoeForCausalLM",
            "gonka_poc.models.qwen3_moe_poc:Qwen3MoeForCausalLMPoC",
        )

    The "<module>:<class>" form avoids importing torch at plugin load time.
    """
    # TODO(layer-1): if we end up shipping a Qwen3MoeForCausalLMPoC subclass,
    # register it here. For now this is a no-op so the function still has a
    # stable call site for tests.
    from vllm import ModelRegistry  # deferred import

    _ = ModelRegistry  # touch to assert the import path is real
    logger.debug("gonka_poc: model registry touched (no overrides registered)")


def _register_custom_ops() -> None:
    """Import side-effect modules whose decorators register OOT custom ops.

    Each module under ``gonka_poc.custom_ops.*`` is expected to apply
    ``@RotaryEmbedding.register_oot`` / ``@CustomOp.register_oot(name=...)``
    decorators at import time.

    Until layer-1 PoC custom_ops land, this is a no-op so plugin
    registration stays cheap.
    """
    # TODO(layer-1): create gonka_poc/custom_ops/__init__.py with the OOT
    # decorators and `import gonka_poc.custom_ops  # noqa: F401` here.
    return None


__all__ = ["register"]
