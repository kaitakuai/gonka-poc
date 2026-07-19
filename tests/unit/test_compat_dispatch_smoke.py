"""CPU-level smoke test for the ``gonka_poc._compat`` public dispatch API.

This test exists to catch a class of regression that hit us once already:
``current`` used to be exposed as an ``lru_cache``-wrapped function with a
PEP 562 ``__getattr__`` shim that tried to dual-publish the compat module's
attrs through the function object. Real ``from gonka_poc._compat import
current`` binds the *function*, not the module, so the very first call site
of the form ``compat.build_common_attention_metadata(...)`` raised
``AttributeError`` on hardware. The contract is now: ``current()`` returns a
*module*; the caller MUST invoke it.

This test pins that contract on every CI run, in <1 second, with zero GPU.
It also imports ``gonka_poc.poc.poc_model_runner`` to catch a future
compat-binding regression at *import time* rather than on first forward.

Pattern: ``pytest.importorskip("vllm")`` keeps the suite usable on
contributor laptops without a vLLM install; once vLLM is importable, we
resolve the compat module exactly once via the documented public API and
assert the full attribute surface promised by ``_compat/v0_23.py::__all__``.
"""
from __future__ import annotations

from types import ModuleType

import pytest

# Compat dispatch reads ``vllm.__version__`` to pick a submodule. Without
# vllm the dispatch path cannot resolve, so we skip on contributor machines
# that don't have it installed. CI's vLLM image has it.
pytest.importorskip("vllm")

from gonka_poc._compat import current  # noqa: E402 -- after importorskip


# Symbols every supported compat submodule (currently only v0_23) MUST
# expose. Mirror of ``gonka_poc/_compat/v0_23.py::__all__``. Optional symbols
# (e.g. install_model_runner_forward_hook if it returns) go in
# ``_OPTIONAL_ATTRS`` -- absence is fine, but if present they must be
# callable.
_REQUIRED_ATTRS: tuple[str, ...] = (
    "build_common_attention_metadata",
    "build_attn_metadata_per_group",
    "get_kv_cache_pool",
    "abort_all_requests",
)

_OPTIONAL_ATTRS: tuple[str, ...] = (
    "install_model_runner_forward_hook",
)


def _resolve_compat_module() -> ModuleType:
    """Resolve the compat submodule via the only supported public API.

    Centralised so individual asserts read against a single resolution
    rather than reinventing the call shape per-test.
    """
    compat = current()
    return compat


def test_current_returns_module() -> None:
    """``current()`` MUST return a *module*, not a function/proxy/lru_cache.

    This is the exact contract that broke in the earlier revision; pinning
    it here means a future refactor cannot silently regress without CI red.
    """
    compat = _resolve_compat_module()
    assert isinstance(compat, ModuleType), (
        f"gonka_poc._compat.current() must return a module instance; "
        f"got {type(compat).__name__!r}. See _compat/__init__.py history "
        "for why lru_cache/PEP 562 dual-publish is forbidden."
    )


def test_current_exposes_required_attrs() -> None:
    """All required compat hooks MUST exist and MUST be callable."""
    compat = _resolve_compat_module()
    missing = [name for name in _REQUIRED_ATTRS if not hasattr(compat, name)]
    assert not missing, (
        f"gonka_poc._compat.current() module is missing required attrs: "
        f"{missing}. Expected the full surface from v0_23.py::__all__."
    )
    non_callable = [
        name for name in _REQUIRED_ATTRS if not callable(getattr(compat, name))
    ]
    assert not non_callable, (
        f"gonka_poc._compat.current() module exposes non-callable attrs: "
        f"{non_callable}. Every compat hook must be a callable."
    )


def test_current_optional_attrs_are_callable_if_present() -> None:
    """Optional attrs (e.g. ``install_model_runner_forward_hook``) — if the
    compat submodule re-introduces them, they MUST still be callable so
    callers can dispatch uniformly."""
    compat = _resolve_compat_module()
    for name in _OPTIONAL_ATTRS:
        if hasattr(compat, name):
            assert callable(getattr(compat, name)), (
                f"Optional compat attr {name!r} is present but not callable."
            )


def test_current_is_cached() -> None:
    """Repeated ``current()`` calls MUST return the same module object.

    The dispatch implementation uses ``lru_cache`` for this; we verify the
    observable contract rather than the implementation detail.
    """
    assert _resolve_compat_module() is _resolve_compat_module(), (
        "gonka_poc._compat.current() must return the same module instance "
        "on repeated calls (otherwise dispatch is re-importing per use)."
    )


def test_poc_model_runner_imports_cleanly() -> None:
    """Importing ``gonka_poc.poc.poc_model_runner`` MUST NOT crash.

    The runner imports ``current as _current_compat`` at module load; a
    future refactor that re-binds ``current`` to something un-callable at
    *import* time (rather than at first call) would otherwise only surface
    on real hardware during the first PoC forward. Catch it here in 0.5s.
    """
    import importlib

    mod = importlib.import_module("gonka_poc.poc.poc_model_runner")
    # Sanity: the runner exposes *something* — we don't care what, only
    # that import didn't raise.
    assert mod is not None
