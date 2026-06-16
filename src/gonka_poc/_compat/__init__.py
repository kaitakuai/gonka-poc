"""Version-dispatched compat shim isolating private vLLM API touchpoints.

The plugin is officially pinned to vllm>=0.23.0,<0.24 in ``pyproject.toml``.
Within that range some private structures (CommonAttentionMetadata,
kv_caches layout, internal model_runner attributes) may shift; we localise
every such touchpoint inside one of the ``vN_M.py`` modules in this package.

Selection happens once at import time. The selected module is exposed as
``gonka_poc._compat.current``::

    from gonka_poc._compat import current as compat
    md = compat.build_common_attention_metadata(...)

Add a new ``vN_M.py`` AND register it in :data:`_DISPATCH` when porting to a
new vLLM minor; keep contract tests (``tests/contract/test_v*_api_surface.py``)
in lock-step.
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Callable, Mapping, Tuple


def _parse_version(v: str) -> Tuple[int, ...]:
    parts: list[int] = []
    for chunk in v.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        if len(parts) >= 3:
            break
    return tuple(parts)


def _detect_vllm_version() -> Tuple[int, ...]:
    try:
        import vllm  # type: ignore
        return _parse_version(getattr(vllm, "__version__", "0.0.0"))
    except Exception:
        return (0, 0, 0)


# Map of (major, minor) -> module name within this package.
_DISPATCH: Mapping[Tuple[int, int], str] = {
    (0, 23): "gonka_poc._compat.v0_23",
}


def _select_module() -> ModuleType:
    major, minor, *_ = (_detect_vllm_version() + (0, 0, 0))[:3]
    mod_name = _DISPATCH.get((major, minor))
    if mod_name is None:
        # Fall back to v0.23 -- the only supported minor today. The contract
        # test in tests/contract/ will surface a real drift.
        mod_name = "gonka_poc._compat.v0_23"
    return importlib.import_module(mod_name)


current: ModuleType = _select_module()


__all__ = ["current"]
