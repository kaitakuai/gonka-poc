"""Version-dispatched compat shim isolating private vLLM API touchpoints.

The plugin is officially pinned to vllm>=0.23.0,<0.24 in ``pyproject.toml``.
Within that range some private structures (CommonAttentionMetadata,
kv_caches layout, internal model_runner attributes) may shift; we localise
every such touchpoint inside one of the ``vN_M.py`` modules in this package.

Selection happens LAZILY -- the first call to :func:`current` (or the first
attribute lookup on the package via PEP 562 ``__getattr__``) imports the
matching submodule. This keeps test / lint discovery from forcing a real
``import vllm`` at module load time.

Usage::

    from gonka_poc._compat import current as compat
    md = compat().build_common_attention_metadata(...)

Or via PEP 562 attribute access::

    from gonka_poc import _compat
    md = _compat.current.build_common_attention_metadata(...)

Add a new ``vN_M.py`` AND register it in :data:`_DISPATCH` when porting to a
new vLLM minor; keep contract tests (``tests/contract/test_v*_api_surface.py``)
in lock-step.
"""
from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Mapping, Tuple


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


@lru_cache(maxsize=1)
def current() -> ModuleType:
    """Return the compat submodule matching the installed vllm minor.

    Cached so repeat callers don't re-import. Raises ``RuntimeError`` with the
    list of supported versions when the installed vllm minor is not in
    :data:`_DISPATCH` -- we explicitly do NOT silently fall back to the
    latest-known shim, because contract drift across minors can silently
    corrupt PoC outputs.
    """
    major, minor, *_ = (_detect_vllm_version() + (0, 0, 0))[:3]
    mod_name = _DISPATCH.get((major, minor))
    if mod_name is None:
        supported = ", ".join(
            f"{m}.{n}" for (m, n) in sorted(_DISPATCH.keys())
        )
        raise RuntimeError(
            f"gonka_poc._compat: vllm {major}.{minor} is not supported. "
            f"Supported vllm minors: {supported}. "
            "Add a new gonka_poc/_compat/vN_M.py and register it in _DISPATCH."
        )
    return importlib.import_module(mod_name)


def __getattr__(name: str) -> object:
    """PEP 562 hook so ``gonka_poc._compat.current`` (attribute style) and
    ``gonka_poc._compat.<symbol>`` both resolve through :func:`current`
    without triggering ``import vllm`` at module load time.
    """
    if name == "current":
        return current()
    # Fall through for any other attribute the user may add to a submodule.
    raise AttributeError(f"module 'gonka_poc._compat' has no attribute {name!r}")


__all__ = ["current"]
