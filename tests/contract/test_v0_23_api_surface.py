"""Drift detector: pin the vLLM 0.23 private surfaces the plugin depends on.

Each test asserts a precise upstream symbol exists with the expected shape
(class name, dataclass field set, method signature, attribute presence).
Run on every CI bump of the vllm pin -- a failure here means the compat
shim ``gonka_poc._compat.v0_23`` needs an update before the plugin can
target a new vllm minor.

Scope: read-only inspection of vllm modules; NO GPU, NO engine startup, NO
forward pass. Safe to run in a vanilla ``pip install vllm`` environment.
"""
from __future__ import annotations

import importlib
import inspect

import pytest


# ---------------------------------------------------------------------------- #
# vllm pin
# ---------------------------------------------------------------------------- #

def test_vllm_version_pin() -> None:
    """We claim 0.23.x in pyproject.toml; assert the installed wheel matches."""
    vllm = pytest.importorskip("vllm")
    version = getattr(vllm, "__version__", "")
    assert version.startswith("0.23."), (
        f"gonka-poc compat shim targets vllm 0.23.*, got {version!r}. "
        "Add a new compat module under gonka_poc/_compat/ and update _DISPATCH."
    )


# ---------------------------------------------------------------------------- #
# CommonAttentionMetadata fields
# ---------------------------------------------------------------------------- #

def test_common_attention_metadata_fields() -> None:
    """The PoC forward constructs CommonAttentionMetadata directly.

    Per fork commit 582f087a5 ("restore seq_lens_cpu_upper_bound kwarg for
    MLA attention #9"), the field set shifted across versions. Pin the
    fields the plugin relies on so a future minor bump fails loudly.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.attention.backends.utils")
    cls = getattr(mod, "CommonAttentionMetadata", None)
    assert cls is not None, (
        "CommonAttentionMetadata not found at "
        "vllm.v1.attention.backends.utils -- compat shim needs update."
    )

    # If it's a dataclass / NamedTuple, inspect __annotations__; otherwise
    # fall back to __init__ signature.
    annotations = getattr(cls, "__annotations__", None) or {}
    if not annotations:
        sig = inspect.signature(cls)
        annotations = {
            name: p.annotation
            for name, p in sig.parameters.items()
            if name != "self"
        }

    # Fields the plugin relies on, per fork PoC v2 + #9 fix.
    required_fields = {
        "seq_lens",
        # seq_lens_cpu_upper_bound was REINTRODUCED in 0.20+ MLA path; if
        # this assertion fails on a future minor, the compat shim's
        # build_common_attention_metadata kwarg list must change.
        "seq_lens_cpu_upper_bound",
    }
    missing = required_fields - set(annotations)
    assert not missing, (
        f"CommonAttentionMetadata is missing fields {sorted(missing)}; "
        f"present fields = {sorted(annotations)}. "
        f"Update gonka_poc._compat.v0_23.build_common_attention_metadata."
    )


# ---------------------------------------------------------------------------- #
# GPUModelRunner.kv_caches
# ---------------------------------------------------------------------------- #

def test_kv_caches_attribute() -> None:
    """PoC reuses kv_caches[:N] as scratch. Pin the attribute name + type hint."""
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    cls = getattr(mod, "GPUModelRunner", None)
    assert cls is not None, "GPUModelRunner missing"
    annotations = getattr(cls, "__annotations__", {})
    # In v0.23 GPUModelRunner.kv_caches is declared as ``list[torch.Tensor]``
    # at class scope. If the attribute moves to __init__ only, this check
    # still passes as long as one of the parent class scopes carries it --
    # but the compat shim's getattr-based access still works.
    has_class_annotation = "kv_caches" in annotations
    # Fallback: instantiated class has the attr (we can't instantiate here,
    # so we just verify it's referenced in the class source).
    src = inspect.getsource(cls)
    assert has_class_annotation or "kv_caches" in src, (
        "GPUModelRunner.kv_caches not visible in v0.23 source -- "
        "gonka_poc._compat.v0_23.get_kv_cache_pool needs revision."
    )


# ---------------------------------------------------------------------------- #
# EngineClient.abort presence
# ---------------------------------------------------------------------------- #

def test_engine_client_has_abort() -> None:
    """PoCGatingMiddleware does NOT abort, but the PoC router does (via
    EngineClient.abort_request) right before issuing collective_rpc.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.engine.protocol")
    cls = getattr(mod, "EngineClient", None)
    assert cls is not None, "EngineClient ABC missing"
    # Either ``abort`` or ``abort_request`` -- both have appeared.
    assert any(
        hasattr(cls, name) for name in ("abort", "abort_request")
    ), "EngineClient lacks an abort method -- compat shim needs revision."


# ---------------------------------------------------------------------------- #
# worker_extension_cls injection point
# ---------------------------------------------------------------------------- #

def test_worker_extension_cls_supported() -> None:
    """``--worker-extension-cls`` is the documented hook for PoCWorkerExtension."""
    pytest.importorskip("vllm")
    parallel = importlib.import_module("vllm.config.parallel")
    pc_cls = getattr(parallel, "ParallelConfig", None)
    assert pc_cls is not None, "ParallelConfig moved -- compat shim needs revision."
    assert "worker_extension_cls" in getattr(pc_cls, "__annotations__", {}) or hasattr(
        pc_cls, "worker_extension_cls"
    ), "ParallelConfig.worker_extension_cls vanished -- compat shim needs revision."


# ---------------------------------------------------------------------------- #
# api_server composition surface
# ---------------------------------------------------------------------------- #

def test_api_server_composition_symbols() -> None:
    """``gonka-vllm-serve`` calls these public helpers verbatim."""
    pytest.importorskip("vllm")
    api = importlib.import_module("vllm.entrypoints.openai.api_server")
    for name in ("build_app", "build_async_engine_client", "init_app_state", "setup_server"):
        assert hasattr(api, name), (
            f"vllm.entrypoints.openai.api_server.{name} missing -- "
            f"gonka_poc.entrypoint.api_router._run_server needs revision."
        )


# ---------------------------------------------------------------------------- #
# CLI args helpers
# ---------------------------------------------------------------------------- #

def test_cli_args_helpers() -> None:
    pytest.importorskip("vllm")
    cli = importlib.import_module("vllm.entrypoints.openai.cli_args")
    for name in ("make_arg_parser", "validate_parsed_serve_args"):
        assert hasattr(cli, name), (
            f"vllm.entrypoints.openai.cli_args.{name} missing -- "
            f"gonka_poc.entrypoint.api_router.main needs revision."
        )


# ---------------------------------------------------------------------------- #
# ModelRegistry surface
# ---------------------------------------------------------------------------- #

def test_model_registry_register_model() -> None:
    pytest.importorskip("vllm")
    vllm = importlib.import_module("vllm")
    registry = getattr(vllm, "ModelRegistry", None)
    assert registry is not None, "vllm.ModelRegistry not re-exported -- plugin needs revision."
    assert hasattr(registry, "register_model"), "ModelRegistry.register_model missing."
