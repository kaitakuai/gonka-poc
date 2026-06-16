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


# ---------------------------------------------------------------------------- #
# Distributed group getters
# ---------------------------------------------------------------------------- #

def test_distributed_groups_present() -> None:
    """PoCWorkerExtension uses these for the inter-rank PoC dispatch flow.

    ``get_tp_group`` / ``get_pp_group`` live in
    ``vllm.distributed.parallel_state`` and are re-exported from the package
    root via ``vllm.distributed`` (``from .parallel_state import *``). The
    plugin imports the short path -- pin both surfaces so a re-shuffle that
    drops the re-export breaks here, not at runtime.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.distributed")
    for name in ("get_pp_group", "get_tp_group"):
        fn = getattr(mod, name, None)
        assert fn is not None and callable(fn), (
            f"vllm.distributed.{name} missing or non-callable -- "
            f"gonka_poc._compat.v0_23 must reroute via "
            f"vllm.distributed.parallel_state."
        )
        # Both are zero-arg getters returning a GroupCoordinator. If a
        # future minor introduces required args, the compat shim must wrap.
        sig = inspect.signature(fn)
        required = [
            p for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        assert not required, (
            f"vllm.distributed.{name} acquired required args {required!r}; "
            f"PoCWorkerExtension call site needs revision."
        )


# ---------------------------------------------------------------------------- #
# broadcast_tensor_dict fast-path
# ---------------------------------------------------------------------------- #

def test_communication_op_broadcast() -> None:
    """Optional fast-path for PoC payload broadcast across the TP group.

    Absence here means PoCWorkerExtension must fall back to
    ``get_tp_group().broadcast_tensor_dict(...)`` directly. We still pin the
    free-function form so a future rename forces a compat-shim review.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.distributed.communication_op")
    fn = getattr(mod, "broadcast_tensor_dict", None)
    assert fn is not None and callable(fn), (
        "vllm.distributed.communication_op.broadcast_tensor_dict missing -- "
        "compat shim needs update (fallback: get_tp_group().broadcast_tensor_dict)."
    )
    sig = inspect.signature(fn)
    # In v0.23 the signature is (tensor_dict=None, src=0). The plugin passes
    # both positionally; pin the parameter names to catch a kwarg rename.
    params = list(sig.parameters)
    assert params[:2] == ["tensor_dict", "src"], (
        f"broadcast_tensor_dict signature changed: {params!r}; "
        f"compat shim needs update."
    )


# ---------------------------------------------------------------------------- #
# set_forward_context contextmanager
# ---------------------------------------------------------------------------- #

def test_forward_context_set() -> None:
    """PoC forward needs to enter a forward context bearing AttentionMetadata.

    ``vllm.forward_context.set_forward_context`` is a ``@contextmanager``
    accepting ``(attn_metadata, vllm_config, ...)``. Pin both required
    parameters so a future split (e.g., into ``begin_forward`` /
    ``end_forward``) is caught here.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.forward_context")
    fn = getattr(mod, "set_forward_context", None)
    assert fn is not None and callable(fn), (
        "vllm.forward_context.set_forward_context missing -- "
        "compat shim needs update."
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    for required in ("attn_metadata", "vllm_config"):
        assert required in params, (
            f"set_forward_context lost parameter {required!r}; "
            f"present = {params!r}; compat shim needs update."
        )


# ---------------------------------------------------------------------------- #
# IntermediateTensors (PP>1 PoC)
# ---------------------------------------------------------------------------- #

def test_intermediate_tensors() -> None:
    """PP > 1 PoC may pass / receive IntermediateTensors between stages.

    In v0.23 it lives at ``vllm.sequence.IntermediateTensors`` as a
    ``@dataclass`` with a single field ``tensors: dict[str, torch.Tensor]``.
    If a future minor relocates it (likely candidates:
    ``vllm.v1.outputs`` or ``vllm.v1.sequence``), this test fails and the
    compat shim must add a fallback import path.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.sequence")
    cls = getattr(mod, "IntermediateTensors", None)
    assert cls is not None, (
        "vllm.sequence.IntermediateTensors missing -- "
        "try vllm.v1.outputs or vllm.v1.sequence and update the compat shim."
    )
    # Pin the dataclass field name; a rename of ``tensors`` would break the
    # PP-stage hand-off in PoCWorkerExtension.
    annotations = getattr(cls, "__annotations__", {}) or {}
    assert "tensors" in annotations, (
        f"IntermediateTensors field set changed: {sorted(annotations)!r}; "
        f"compat shim needs update."
    )


# ---------------------------------------------------------------------------- #
# launcher.serve_http
# ---------------------------------------------------------------------------- #

def test_launcher_serve_http() -> None:
    """``gonka-vllm-serve`` composition calls ``serve_http`` directly.

    Confirmed at ``vllm.entrypoints.launcher.serve_http`` in v0.23 -- an
    ``async def`` taking ``(app, sock, enable_ssl_refresh=False, **uvicorn_kwargs)``.
    If a future minor inlines it back into ``api_server`` or renames it to
    ``_serve_http``, this test fires.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.entrypoints.launcher")
    fn = getattr(mod, "serve_http", None)
    assert fn is not None and callable(fn), (
        "vllm.entrypoints.launcher.serve_http missing -- "
        "gonka_poc.entrypoint.api_router._run_server needs revision."
    )
    assert inspect.iscoroutinefunction(fn), (
        "serve_http is no longer a coroutine -- await-site needs revision."
    )
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    # The plugin passes `app` and `sock` positionally; pin the leading pair.
    assert params[:2] == ["app", "sock"], (
        f"serve_http signature drifted: {params!r}; compat shim needs update."
    )


# ---------------------------------------------------------------------------- #
# SamplingParams residual-fork bridge (logprobs_mode + enforced_token_ids)
# ---------------------------------------------------------------------------- #

def test_sampling_params_has_fork_patches() -> None:
    """Pin the residual-fork SamplingParams fields the plugin depends on.

    Background -- the engine ships as ``vllm==0.23.0+gonka.sampler1``: a
    residual fork of upstream 0.23.0 carrying 6 sampler patches
    (poc-sampler-residual-v0.23). Two of those patches add per-request
    ``logprobs_mode`` and ``enforced_token_ids`` fields to SamplingParams;
    the PoC v2 mixed-mode sampling path (validator replay + logits-mode
    selection) reads them at request admission.

    Why this test belongs in the PLUGIN contract suite (not just the fork):
        The plugin advertises ``vllm>=0.23.0,<0.24`` as its install pin.
        ``pip install vllm==0.23.0`` (vanilla, no ``+gonka.sampler1``)
        satisfies that pin -- and the other contract tests stay GREEN
        against vanilla vllm 0.23, but engine startup crashes the moment
        a request with ``logprobs_mode`` arrives. This pin catches that
        misconfiguration BEFORE production.

    Pattern: same as the residual branch's
    ``tests/contract/test_sampler_surface.py::test_sampling_params_has_poc_fields``.
    SamplingParams is a ``msgspec.Struct`` in v0.23, so we read
    ``__struct_fields__`` first; we fall back to ``__annotations__`` to
    stay forward-compatible with a future dataclass conversion.
    """
    pytest.importorskip("vllm")
    mod = importlib.import_module("vllm.sampling_params")
    cls = getattr(mod, "SamplingParams", None)
    assert cls is not None, "vllm.sampling_params.SamplingParams missing"

    # Primary: msgspec.Struct exposes the field tuple via __struct_fields__.
    fields: set = set(getattr(cls, "__struct_fields__", ()) or ())
    # Fallback chain: dataclass / NamedTuple / annotated class.
    if not fields:
        fields = set(getattr(cls, "__annotations__", {}) or {})
    if not fields:
        sig = inspect.signature(cls.__init__)
        fields = {
            name for name, p in sig.parameters.items()
            if name != "self"
        }

    for required in ("logprobs_mode", "enforced_token_ids"):
        assert required in fields, (
            f"SamplingParams.{required} missing -- the installed vllm wheel "
            f"is NOT the residual fork (0.23.0+gonka.sampler1). The plugin "
            f"contract tests stay green against vanilla 0.23 but the engine "
            f"WILL crash at request time. Install via "
            f"`pip install vllm==0.23.0+gonka.sampler1` (or the residual "
            f"wheel built from branch poc-sampler-residual-v0.23)."
        )


# ---------------------------------------------------------------------------- #
# OpenAIServingChat export (RESTRUCTURED in v0.23)
# ---------------------------------------------------------------------------- #

def test_openai_serving_chat_export() -> None:
    """PoC gating middleware needs an OpenAIServingChat handle to abort
    in-flight requests.

    NOTE -- v0.23 restructure: ``vllm.entrypoints.openai.serving_chat`` no
    longer exists. The class moved to
    ``vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat``.
    The ``chat_completion`` package's ``__init__.py`` does NOT re-export
    the symbol, so the compat shim MUST import from the deep path.
    """
    pytest.importorskip("vllm")

    # The pre-0.23 path is gone -- assert that explicitly so a future
    # vllm bump that *restores* the alias surfaces here and lets us
    # simplify the compat shim.
    try:
        legacy = importlib.import_module("vllm.entrypoints.openai.serving_chat")
    except ImportError:
        legacy = None
    if legacy is not None and hasattr(legacy, "OpenAIServingChat"):
        # Legacy path restored -- compat shim can drop the deep-path fallback.
        return

    mod = importlib.import_module("vllm.entrypoints.openai.chat_completion.serving")
    cls = getattr(mod, "OpenAIServingChat", None)
    assert cls is not None and inspect.isclass(cls), (
        "OpenAIServingChat not found at "
        "vllm.entrypoints.openai.chat_completion.serving -- "
        "the compat shim must search a new path "
        "(check vllm.entrypoints.openai.chat_completion.__init__ for re-exports)."
    )
    # Pin a method the gating middleware depends on for response handling.
    # ``create_chat_completion`` is the documented entry point; if a future
    # version renames it (e.g., ``handle_chat_request``), the middleware
    # wrapper must follow.
    assert hasattr(cls, "create_chat_completion"), (
        "OpenAIServingChat.create_chat_completion missing -- "
        "gonka_poc gating middleware needs revision."
    )
