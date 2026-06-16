"""Compat shim for vLLM 0.23.x.

Every function below touches a vLLM private surface. Each docstring records:
  * the upstream symbol (with file path + line if stable);
  * the version constraint we're claiming;
  * the contract-test reference that must stay green.

If any of these shift in a future vLLM minor, copy this file to
``v0_24.py``, edit the relevant function, and register the new dispatch
mapping in ``gonka_poc/_compat/__init__.py``.
"""
from __future__ import annotations

from typing import Any, Optional


# ---------------------------------------------------------------------------- #
# Attention metadata (CommonAttentionMetadata)
# ---------------------------------------------------------------------------- #

def build_common_attention_metadata(
    *,
    model_runner: Any,
    seq_len: int,
    batch_size: int = 32,
    seq_lens_cpu_upper_bound: Optional[int] = None,
) -> Any:
    """Construct a ``CommonAttentionMetadata`` for a PoC forward pass.

    Upstream symbol: ``vllm.v1.attention.backends.utils.CommonAttentionMetadata``
        (private; constructor signature shifted in v0.20+ to add the
        ``seq_lens_cpu_upper_bound`` kwarg required by MLA-style backends --
        see fork commit 582f087a5 "fix(poc): restore seq_lens_cpu_upper_bound
        kwarg for MLA attention (#9)").

    Version constraint: vllm == 0.23.*

    Contract test:
        tests/contract/test_v0_23_api_surface.py::test_common_attention_metadata_fields

    TODO(layer-2, hardware-validation): finalise the kwarg list once the
    contract test introspects the actual dataclass on the installed wheel.
    The fork's call site is in vllm/poc/poc_model_runner.py
    ``_create_v1_attn_metadata`` -- mirror its signature here.
    """
    # TODO: import and call the real CommonAttentionMetadata once contract
    # test asserts the field set.
    #
    #     from vllm.v1.attention.backends.utils import CommonAttentionMetadata
    #     return CommonAttentionMetadata(
    #         seq_lens=...,
    #         seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound or seq_len,
    #         ...
    #     )
    raise NotImplementedError(
        "build_common_attention_metadata: stub. Implementation deferred "
        "until tests/contract/test_v0_23_api_surface.py pins the dataclass "
        "field set on the installed vllm wheel."
    )


# ---------------------------------------------------------------------------- #
# KV cache pool access
# ---------------------------------------------------------------------------- #

def get_kv_cache_pool(model_runner: Any) -> list:
    """Return the worker's KV-cache tensor list.

    Upstream symbol: ``GPUModelRunner.kv_caches`` (list[torch.Tensor])
        declared at vllm/v1/worker/gpu_model_runner.py:525 (v0.23.0).

    Version constraint: vllm == 0.23.*

    Contract test:
        tests/contract/test_v0_23_api_surface.py::test_kv_caches_attribute

    The PoC forward reuses blocks starting at index 0 as scratch space; the
    99a372d4e fork commit ("safer kv cache reuse") added dtype/contiguity
    checks. That logic lives in gonka_poc.poc.poc_model_runner; this helper
    is just the access point.
    """
    kv = getattr(model_runner, "kv_caches", None)
    if kv is None:
        raise RuntimeError(
            "model_runner.kv_caches is not populated -- worker initialise_kv_caches() "
            "must run before PoC forward. See vllm/v1/worker/gpu_worker.py."
        )
    return kv


# ---------------------------------------------------------------------------- #
# Engine client abort (used by gating to stop in-flight inference)
# ---------------------------------------------------------------------------- #

async def abort_all_requests(engine_client: Any) -> int:
    """Abort every in-flight inference request before PoC seizes the GPU.

    Upstream symbol: ``vllm.engine.protocol.EngineClient.abort`` (public-ish;
        method signature ``async def abort(request_id: str | Iterable[str])``).

    Version constraint: vllm == 0.23.*

    Contract test:
        tests/contract/test_v0_23_api_surface.py::test_engine_client_has_abort

    Returns number of requests aborted (best-effort).

    TODO(layer-1): list active requests on the EngineClient first -- in
    v0.23 the API is ``engine_client.list_requests()`` or
    ``engine_client.get_lora_requests()`` depending on minor; contract test
    pins the symbol.
    """
    # Stub: real implementation iterates app.state.openai_serving_chat /
    # _completion in-flight request_ids and calls engine_client.abort(...).
    return 0


# ---------------------------------------------------------------------------- #
# Model runner forward bind (one-shot install)
# ---------------------------------------------------------------------------- #

def install_model_runner_forward_hook(model_runner: Any, hook: Any) -> None:
    """Rebind ``model_runner.execute_model`` or ``.model.forward`` so PoC
    intercepts the live serving forward, if needed.

    Upstream symbol: ``vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model``
        (private; per-step entry from MultiprocExecutor).

    Version constraint: vllm == 0.23.*

    Contract test:
        tests/contract/test_v0_23_api_surface.py::test_execute_model_signature

    NOTE: this is only required if PoC needs to intercept the per-step
    forward. The default path (PoCWorkerExtension.execute_poc_forward
    invoked out-of-band via collective_rpc) does NOT need this hook.
    """
    # TODO(layer-2): implement once we decide whether PoC v2 on 0.23 still
    # needs per-step interception or can live entirely off the collective_rpc
    # entry point.
    raise NotImplementedError(
        "install_model_runner_forward_hook: TODO -- only needed if PoC "
        "intercepts the live serving forward in addition to the "
        "out-of-band collective_rpc path."
    )


__all__ = [
    "build_common_attention_metadata",
    "get_kv_cache_pool",
    "abort_all_requests",
    "install_model_runner_forward_hook",
]
