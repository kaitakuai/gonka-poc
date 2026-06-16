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

import logging
import warnings
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- #
# Attention metadata (CommonAttentionMetadata)
# ---------------------------------------------------------------------------- #

def build_common_attention_metadata(
    *,
    query_start_loc: Any,
    query_start_loc_cpu: Any,
    seq_lens: Any,
    num_reqs: int,
    num_actual_tokens: int,
    max_query_len: int,
    max_seq_len: int,
    block_table_tensor: Any,
    slot_mapping: Any,
    causal: bool = True,
    seq_lens_cpu_upper_bound: Optional[Any] = None,
    _seq_lens_cpu: Optional[Any] = None,
    _num_computed_tokens_cpu: Optional[Any] = None,
) -> Any:
    """Construct a ``CommonAttentionMetadata`` for a PoC forward pass.

    Upstream symbol: ``vllm.v1.attention.backends.utils.CommonAttentionMetadata``
        (re-exported from ``vllm.v1.attention.backend``; private; constructor
        signature shifted in v0.20+ to add the ``seq_lens_cpu_upper_bound``
        kwarg required by MLA-style backends -- see fork commit 582f087a5
        "fix(poc): restore seq_lens_cpu_upper_bound kwarg for MLA attention
        (#9)").

    Version constraint: vllm == 0.23.*

    Contract test:
        tests/contract/test_v0_23_api_surface.py::test_common_attention_metadata_fields

    The kwarg list mirrors the fork's ``_create_v1_attn_metadata`` call site at
    ``vllm/poc/poc_model_runner.py`` (branch mb/feat/port-pocv2-vllm-0.23.0):

        CommonAttentionMetadata(
            query_start_loc=...,
            query_start_loc_cpu=...,
            seq_lens=...,
            num_reqs=batch_size,
            num_actual_tokens=batch_size * seq_len,
            max_query_len=seq_len,
            max_seq_len=seq_len,
            block_table_tensor=...,
            slot_mapping=...,
            causal=True,
            _seq_lens_cpu=...,
            seq_lens_cpu_upper_bound=...,
            _num_computed_tokens_cpu=torch.zeros(batch_size, ...),
        )

    Caller (poc_model_runner) is responsible for building the GPU/CPU tensors;
    this helper is the version-pinned constructor binding only.
    """
    # Import the real dataclass; resolves through utils to keep the contract
    # test's pin point (vllm.v1.attention.backends.utils.CommonAttentionMetadata)
    # authoritative.
    from vllm.v1.attention.backends.utils import CommonAttentionMetadata

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_reqs,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=causal,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
        _seq_lens_cpu=_seq_lens_cpu,
        _num_computed_tokens_cpu=_num_computed_tokens_cpu,
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

def _list_in_flight_request_ids(engine_client: Any) -> list[str]:
    """Best-effort enumeration of in-flight request IDs on the EngineClient.

    v0.23 ``EngineClient`` ABC (vllm/engine/protocol.py) does NOT expose a
    ``list_requests`` / ``get_active_requests`` method. The concrete
    ``AsyncLLM`` impl (vllm/v1/engine/async_llm.py) keeps state on its
    ``output_processor.request_states`` dict[str, RequestState] -- that is
    the only stable hook in v0.23.

    Returns [] if no enumerable surface is available; caller then logs and
    relies on PoCGatingMiddleware to gate new requests instead.
    """
    # Path 1: AsyncLLM output_processor (v1 engine).
    op = getattr(engine_client, "output_processor", None)
    if op is not None:
        states = getattr(op, "request_states", None)
        if isinstance(states, dict):
            # Snapshot keys so concurrent mutation during abort doesn't raise.
            return list(states.keys())
    # Path 2: legacy v0 engine (kept for defensive fallback; should not hit
    # in a 0.23-only deployment but harmless if a wrapped client appears).
    engine = getattr(engine_client, "engine", None)
    if engine is not None:
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None:
            # scheduler may be a list (per-vp) or a single instance.
            schedulers = scheduler if isinstance(scheduler, list) else [scheduler]
            ids: list[str] = []
            for sch in schedulers:
                for queue_name in ("running", "waiting", "swapped"):
                    queue = getattr(sch, queue_name, None) or []
                    for seq_group in queue:
                        rid = getattr(seq_group, "request_id", None)
                        if rid is not None:
                            ids.append(str(rid))
            return ids
    return []


async def abort_all_requests(engine_client: Any) -> int:
    """Abort every in-flight inference request before PoC seizes the GPU.

    Upstream symbol: ``vllm.engine.protocol.EngineClient.abort`` (ABC method;
        ``async def abort(request_id: str | Iterable[str]) -> None``).

    Version constraint: vllm == 0.23.*

    Contract test:
        tests/contract/test_v0_23_api_surface.py::test_engine_client_has_abort

    Returns: number of requests aborted (best-effort).

    CALLER NOTE: v0.23 ``EngineClient`` ABC has no public ``list_requests``
    API. We introspect the concrete ``AsyncLLM.output_processor.request_states``
    dict. If that surface vanishes in a future minor, this function returns 0
    and logs a warning -- PoCGatingMiddleware alone is sufficient to gate new
    requests (this abort path is belt-and-suspenders for already-admitted
    requests still mid-decode).
    """
    abort_fn = getattr(engine_client, "abort", None)
    if abort_fn is None:
        logger.warning(
            "abort_all_requests: engine_client has no .abort method "
            "(expected per EngineClient ABC v0.23) -- skipping."
        )
        return 0

    request_ids = _list_in_flight_request_ids(engine_client)
    if not request_ids:
        logger.warning(
            "abort_all_requests: no enumerable in-flight requests on "
            "engine_client (output_processor.request_states unavailable). "
            "PoCGatingMiddleware still blocks new admissions."
        )
        return 0

    aborted = 0
    for rid in request_ids:
        try:
            # EngineClient.abort accepts a single id or an iterable; we issue
            # per-id calls so one bad id does not abort the rest of the loop.
            await abort_fn(rid)
            aborted += 1
        except Exception as exc:  # noqa: BLE001 -- best-effort, never raise
            logger.warning(
                "abort_all_requests: abort(%s) failed: %s", rid, exc
            )
    return aborted


# ---------------------------------------------------------------------------- #
# Model runner forward bind (DEFERRED-BY-DESIGN)
# ---------------------------------------------------------------------------- #

def install_model_runner_forward_hook(model_runner: Any, hook: Any) -> None:
    """Deprecated no-op -- kept for import compatibility.

    The PoC v2 architecture on 0.23 invokes ``PoCWorkerExtension.execute_poc_forward``
    out-of-band via ``collective_rpc``; it does NOT intercept the per-step
    serving forward. This hook is therefore deferred-by-design.

    If a future architecture revision needs per-step interception, restore an
    implementation that rebinds ``GPUModelRunner.execute_model`` and add a
    contract test (``tests/contract/test_v0_23_api_surface.py::
    test_execute_model_signature``) that pins the upstream signature.
    """
    warnings.warn(
        "install_model_runner_forward_hook is a deferred-by-design no-op on "
        "vllm 0.23.x (PoC v2 uses out-of-band collective_rpc, not per-step "
        "forward interception). Calling this function has no effect.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None


__all__ = [
    "build_common_attention_metadata",
    "get_kv_cache_pool",
    "abort_all_requests",
]
