"""Compat shim for vLLM 0.23.x.

Every function below touches a vLLM private surface. Each docstring records:
  * the upstream symbol (with file path + line if stable);
  * the version constraint we're claiming;
  * the contract-test reference that must stay green.

If any of these shift in a future vLLM minor, copy this file to
``v0_24.py``, edit the relevant function, and register the new dispatch
mapping in ``gonka_poc/_compat/__init__.py``.

CommonAttentionMetadata import-path policy
------------------------------------------
v0.23 exposes ``CommonAttentionMetadata`` at TWO paths:

  * ``vllm.v1.attention.backends.utils`` — canonical declaration site.
  * ``vllm.v1.attention.backend``       — convenience re-export.

We import from the canonical ``vllm.v1.attention.backends.utils`` path
because that is the one pinned by
``tests/contract/test_v0_23_api_surface.py::test_common_attention_metadata_fields``
(the contract test verifies the field set at the declaration site, so the
shim and the test must reference the same path; otherwise an upstream
re-export removal would break the shim silently while the contract test
stays green).
"""
from __future__ import annotations

import logging
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
    positions: Optional[Any] = None,
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
        positions=positions,
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
# Per-group AttentionMetadata expansion (model_runner.attn_groups iteration)
# ---------------------------------------------------------------------------- #

def build_attn_metadata_per_group(
    model_runner: Any,
    *,
    layout_for_block_size: Any,
    common_metadata_for_layout: Any,
) -> tuple[dict, dict]:
    """Build per-layer ``AttentionMetadata``, one build per attention group.

    Models may register KV cache groups with DIFFERENT block sizes (e.g.
    DeepSeek-V4: sparse MLA / indexer at ``cache_config.block_size``, the SWA
    compressor at 8). Each group's metadata must therefore be built from a
    layout computed for THAT group's block size — sharing one layout hands
    out-of-range slot ids to the other pools (OOB writes: illegal memory
    access on sm_90, silent corruption elsewhere). For single-group models
    the loop reproduces the historical single-layout behaviour exactly.

    Upstream symbols (all version-pinned here, callers stay generic):
        * ``GPUModelRunner.attn_groups`` (list[list[AttentionGroup]]),
          vllm/v1/worker/gpu_model_runner.py (v0.23.0)
        * ``AttentionGroup.get_metadata_builder(index)`` and
          ``AttentionGroup.layer_names`` / ``.kv_cache_spec``
          (vllm/v1/worker/utils.py)
        * ``builder.kv_cache_spec`` — preferred block-size source because it
          reflects ``kernel_block_size`` splits
          (``KVCacheSpec.copy_with_new_block_size``); the group spec is the
          fallback. A literal 0 is treated as present (explicit ``is None``
          checks), even though no current spec produces it.
        * ``builder.build(common_prefix_len, common_attn_metadata)`` — the v1
          entry point for materialising backend-specific metadata.

    Version constraint: vllm == 0.23.*

    Contract test:
        tests/contract/test_v0_23_api_surface.py::test_kv_caches_attribute
        (covers the GPUModelRunner declaration site; attn_groups lives in
        the same class so any reshuffle that breaks one breaks the other)

    Args:
        model_runner: live ``GPUModelRunner`` instance from the worker.
        layout_for_block_size: callable ``(block_size:int) ->
            (slot_mapping, block_table)``; the caller owns the tensor math
            and may cache per block size.
        common_metadata_for_layout: callable ``(slot_mapping, block_table) ->
            CommonAttentionMetadata`` (normally a partial application of
            :func:`build_common_attention_metadata`).

    Returns:
        ``(attn_metadata_dict, slot_mapping_dict)`` — both keyed by layer
        name; the slot mapping for each layer is its group's. Pass straight
        into ``set_forward_context(attn_metadata=..., slot_mapping=...)``.
    """
    attn_metadata_dict: dict = {}
    slot_mapping_dict: dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            spec = getattr(builder, "kv_cache_spec", None)
            block_size = getattr(spec, "block_size", None)
            if block_size is None:
                block_size = attn_group.kv_cache_spec.block_size
            slot_mapping, block_table = layout_for_block_size(block_size)
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_metadata_for_layout(
                    slot_mapping, block_table
                ),
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


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


def unlock_moe_workspace() -> bool:
    """Unlock the v0.23 lockable MoE ``WorkspaceManager`` so the PoC forward can
    grow it past the inference-shaped locked size.

    The private ``vllm.v1.worker.workspace`` touchpoint lives here (the only
    place allowed to reach into ``vllm.v1.*``). Returns ``True`` if a manager
    was unlocked, ``False`` if there is no active manager (non-MoE model, where
    ``current_workspace_manager()`` asserts) — the caller then skips the
    re-lock.
    """
    from vllm.v1.worker.workspace import unlock_workspace

    try:
        unlock_workspace()
        return True
    except Exception:  # no MoE workspace manager (non-MoE model)
        return False


def lock_moe_workspace() -> None:
    """Re-lock the v0.23 MoE ``WorkspaceManager`` after the PoC forward.

    Only called by the context manager when :func:`unlock_moe_workspace`
    returned ``True`` (so a manager exists); failures are surfaced to the caller.
    """
    from vllm.v1.worker.workspace import lock_workspace

    lock_workspace()


__all__ = [
    "build_common_attention_metadata",
    "build_attn_metadata_per_group",
    "get_kv_cache_pool",
    "abort_all_requests",
    "unlock_moe_workspace",
    "lock_moe_workspace",
]
