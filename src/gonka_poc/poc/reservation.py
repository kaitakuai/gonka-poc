"""KV block reservation for PoC validation — inference keeps running.

Port of gonka-ai/vllm branch ``qd/combine-poc-and-inference`` into the
plugin architecture — commits, design, and rationale in ADR-0015.

Multi-group ("shared lease"): the block-id namespace is pool-global, so one
lease of ``num_nonces * max_g ceil(seq_len/block_size_g)`` blocks reserves
every group's tensors at once; the engine core sizes the stripe, the worker
expands ids per group (ADR-0015 Decisions 1-3,
``poc_model_runner._borrowed_layout``).

Safety contract:
    * Borrowed path: the only KV bytes the PoC forward writes belong to
      leased blocks (``ref_cnt`` held by the lease) — disjoint from every
      live request and from the prefix cache (``get_new_blocks`` evicts
      cached hashes on lease).
    * Legacy fallback (no lease obtainable): inference is ABORTED first —
      restoring the abort-before-overwrite invariant that the plugin's
      ``/generate`` path silently lacked — and the prefix cache is reset
      on exit, because blocks ``0..N`` were overwritten without hash
      invalidation (a prefix hit would otherwise serve poisoned KV).

Concurrency: the FIFO ``asyncio.Lock`` serializes validations in this
API-server process, so queued validations WAIT instead of each tripping
the abort-inference escalation, and the peak KV lease stays 1×. With
``--api-server-count > 1`` the bound degrades to P×; a cross-process cap
would have to live in EngineCore (the single pool owner). Single process
today. Mining (``/init/generate``) never takes this lock: its priority is
enforced by the callers' ``_is_generation_active`` spins.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional, Tuple

from gonka_poc._compat import current as _current_compat

logger = logging.getLogger(__name__)

# Total budget for obtaining a lease (poll + escalation re-poll each get
# one window). Matches the upstream default order of magnitude.
POC_BORROW_TIMEOUT_MS = int(os.getenv("POC_BORROW_TIMEOUT_MS", "3000"))

_BORROW_POLL_SEC = 0.05
_ABORT_SETTLE_SEC = 0.05

# FIFO per-process lock — see module docstring.
_poc_reservation_lock = asyncio.Lock()


async def _borrow_with_retry(
    engine_client: Any,
    num_nonces: int,
    seq_len: int,
    deadline: float,
) -> Tuple[Optional[dict], bool]:
    """Poll the engine core for a lease until ``deadline``.

    Returns ``(lease, rpc_broken)``. ``rpc_broken=True`` means the RPC
    surface itself failed (feature not installed / transport error) — the
    caller must NOT keep polling or escalating for a better answer.
    """
    compat = _current_compat()
    while True:
        try:
            lease = await compat.borrow_poc_blocks(
                engine_client, num_nonces, seq_len)
        except Exception as exc:
            logger.warning(
                "PoC borrow RPC failed (engine lacks the borrow methods?): %s",
                exc)
            return None, True
        if lease is not None:
            return lease, False
        if time.monotonic() > deadline:
            return None, False
        await asyncio.sleep(_BORROW_POLL_SEC)


async def reserve_poc_blocks(
    engine_client: Any,
    num_nonces: int,
    seq_len: int,
    timeout_ms: int = POC_BORROW_TIMEOUT_MS,
) -> Optional[dict]:
    """Lease KV blocks, escalating once through an inference abort.

    Order: (1) poll-borrow within ``timeout_ms``; (2) on failure abort all
    in-flight inference — this is BOTH the escalation (freed blocks make the
    re-borrow succeed) AND the safety precondition for the legacy fallback
    (which overwrites blocks ``0..N`` in place); (3) re-poll unless the RPC
    surface itself is broken. Returns the lease dict or ``None`` — by the
    time ``None`` is returned, inference has been aborted, so the caller may
    safely run the legacy in-place path.
    """
    deadline = time.monotonic() + timeout_ms / 1000.0
    lease, rpc_broken = await _borrow_with_retry(
        engine_client, num_nonces, seq_len, deadline)
    if lease is not None:
        return lease

    compat = _current_compat()
    try:
        aborted = await compat.abort_all_requests(engine_client)
    except Exception as exc:
        aborted = 0
        logger.error("PoC reservation: abort escalation failed: %s", exc)
    logger.warning(
        "PoC reservation: no lease within %dms%s — aborted %d in-flight "
        "request(s), %s",
        timeout_ms,
        " (borrow RPC broken)" if rpc_broken else "",
        aborted,
        "falling back to the legacy in-place path" if rpc_broken
        else "retrying the borrow once")
    await asyncio.sleep(_ABORT_SETTLE_SEC)

    if not rpc_broken:
        lease, _ = await _borrow_with_retry(
            engine_client, num_nonces, seq_len,
            time.monotonic() + timeout_ms / 1000.0)
        if lease is not None:
            return lease
    return None


@asynccontextmanager
async def poc_reservation(
    engine_client: Any,
    num_nonces: int,
    seq_len: int,
    timeout_ms: int = POC_BORROW_TIMEOUT_MS,
):
    """Serialized lease → (caller's forwards) → return, as a context manager.

    Yields the lease dict (``{"block_ids": [...], "blocks_per_seq": int}``)
    or ``None`` for the legacy fallback — inference has already been aborted
    by then. Block return is structurally guaranteed (``finally``); on the
    legacy path the prefix cache is reset instead (see module docstring).
    """
    async with _poc_reservation_lock:
        lease = await reserve_poc_blocks(
            engine_client, num_nonces, seq_len, timeout_ms)
        try:
            yield lease
        finally:
            if lease is not None:
                try:
                    await _current_compat().return_poc_blocks(
                        engine_client, lease["block_ids"])
                except Exception as exc:
                    logger.error(
                        "PoC failed to return %d leased KV blocks: %s",
                        len(lease["block_ids"]), exc)
            else:
                await reset_prefix_cache_after_inplace_poc(engine_client)


async def reset_prefix_cache_after_inplace_poc(engine_client: Any) -> None:
    """Drop the prefix cache after an in-place (blocks ``0..N``) PoC round.

    The legacy path overwrites cached blocks WITHOUT evicting their hashes
    (``free_blocks`` keeps ``block_hash`` for reuse), so a later prefix hit
    would silently serve PoC garbage as KV. Best-effort: the engine may not
    expose ``reset_prefix_cache`` (then the operator must not enable prefix
    caching with PoC — pre-existing behaviour).
    """
    reset = getattr(engine_client, "reset_prefix_cache", None)
    if reset is None:
        logger.warning(
            "engine client lacks reset_prefix_cache — prefix cache may hold "
            "PoC-clobbered blocks after an in-place PoC round")
        return
    try:
        await reset()
        logger.info("prefix cache reset after in-place PoC round")
    except Exception as exc:
        logger.warning("reset_prefix_cache failed: %s", exc)
