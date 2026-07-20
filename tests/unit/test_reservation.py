"""Unit tests for the PoC KV reservation layer (``gonka_poc.poc.reservation``).

Uses a fake engine client exposing the two surfaces the compat wrappers
touch (``engine_core.call_utility_async``) plus the escalation/abort and
prefix-reset surfaces (``abort``, ``reset_prefix_cache``). vllm must be
importable (the compat dispatch resolves by installed version) but no
engine is started.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("vllm")

from gonka_poc.poc import reservation  # noqa: E402


class FakeCore:
    def __init__(self, script):
        # script: list of returns; an Exception instance is raised instead.
        # A test may instead set ``responder`` = callable(method, args) for
        # state-dependent behaviour.
        self.script = list(script)
        self.calls = []
        self.responder = None

    async def call_utility_async(self, method, *args):
        self.calls.append((method, args))
        if method == "gonka_poc_return_blocks":
            return None
        item = self.responder(method, args) if self.responder else self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeEngine:
    def __init__(self, script, with_abort=True):
        self.engine_core = FakeCore(script)
        self.aborted = 0
        self.prefix_resets = 0
        if with_abort:
            # abort_all_requests enumerates output_processor.request_states
            # and calls .abort(rid); give it one in-flight request.
            class _OP:
                request_states = {"req-1": object()}
            self.output_processor = _OP()

    async def abort(self, request_id, **kwargs):
        self.aborted += 1

    async def reset_prefix_cache(self):
        self.prefix_resets += 1


LEASE = {"block_ids": [7, 3, 12, 5], "blocks_per_seq": 2}


def _run(coro):
    return asyncio.run(coro)


def test_lease_yielded_and_returned():
    eng = FakeEngine([LEASE])

    async def go():
        async with reservation.poc_reservation(eng, 2, 32) as lease:
            assert lease == LEASE
        methods = [m for m, _ in eng.engine_core.calls]
        assert methods == [
            "gonka_poc_borrow_blocks", "gonka_poc_return_blocks"]
        assert eng.engine_core.calls[1][1] == (LEASE["block_ids"],)

    _run(go())
    assert eng.aborted == 0
    assert eng.prefix_resets == 0


def test_blocks_returned_on_body_exception():
    eng = FakeEngine([LEASE])

    async def go():
        with pytest.raises(RuntimeError, match="boom"):
            async with reservation.poc_reservation(eng, 2, 32):
                raise RuntimeError("boom")

    _run(go())
    assert [m for m, _ in eng.engine_core.calls][-1] == "gonka_poc_return_blocks"


def test_pool_busy_escalates_through_abort_then_succeeds():
    # Pool stays busy (None) UNTIL inference is aborted — only the abort
    # escalation can free blocks, so a lease before eng.aborted would be a
    # logic error, not a lucky poll.
    eng = FakeEngine([])
    eng.engine_core.responder = (
        lambda method, args: LEASE if eng.aborted else None)

    async def go():
        async with reservation.poc_reservation(eng, 2, 32, timeout_ms=60) as lease:
            assert lease == LEASE

    _run(go())
    assert eng.aborted == 1  # escalation drained the in-flight request
    assert eng.prefix_resets == 0


def test_rpc_broken_falls_back_with_abort_and_prefix_reset():
    # Borrow RPC raises (engine without the injected methods): no re-borrow,
    # inference aborted for the legacy in-place path, prefix cache reset on
    # exit because blocks 0..N were clobbered.
    eng = FakeEngine([AttributeError("no gonka_poc_borrow_blocks")])

    async def go():
        async with reservation.poc_reservation(eng, 2, 32, timeout_ms=1) as lease:
            assert lease is None

    _run(go())
    borrow_calls = [
        m for m, _ in eng.engine_core.calls if m == "gonka_poc_borrow_blocks"]
    assert len(borrow_calls) == 1  # broken RPC is not retried
    assert eng.aborted == 1        # legacy fallback is abort-protected
    assert eng.prefix_resets == 1  # poisoned prefix cache dropped


def test_reservation_lock_serializes():
    eng = FakeEngine([LEASE, LEASE])
    order = []

    async def user(tag, hold):
        async with reservation.poc_reservation(eng, 1, 16):
            order.append(f"{tag}-in")
            await asyncio.sleep(hold)
            order.append(f"{tag}-out")

    async def go():
        await asyncio.gather(user("a", 0.05), user("b", 0))

    _run(go())
    assert order == ["a-in", "a-out", "b-in", "b-out"]
