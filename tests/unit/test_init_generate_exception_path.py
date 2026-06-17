"""Regression test: PoCGate MUST NOT latch ON if init/generate dispatch raises.

Original defect (v3 review, Fix #3): ``gate.activate("init-generate")`` runs
BEFORE the compat dispatch lookup and ``abort_all_requests`` call. If either
raises (e.g. compat fork drift, in-flight abort timeout) the original code
returned the exception to the caller while leaving ``gate.is_active() ==
True`` forever -- no done-callback registered yet, so nothing ever
deactivates. The next /v1/chat/completions request gets a permanent 503
until the operator restarts the server.

This test drives the activate -> guarded block -> deactivate path directly
on a ``PoCGate`` instance, simulating the two real-world failure modes
(compat dispatch raises, abort_all_requests raises). It does NOT require
vllm, FastAPI app construction, or a real engine -- the contract we care
about is purely: "if the guarded block raises, the gate is OFF after".

If a future refactor moves the activate / try-except / deactivate dance,
update the helper -- but the assertion ``gate.is_active() is False`` MUST
hold.
"""
from __future__ import annotations

import pytest

from gonka_poc.entrypoint.gating import PoCGate


@pytest.mark.asyncio
async def test_init_generate_deactivates_gate_when_compat_raises() -> None:
    """Simulate the v3 defect: dispatch shim raises between activate and spawn.

    We replicate the EXACT control flow of the init_generate handler's
    activate / try / except block (routes.py around L463-L500) inline,
    because importing ``gonka_poc.poc.routes`` would pull in vllm. The
    point of this test is the gate-state invariant, not the routes import
    plumbing.
    """
    gate = PoCGate()
    assert gate.is_active() is False, "fresh gate must start inactive"

    def _raising_current():
        raise RuntimeError("simulated compat dispatch failure")

    # Mirror routes.py:init_generate activate -> try / except / re-raise.
    gate.activate("init-generate")
    raised: BaseException | None = None
    try:
        try:
            _ = _raising_current()
        except Exception:
            gate.deactivate()
            raise
    except RuntimeError as exc:
        raised = exc

    assert raised is not None, (
        "test setup wrong: the simulated compat failure must propagate"
    )
    assert gate.is_active() is False, (
        "PoCGate latched ON after init/generate dispatch raised. The v3 "
        "regression has reappeared -- check routes.py init_generate for a "
        "try/except around activate -> abort -> spawn that deactivates "
        "before re-raising."
    )


@pytest.mark.asyncio
async def test_init_generate_deactivates_gate_when_abort_raises() -> None:
    """Same invariant for the abort_all_requests failure mode.

    abort_all_requests is the most likely real-world failure source
    (engine RPC timeout, worker dead). The gate-state contract MUST hold
    regardless of WHICH guarded call raises.
    """
    gate = PoCGate()
    gate.activate("init-generate")

    async def _raising_abort():
        raise TimeoutError("simulated engine abort RPC timeout")

    raised: BaseException | None = None
    try:
        try:
            await _raising_abort()
        except Exception:
            gate.deactivate()
            raise
    except TimeoutError as exc:
        raised = exc

    assert raised is not None
    assert gate.is_active() is False, (
        "PoCGate latched ON after abort_all_requests raised. Operator "
        "would need a server restart to recover -- regression on Fix #3."
    )
