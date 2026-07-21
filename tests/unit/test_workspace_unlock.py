"""Unit tests for ``gonka_poc.worker.extension.unlocked_moe_workspace``.

The vLLM 0.23 lockable MoE ``WorkspaceManager`` sizes the MoE scratch from
inference shapes and LOCKS it before the PoC forward runs; the PoC forward's
larger shape would then raise "Workspace is locked". The fix unlocks the
workspace around the forward (via the ``_compat`` shim) and re-locks after.

CPU-only: we replace the version-dispatched ``_compat_current()`` with a fake compat
module exposing ``unlock_moe_workspace`` / ``lock_moe_workspace`` — no real
vllm needed (importing the extension module pulls only the light ``_compat``).
"""
from __future__ import annotations

import types

import pytest

import gonka_poc.worker.extension as ext


def _fake_compat(monkeypatch, *, unlock, lock):
    compat = types.SimpleNamespace(
        unlock_moe_workspace=unlock, lock_moe_workspace=lock
    )
    monkeypatch.setattr(ext, "_compat_current", lambda: compat)


def test_unlocks_then_relocks(monkeypatch):
    calls = []
    _fake_compat(
        monkeypatch,
        unlock=lambda: (calls.append("unlock"), True)[1],
        lock=lambda: calls.append("lock"),
    )
    with ext.unlocked_moe_workspace():
        calls.append("forward")
    assert calls == ["unlock", "forward", "lock"]


def test_relocks_even_when_body_raises(monkeypatch):
    calls = []
    _fake_compat(
        monkeypatch,
        unlock=lambda: (calls.append("unlock"), True)[1],
        lock=lambda: calls.append("lock"),
    )
    with pytest.raises(ValueError):
        with ext.unlocked_moe_workspace():
            raise ValueError("boom")
    assert calls == ["unlock", "lock"]  # re-locked in finally, after unlock


def test_no_active_manager_is_a_noop(monkeypatch):
    # unlock returns False (non-MoE model: current_workspace_manager() asserts) ->
    # we must NOT call lock.
    calls = []
    _fake_compat(monkeypatch, unlock=lambda: False, lock=lambda: calls.append("lock"))
    with ext.unlocked_moe_workspace():
        calls.append("forward")
    assert calls == ["forward"]


def test_unlock_raising_is_swallowed(monkeypatch):
    calls = []

    def _raises():
        raise AssertionError("workspace manager not initialized")

    _fake_compat(monkeypatch, unlock=_raises, lock=lambda: calls.append("lock"))
    with ext.unlocked_moe_workspace():
        calls.append("forward")
    assert calls == ["forward"]  # swallowed, body ran, no re-lock


def test_compat_without_workspace_symbols_is_a_noop(monkeypatch):
    # older shim / vLLM < 0.23: compat module lacks the functions -> no-op.
    monkeypatch.setattr(ext, "_compat_current", lambda: types.SimpleNamespace())
    ran = []
    with ext.unlocked_moe_workspace():
        ran.append("forward")
    assert ran == ["forward"]


def test_current_dispatch_failure_is_a_noop(monkeypatch):
    # _compat_current() itself failing (vllm unavailable / version unmapped) -> no-op.
    def _boom():
        raise RuntimeError("no vllm")

    monkeypatch.setattr(ext, "_compat_current", _boom)
    ran = []
    with ext.unlocked_moe_workspace():
        ran.append("forward")
    assert ran == ["forward"]
