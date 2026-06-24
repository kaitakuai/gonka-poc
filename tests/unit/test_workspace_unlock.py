"""Unit tests for ``gonka_poc.worker.extension.unlocked_moe_workspace``.

The vLLM 0.23 lockable MoE ``WorkspaceManager`` sizes the MoE scratch from
inference shapes and LOCKS it before the PoC forward runs; the PoC forward's
larger shape would then raise "Workspace is locked". The fix unlocks the
workspace around the forward and re-locks after.

These tests inject a fake ``vllm.v1.worker.workspace`` via ``sys.modules`` so
they are CPU-only and need no real vllm (importing the extension module itself
pulls only ``gonka_poc._compat``, which is light).
"""
from __future__ import annotations

import sys
import types

import pytest

from gonka_poc.worker.extension import unlocked_moe_workspace


def _install_fake_vllm_workspace(monkeypatch, *, unlock, lock):
    """Make ``from vllm.v1.worker.workspace import lock_workspace,
    unlock_workspace`` resolve to the supplied callables (wins over a real vllm
    install because ``sys.modules`` takes precedence)."""
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    ws = types.ModuleType("vllm.v1.worker.workspace")
    ws.unlock_workspace = unlock
    ws.lock_workspace = lock
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", ws)


def test_unlocks_then_relocks(monkeypatch):
    calls = []
    _install_fake_vllm_workspace(
        monkeypatch,
        unlock=lambda: calls.append("unlock"),
        lock=lambda: calls.append("lock"),
    )
    with unlocked_moe_workspace():
        calls.append("forward")
    assert calls == ["unlock", "forward", "lock"]


def test_relocks_even_when_body_raises(monkeypatch):
    calls = []
    _install_fake_vllm_workspace(
        monkeypatch,
        unlock=lambda: calls.append("unlock"),
        lock=lambda: calls.append("lock"),
    )
    with pytest.raises(ValueError):
        with unlocked_moe_workspace():
            raise ValueError("boom")
    # lock must still fire in the finally, and only after unlock
    assert calls == ["unlock", "lock"]


def test_no_manager_is_a_noop(monkeypatch):
    # current_workspace_manager() asserts when no MoE workspace exists, so
    # unlock_workspace() raises — it must be swallowed and lock must NOT fire.
    calls = []

    def _unlock_raises():
        raise AssertionError("workspace manager not initialized")

    _install_fake_vllm_workspace(
        monkeypatch, unlock=_unlock_raises, lock=lambda: calls.append("lock")
    )
    with unlocked_moe_workspace():
        calls.append("forward")
    assert calls == ["forward"]  # unlock failed -> not unlocked -> no re-lock


def test_vllm_below_0_23_is_a_noop(monkeypatch):
    # Pre-0.23 has no lock/unlock symbols -> ImportError on the name import ->
    # the manager is a no-op, the body still runs, nothing raises.
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.workspace",
        types.ModuleType("vllm.v1.worker.workspace"),  # lacks lock/unlock attrs
    )
    ran = []
    with unlocked_moe_workspace():
        ran.append("forward")
    assert ran == ["forward"]
