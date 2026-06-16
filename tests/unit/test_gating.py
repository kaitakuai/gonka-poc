"""Unit tests for ``gonka_poc.entrypoint.gating.PoCGatingMiddleware``.

Replaces the deleted ``tests/gonka/test_chat_priority_gating.py``, which had
been written against a pre-refactor architecture (module-global flag +
AsyncLLM monkey-patching). The new gate lives entirely in the ASGI
middleware ``PoCGatingMiddleware`` + the ``PoCGate`` flag object; we
exercise it directly via a Starlette test client mounted on a minimal
FastAPI app.

No vllm, no engine, no monkey-patching: this test stays CPU-only and runs
on the same wheel as the contract tests but does not need vllm imported.
"""
from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gonka_poc.entrypoint.gating import (
    DEFAULT_BLOCKED_PREFIXES,
    PoCGate,
    PoCGatingMiddleware,
)


# --------------------------------------------------------------------------- #
# Fail-fast API surface assertions.
#
# If these break, the middleware contract changed and every test below is
# meaningless until the new shape is reflected here. Catch this loudly.
# --------------------------------------------------------------------------- #


def test_middleware_api_surface_unchanged() -> None:
    """Pin the public symbols this whole test module relies on.

    If ``activate``/``deactivate``/``is_active`` ever become async, or the
    ``blocked_prefixes`` kwarg is renamed, the rest of this file is wrong.
    Fail here instead of producing a cascade of confusing 200/503 mismatches.
    """
    gate = PoCGate()
    # PoCGate.is_active / activate / deactivate are sync (verified in
    # src/gonka_poc/entrypoint/gating.py:42-51).
    assert callable(gate.is_active)
    assert callable(gate.activate)
    assert callable(gate.deactivate)
    assert gate.is_active() is False

    # DEFAULT_BLOCKED_PREFIXES MUST include both OpenAI-compatible paths;
    # any narrowing here means a chat request could leak through during PoC.
    assert "/v1/chat/completions" in DEFAULT_BLOCKED_PREFIXES
    assert "/v1/completions" in DEFAULT_BLOCKED_PREFIXES


# --------------------------------------------------------------------------- #
# App fixture
# --------------------------------------------------------------------------- #


def _build_app(
    blocked_prefixes: Optional[tuple[str, ...]] = None,
) -> tuple[FastAPI, PoCGate]:
    """Minimal FastAPI app wired with the middleware -- mirrors
    :func:`gonka_poc.entrypoint.api_router.build_gonka_app` but without the
    vllm router import.
    """
    app = FastAPI()
    gate = PoCGate()
    app.state.gonka_gate = gate

    kwargs: dict = {"gate": gate}
    if blocked_prefixes is not None:
        kwargs["blocked_prefixes"] = blocked_prefixes
    app.add_middleware(PoCGatingMiddleware, **kwargs)

    @app.post("/v1/chat/completions")
    async def chat() -> dict:
        return {"ok": True, "route": "chat"}

    @app.post("/v1/completions")
    async def comp() -> dict:
        return {"ok": True, "route": "comp"}

    @app.post("/api/v1/pow/init")
    async def pow_init() -> dict:
        # PoC routes MUST stay reachable even when the gate is active;
        # otherwise we deadlock (PoC can never deactivate the gate).
        return {"ok": True, "route": "pow"}

    @app.post("/foo/bar")
    async def foo_bar() -> dict:
        return {"ok": True, "route": "foo"}

    @app.post("/unrelated")
    async def unrelated() -> dict:
        return {"ok": True, "route": "unrelated"}

    return app, gate


@pytest.fixture
def app_and_gate() -> tuple[FastAPI, PoCGate]:
    return _build_app()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_gate_default_state_inactive(app_and_gate: tuple[FastAPI, PoCGate]) -> None:
    """A freshly constructed PoCGate is inactive => requests pass through."""
    app, gate = app_and_gate
    assert gate.is_active() is False

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "route": "chat"}


def test_chat_completions_blocked_when_active(
    app_and_gate: tuple[FastAPI, PoCGate],
) -> None:
    """Gate active => /v1/chat/completions returns 503 + Retry-After."""
    app, gate = app_and_gate
    gate.activate(reason="poc-generation")

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions")

    assert r.status_code == 503, r.text
    assert "retry-after" in {k.lower() for k in r.headers.keys()}


def test_completions_blocked_when_active(
    app_and_gate: tuple[FastAPI, PoCGate],
) -> None:
    """Gate active => /v1/completions also returns 503."""
    app, gate = app_and_gate
    gate.activate()

    with TestClient(app) as client:
        r = client.post("/v1/completions")

    assert r.status_code == 503, r.text


def test_poc_routes_pass_when_gate_active(
    app_and_gate: tuple[FastAPI, PoCGate],
) -> None:
    """Gate active MUST NOT block /api/v1/pow/* -- otherwise PoC can never
    deactivate the gate and we deadlock the server.
    """
    app, gate = app_and_gate
    gate.activate()

    with TestClient(app) as client:
        r = client.post("/api/v1/pow/init")

    assert r.status_code != 503, (
        f"PoC route was 503-blocked while the gate is active. Body: {r.text}. "
        "If you intentionally added /api/v1/pow/* to DEFAULT_BLOCKED_PREFIXES "
        "you just deadlocked the server -- revert."
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "route": "pow"}


def test_chat_resumes_after_deactivate(
    app_and_gate: tuple[FastAPI, PoCGate],
) -> None:
    """activate => 503; deactivate => 200 again (no sticky state)."""
    app, gate = app_and_gate

    with TestClient(app) as client:
        gate.activate()
        r1 = client.post("/v1/chat/completions")
        assert r1.status_code == 503, r1.text

        gate.deactivate()
        r2 = client.post("/v1/chat/completions")
        assert r2.status_code == 200, r2.text
        assert r2.json() == {"ok": True, "route": "chat"}


def test_custom_blocked_prefixes() -> None:
    """Constructor accepts blocked_prefixes override; only those 503."""
    app, gate = _build_app(blocked_prefixes=("/foo",))
    gate.activate()

    with TestClient(app) as client:
        # /foo/bar matches the custom prefix -> blocked
        r_foo = client.post("/foo/bar")
        # /unrelated is NOT under /foo -> passes through
        r_other = client.post("/unrelated")
        # And the upstream default /v1/chat/completions is now NOT in the
        # blocked list (we replaced it) -> also passes through. This double-
        # asserts that the override replaces, not augments, the default.
        r_chat = client.post("/v1/chat/completions")

    assert r_foo.status_code == 503, r_foo.text
    assert r_other.status_code == 200, r_other.text
    assert r_chat.status_code == 200, r_chat.text


def test_503_body_is_json_with_retry_after(
    app_and_gate: tuple[FastAPI, PoCGate],
) -> None:
    """503 body carries a structured error + numeric Retry-After header."""
    app, gate = app_and_gate
    gate.activate(reason="custom-reason-string")

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions")

    assert r.status_code == 503
    body = r.json()
    # Assert the actual shape returned by gating.py:86-93. If a future
    # refactor restructures the body, this fails loudly so downstream
    # consumers (CLI, Gonka network node) can be updated in sync.
    assert body["error"] == "poc_generation_active"
    assert body["reason"] == "custom-reason-string"
    assert isinstance(body["retry_after_ms"], int)

    retry_after = r.headers.get("retry-after")
    assert retry_after is not None, "Retry-After header missing on 503"
    # HTTP spec: Retry-After is either a numeric (delta-seconds) or HTTP-date.
    # gating.py emits a numeric string ("1") -- assert that shape.
    assert retry_after.isdigit(), (
        f"Retry-After must be numeric delta-seconds, got {retry_after!r}"
    )
