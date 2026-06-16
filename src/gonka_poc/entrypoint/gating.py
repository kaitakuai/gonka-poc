"""PoCGatingMiddleware -- 503 + abort-in-flight on PoC priority gate.

Modeled on ``vllm/entrypoints/serve/elastic_ep/middleware.py:ScalingMiddleware``
(v0.23.0). When PoC generation is active, new chat/completion requests get a
503 response BEFORE reaching any vLLM handler.

In-flight aborts: a Starlette middleware cannot directly cancel a
``StreamingResponse`` that has already begun yielding bytes. The companion
shutdown path is the explicit ``EngineClient.abort_request(request_id)``
call issued from the PoC router itself (see ``api_router.py``) right before
GPU work starts. The middleware is purely a "don't accept new work" gate.
"""
from __future__ import annotations

from typing import Awaitable, Callable, Iterable, Optional

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


# Paths that PoC priority blocks. Keep this list narrow -- /gonka/poc/* and
# /api/v1/pow/* MUST always be reachable (otherwise we deadlock).
DEFAULT_BLOCKED_PREFIXES: tuple[str, ...] = (
    "/v1/chat/completions",
    "/v1/completions",
)


class PoCGate:
    """Process-local PoC priority flag.

    Toggled by the PoC router when generation starts / stops. Lives on
    ``app.state.gonka_gate`` so the middleware can read it without import
    cycles. Thread-safe enough for asyncio (single-writer pattern: the PoC
    router serialises start/stop).
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._reason: Optional[str] = None

    def is_active(self) -> bool:
        return self._active

    def activate(self, reason: str = "poc-generation") -> None:
        self._active = True
        self._reason = reason

    def deactivate(self) -> None:
        self._active = False
        self._reason = None

    @property
    def reason(self) -> Optional[str]:
        return self._reason


class PoCGatingMiddleware:
    """ASGI middleware. When the gate is active, return 503 for blocked paths.

    Starlette wraps middlewares in REVERSE insertion order, so installing this
    AFTER ``vllm.entrypoints.openai.api_server.build_app(...)`` returns puts it
    OUTERMOST -- it runs before any vLLM handler.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        gate: PoCGate,
        blocked_prefixes: Iterable[str] = DEFAULT_BLOCKED_PREFIXES,
    ) -> None:
        self.app = app
        self.gate = gate
        self.blocked_prefixes = tuple(blocked_prefixes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            # websockets / lifespan: pass through unchanged
            await self.app(scope, receive, send)
            return

        if self.gate.is_active():
            path: str = scope.get("path", "")
            if any(path.startswith(prefix) for prefix in self.blocked_prefixes):
                response = JSONResponse(
                    status_code=503,
                    content={
                        "error": "poc_generation_active",
                        "reason": self.gate.reason or "poc-generation",
                        "retry_after_ms": 100,
                    },
                    headers={"Retry-After": "1"},
                )
                await response(scope, receive, send)
                return

        await self.app(scope, receive, send)


__all__ = ["PoCGate", "PoCGatingMiddleware", "DEFAULT_BLOCKED_PREFIXES"]
