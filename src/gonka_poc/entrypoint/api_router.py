"""``gonka-vllm-serve`` -- compose a FastAPI app on top of stock vLLM 0.23.

This is a thin wrapper around the upstream public API:

    vllm.entrypoints.openai.api_server.setup_server
    vllm.entrypoints.openai.api_server.build_async_engine_client
    vllm.entrypoints.openai.api_server.build_app
    vllm.entrypoints.openai.api_server.init_app_state
    vllm.entrypoints.launcher.serve_http
    vllm.entrypoints.openai.cli_args.make_arg_parser
    vllm.entrypoints.openai.cli_args.validate_parsed_serve_args

We do NOT patch any vLLM source file. We only:
  1. Build the stock FastAPI app via ``build_app(args, ...)``.
  2. Attach our PoC router (``vllm/poc/routes.py`` ported into
     ``gonka_poc.poc.routes``) via ``app.include_router(...)``.
  3. Install ``PoCGatingMiddleware`` AFTER ``build_app`` returns so it ends up
     OUTERMOST in Starlette's reverse-insertion order, gating the
     ``/v1/chat/completions`` and ``/v1/completions`` routes with 503 when PoC
     is active.
  4. Forward to ``serve_http`` exactly like upstream's ``run_server_worker``.

Middleware ordering note (verified against v0.23.0
``vllm/entrypoints/openai/api_server.py:156-300``): user-supplied
``--middleware`` are added at L287-297; we add OURS AFTER ``build_app`` so we
sit outside them too. The chat-completion handler is reached only if the gate
is open.
"""
from __future__ import annotations

import logging
import signal
import sys
from typing import Any, Iterable, Optional

# fastapi import is cheap; vllm imports are deferred to ``main`` so that
# ``--help`` / argparse error paths don't fork the engine.
from fastapi import FastAPI

from gonka_poc.entrypoint.gating import (
    DEFAULT_BLOCKED_PREFIXES,
    PoCGate,
    PoCGatingMiddleware,
)

logger = logging.getLogger("gonka_poc.entrypoint")


def _interrupt_init(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
    """SIGTERM handler mirroring upstream ``api_server._interrupt_init``.

    Translates SIGTERM into a KeyboardInterrupt so the uvloop event loop
    unwinds cleanly via the same path as Ctrl-C.
    """
    raise KeyboardInterrupt("gonka-vllm-serve received SIGTERM")


def attach_poc_router(app: FastAPI) -> None:
    """Attach the PoC API router (ported from ``vllm/poc/routes.py``)."""
    # Imported lazily because gonka_poc.poc.* pulls vllm.logger which itself
    # requires a configured vllm runtime.
    from gonka_poc.poc.routes import router as poc_router

    app.include_router(poc_router)


def register_gate_presence_warning(app: FastAPI) -> None:
    """Attach a FastAPI startup hook that warns when the plugin is loaded
    without the gating middleware installed.

    This catches the operator footgun where ``vllm serve`` runs instead of
    ``gonka-vllm-serve``: the plugin entry point still executes (so models /
    custom_ops get registered) but no ``app.state.gonka_gate`` is set and the
    chat endpoint will NOT be 503-gated while PoC seizes the GPU.

    Invoked both from :func:`build_gonka_app` (where the warning never fires
    -- the gate is always present by construction) AND from a build_app
    wrapper installed by :func:`gonka_poc.plugin.register` (where the warning
    will fire because plain ``vllm serve`` never attached a gate).
    """

    @app.on_event("startup")
    async def _gonka_gate_presence_warning() -> None:  # pragma: no cover - startup hook
        import gonka_poc

        plugin_loaded = bool(getattr(gonka_poc, "PLUGIN_LOADED", False))
        gate_present = getattr(app.state, "gonka_gate", None) is not None
        if plugin_loaded and not gate_present:
            logger.warning(
                "gonka_poc plugin loaded but app.state.gonka_gate is missing -- "
                "PoC chat-completion gating is DISABLED. "
                "Did you launch with `vllm serve` instead of `gonka-vllm-serve`?"
            )


def build_gonka_app(
    app: FastAPI,
    *,
    gate: PoCGate,
    blocked_prefixes: Optional[Iterable[str]] = None,
) -> FastAPI:
    """Mutate the upstream-built FastAPI app: add PoC router + gating middleware.

    Args:
        app: the FastAPI instance returned by
            ``vllm.entrypoints.openai.api_server.build_app(args, ...)``.
        gate: the shared :class:`PoCGate` flag toggled by the PoC router.
        blocked_prefixes: optional override for the path prefixes the gating
            middleware 503s while PoC is active. ``None`` uses
            :data:`gonka_poc.entrypoint.gating.DEFAULT_BLOCKED_PREFIXES`.

    Returns:
        The same ``app`` instance (mutated). Returned for chainability.
    """
    # Router first so /api/v1/pow/* is registered on the same FastAPI as
    # /v1/chat/completions. Both share ``app.state``.
    attach_poc_router(app)

    # State for both the gating middleware AND the PoC router to read.
    app.state.gonka_gate = gate

    # Install the gating middleware LAST (so it runs FIRST per Starlette's
    # reverse-insertion ordering). MUST be called before app starts -- which
    # is true here since we run before ``serve_http``.
    prefixes = (
        tuple(blocked_prefixes)
        if blocked_prefixes is not None
        else DEFAULT_BLOCKED_PREFIXES
    )
    app.add_middleware(PoCGatingMiddleware, gate=gate, blocked_prefixes=prefixes)

    # Register the startup-event hook AFTER the gate is attached. Here it is
    # a defensive no-op (gate IS present), but the same callback installed by
    # the plugin's ``build_app`` wrapper fires in the bare-``vllm serve`` case.
    register_gate_presence_warning(app)

    return app


async def _run_server(args: Any) -> None:
    """Async body equivalent to ``vllm.entrypoints.openai.api_server.run_server``,
    but inserting PoC composition between ``build_app`` and ``serve_http``.

    Mirrors v0.23.0 ``api_server.py:559-604`` (``build_and_serve``).
    """
    # Deferred imports: keep ``gonka-vllm-serve --help`` fast and isolated
    # from CUDA fork issues.
    import vllm.envs as envs
    from vllm.entrypoints.openai.api_server import (
        build_app,
        build_async_engine_client,
        init_app_state,
        setup_server,
    )
    from vllm.entrypoints.launcher import serve_http

    listen_address, sock = setup_server(args)

    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        # Stock vLLM app + middleware/handlers.
        app = build_app(args, supported_tasks, model_config)

        # Gonka composition: PoC router + gating middleware.
        gate = PoCGate()
        build_gonka_app(
            app,
            gate=gate,
            blocked_prefixes=getattr(args, "gonka_poc_block_prefixes", None),
        )

        # Standard upstream state population (sets app.state.engine_client and
        # app.state.openai_serving_*). MUST run after build_app and after we
        # add our middleware (Starlette freezes the stack on startup, which
        # serve_http triggers).
        await init_app_state(engine_client, app.state, args, supported_tasks)

        # Hand off to uvicorn via the stock launcher.
        shutdown_task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=getattr(args, "uvicorn_log_level", "info"),
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=getattr(args, "ssl_keyfile", None),
            ssl_certfile=getattr(args, "ssl_certfile", None),
            ssl_ca_certs=getattr(args, "ssl_ca_certs", None),
            ssl_cert_reqs=getattr(args, "ssl_cert_reqs", 0),
        )
        await shutdown_task

    sock.close()


def main(argv: list[str] | None = None) -> int:
    """``gonka-vllm-serve`` entry point.

    Mirrors v0.23.0 ``api_server.py:__main__`` (L693-703) but routes through
    :func:`_run_server` so we own the composition step.
    """
    # Deferred to avoid pulling vllm at --help time on a system without it.
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="gonka-vllm-serve: vLLM OpenAI server with Gonka PoC v2 plugin",
    )
    parser = make_arg_parser(parser)

    # Gonka-local toggles (do NOT shadow upstream flag names).
    parser.add_argument(
        "--gonka-poc-block-prefixes",
        nargs="*",
        default=None,
        help="Override which path prefixes PoC priority gates with 503. "
        "Default: /v1/chat/completions /v1/completions",
    )

    args = parser.parse_args(argv)
    validate_parsed_serve_args(args)

    # Mirror upstream ``api_server.run_server``: translate SIGTERM into the
    # same shutdown path as Ctrl-C so the event loop unwinds the engine
    # cleanly instead of being torn down mid-step.
    signal.signal(signal.SIGTERM, _interrupt_init)

    # uvloop.run is the upstream parity choice (matches
    # ``vllm.entrypoints.openai.api_server.__main__``). Deferred so the
    # ``--help`` path stays light on a system without uvloop.
    import uvloop  # type: ignore[import-not-found]

    try:
        uvloop.run(_run_server(args))
    except KeyboardInterrupt:
        logger.info("gonka-vllm-serve interrupted; shutting down")
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
