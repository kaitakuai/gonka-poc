# ADR-0013 — PoC gate / abort / spawn ordering contract (local stub)

**Status:** Accepted. Canonical source lives in **mlnode-foundry** —
[`docs/adr/0013-poc-integration-architecture.md`](https://github.com/kaitakuai/mlnode-foundry/blob/main/docs/adr/0013-poc-integration-architecture.md).
This file is a 1-page summary so an offline reader of the `gonka-poc` repo
can find the rationale for the inline comment at
`src/gonka_poc/poc/routes.py:468` without leaving the package.

**Date:** 2026-06-12 (canonical); local stub 2026-06-17.
**Owners:** @baychak
**Supersedes / amends:** none. **Amended by:** [ADR-0014](ADR-0014-plugin-vs-fork.md).

## Context (why this file exists)

`src/gonka_poc/poc/routes.py` carries an inline reference to ADR-0013 in
`init_generate`:

> Ordering contract (ADR-0013): gate.activate → abort_all_requests → spawn
> gen task.

Without a local ADR file, that comment is a dead link for anyone reading the
plugin source tree in isolation. This stub closes the link-rot. The full
architectural justification (three-layer plugin / shim / upstream story,
fork history, options considered) lives in the canonical mlnode-foundry
ADR — do not duplicate it here.

## Decision (the part the plugin code depends on)

PoC `init/generate` MUST execute three steps in this exact order:

1. **`gate.activate("init-generate")`** — flips the `PoCGate` shared state.
   The Starlette `PoCGatingMiddleware` starts returning HTTP 503 with
   `Retry-After` to `/v1/chat/completions` and `/v1/completions` immediately.
   No module-level flag; the gate is the single source of truth for
   "PoC is currently running".
2. **`await compat.abort_all_requests(engine_client)`** — drains the set of
   already-admitted chat/completions requests that snuck in before the gate
   flipped. The middleware blocks NEW admissions; this call kills in-flight
   ones so the PoC forward runs on an exclusively-owned GPU.
3. **`asyncio.create_task(_generation_loop(...))`** — spawns the PoC
   generation task. Only after (1)+(2) have completed.

Inverting any pair breaks the contract:
- spawn-before-abort → PoC forward fights in-flight chat batches for KV
  cache and execution slots; cross-validator bit-compat drifts.
- spawn-before-activate → new chat requests stream in alongside PoC; same
  failure mode plus a 503-window race.
- activate-after-abort → the gap between `abort_all_requests` and
  `activate` admits new chat requests that the abort already missed.

`gen_task.add_done_callback` deactivates the gate when generation finishes
(or is cancelled) — see `_on_generation_done` in the same function.

## Why this lives in two places

- **Canonical:** mlnode-foundry owns the architectural decision record
  (it covers more than just this plugin — image policy, foundry overlay,
  contract tests across the four-stage pipeline).
- **Local stub:** the plugin code cites this ADR by number; we keep the
  citation resolvable from inside the `gonka-poc` checkout. When
  mlnode-foundry's ADR-0013 changes, update this stub's summary so they
  stay coherent (drift risk acknowledged — bounded by the small surface
  this stub copies).

## Dependencies

- `gonka_poc.entrypoint.gating.PoCGate` — the shared activate/deactivate
  flag (see `tests/unit/test_gating.py` for the Starlette contract).
- `gonka_poc._compat.current()` dispatch — `abort_all_requests` is private
  vLLM surface that varies across minors; the compat shim isolates the
  version-specific call site.

## Links

- Canonical ADR: [mlnode-foundry/docs/adr/0013-poc-integration-architecture.md](https://github.com/kaitakuai/mlnode-foundry/blob/main/docs/adr/0013-poc-integration-architecture.md)
- Source citing this ADR: `src/gonka_poc/poc/routes.py:468`
- Companion ADR: [ADR-0014](ADR-0014-plugin-vs-fork.md) (residual fork as
  permanent infrastructure)
