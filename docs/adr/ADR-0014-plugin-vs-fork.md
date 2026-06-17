# ADR-0014 — Residual vLLM fork as permanent infrastructure (local stub)

**Status:** Accepted (amends ADR-0013 §Layer-3). Canonical source lives in
**mlnode-foundry** —
[`docs/adr/0014-residual-fork-permanent-infra.md`](https://github.com/kaitakuai/mlnode-foundry/blob/main/docs/adr/0014-residual-fork-permanent-infra.md).
This file is a 1-page summary so an offline reader of the `gonka-poc` repo
can find the rationale for the two-artifact (plugin + thin fork) shipping
model without leaving the package.

**Date:** 2026-06-16 (canonical); local stub 2026-06-17.
**Owners:** @baychak
**Amends:** [ADR-0013](ADR-0013-poc-gate-ordering.md).

## Context (why this file exists)

`README.md` cites ADR-0014 from two places:

- Top-of-file status banner (`MIGRATION_FROM_FORK.md` + ADR-0014 explain
  the two-artifact relationship).
- "Why two artifacts" section (`see ADR-0014 in this repo's docs/adr/`).

`tests/gonka/README.md` also cites it alongside ADR-0013 when explaining
why `test_chat_priority_gating.py` was deleted in the arch refactor. Without
a local file, those citations are dead links. This stub closes the link-rot.
The full options-considered narrative (status-quo, monkey-patch sampler,
full-fork rebase, thin-fork-permanent) lives in the canonical mlnode-foundry
ADR — do not duplicate it here.

## Decision (the part the plugin's shipping model depends on)

Gonka PoC ships as **two artifacts** on purpose:

1. **`gonka-poc` plugin** (this repo) — everything reachable through vLLM's
   public extension surfaces:
   - `vllm.general_plugins` entry point.
   - `--worker-extension-cls gonka_poc.worker.PoCWorkerExtension` (the
     official `collective_rpc` surface).
   - `gonka-vllm-serve` composition wrapper around `build_app` (no source
     patches).
2. **`kaitakuai/vllm@poc-sampler-residual-vX.YY`** — a thin (6-commit,
   ~150-line) fork holding the sampler-stack residual: enforced-token
   sampling, per-request `logprobs_mode`, structured-output graceful
   degradation. Those touch private vLLM internals (`vllm/v1/sample/*`,
   `vllm/v1/structured_output/*`, `vllm/v1/worker/gpu_input_batch.py`)
   with no public hook today.

The original ADR-0013 plan was to upstream the sampler hooks to
`vllm-project/vllm` (Layer 3) and retire the fork. ADR-0014 marks that
exit strategy as **DEFERRED-INDEFINITELY**: Kaitaku does not have the
bandwidth or acceptance channel to drive upstream PRs through the
`vllm-project` review process. The thin fork is therefore treated as
**permanent infrastructure**, not a temporary bridge.

## What this means operationally

- Per vLLM minor: rebuild the residual fork as
  `vllm==0.23.0+gonka.samplerN` (mechanical hours, not days). REBASE.md on
  the residual branch documents the cherry-pick order.
- Contract tests on the residual branch (`tests/contract/`) pin the
  private sampler surface so upstream drift fires a CI alert before the
  manual rebase.
- Plugin (`gonka-poc`) and residual wheel are coupled by version:
  `pip install gonka-poc==X` requires `vllm==0.23.Y+gonka.samplerZ`.
- Once an upstream PR ever lands that retires part of the residual, that
  part migrates into the plugin and the corresponding compat shim and
  fork commit go away. The shipping model survives the migration —
  consumers still install one plugin.

## Why this lives in two places

- **Canonical:** mlnode-foundry owns the decision record (it covers
  foundry overlay impact, the CI workflow on the residual branch,
  REBASE.md procedure — all out of scope for the plugin source tree).
- **Local stub:** the plugin README cites this ADR by number; we keep the
  citation resolvable from inside the `gonka-poc` checkout. When
  mlnode-foundry's ADR-0014 changes, update this stub's summary so they
  stay coherent (drift risk acknowledged — bounded by the small surface
  this stub copies).

## Links

- Canonical ADR: [mlnode-foundry/docs/adr/0014-residual-fork-permanent-infra.md](https://github.com/kaitakuai/mlnode-foundry/blob/main/docs/adr/0014-residual-fork-permanent-infra.md)
- Residual fork branch: <https://github.com/kaitakuai/vllm/tree/poc-sampler-residual-v0.23>
- Companion ADR: [ADR-0013](ADR-0013-poc-gate-ordering.md) (PoC gate
  ordering contract)
- Migration guide: `MIGRATION_FROM_FORK.md` (Section 3 — per-commit fork
  inventory and the upstream-PR backlog)
