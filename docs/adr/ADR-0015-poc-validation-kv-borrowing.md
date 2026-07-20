# ADR-0015 — PoC validation on leased KV blocks (inference keeps running)

**Status:** Accepted (2026-07-20).
**Port of:** gonka-ai/vllm branch `qd/combine-poc-and-inference`
(052648bf4, 590616ab0, 32fda8c4f, cbe5380fc) into the plugin architecture.

## Context

The plugin's `/api/v1/pow/generate` (validation) path never activated the
gate nor aborted inference, yet its forward overwrote KV blocks `0..N` —
silent corruption of live inference KV, the exact failure mode Gleb warned
about. Upstream solved it for the in-tree fork by *borrowing* free blocks
from the BlockPool so validation and inference coexist; mining
(`/init/generate`) keeps the abort-everything regime.

## Decision

1. **Shared lease (multi-group variant "a").** vLLM keeps ONE BlockPool per
   engine; the block-id namespace is shared by every kv-cache group
   (per-group state is only block tables). One lease of
   `num_nonces × max_g ceil(seq_len/block_size_g)` pool blocks reserves the
   ids' byte ranges in EVERY group's tensors at once — valid for
   single-group and multi-group (DeepSeek-V4, GLM DSA) models alike.
2. **Sizing lives in EngineCore** (`gonka_poc_borrow_blocks(num_nonces,
   seq_len)` → `{block_ids, blocks_per_seq}`): the frontend-visible scalar
   `cache_config.block_size` undercounts by up to 8× on multi-group models.
   Methods are injected class-level at plugin-register time (general
   plugins load inside the engine-core process) and called over the
   UTILITY RPC — no vllm fork changes.
3. **Worker expands pool ids per group** by the kernel-split ratio
   `r = manager_bs/kernel_bs` (`_borrowed_layout`): slot of token *t* =
   `L[i][t//mbs]·mbs + t%mbs`; table entry *k* = `L[i][k//r]·r + k%r`.
   Physical ids enter ONLY address translation, never attention math —
   artifacts stay bit-identical across block choices (prover and validator
   always hold different blocks).
4. **Reservation CM** (`poc_reservation`): per-process FIFO lock ×
   reserve → forwards → return; escalation aborts inference once and
   re-borrows; final fallback = legacy in-place layout **with inference
   aborted first** (restores the missing safety invariant) and a
   `reset_prefix_cache()` on exit (in-place rounds clobber cached blocks
   without evicting their hashes). Mining also resets the prefix cache
   when its loop ends.
5. **KV-scratch embeds reuse removed** (upstream removed it the same way):
   it wrote outside any lease and silently ignored `poc_stronger_rng`.
   Fresh-buffer vectors are numerically identical on the flag-off path.
6. **Feature detection:** `GET /api/v1/pow/versions` advertises
   `poc_validation_inference: true`; engines without the injected methods
   degrade to the abort-based path automatically.

## Consequences

- Validation no longer stalls or corrupts inference; mining priority and
  its bit-path are untouched (legacy layout unchanged byte-for-byte).
- Known limits: the reservation lock is per API-server process
  (`--api-server-count > 1` → bound degrades to P×); DP>1 engine cores are
  out of scope; during a legacy-fallback validation new admissions are not
  gated (rare: pool must stay exhausted even after an abort).
- Hardware A/B (two different leases → bit-compare artifacts) belongs in
  the V4 experiment programme before production trust.
