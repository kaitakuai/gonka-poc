# Pseudo input-ids derivation convention (PoC v2, token-id-routed architectures)

Status: **v1, proposed** (introduced with DeepSeek-V4 support, PR #14).
Any change to this derivation is a consensus-breaking change: prover and
validator MUST compute identical ids, and PoC vectors depend on them
(measured: changing the scheme moves vectors well above run noise).

## Why this exists

DeepSeek-V4's first `num_hash_layers` layers route MoE experts by token id
(`tid2eid[input_ids]`) and hard-fail on `input_ids=None`. PoC has no real
tokens — inputs are synthetic embeddings seeded from
`(block_hash, public_key, nonce)`. The ids are therefore derived from the
same seed material, through the same framework-independent murmur3 pipeline
as the embeddings: pure integer arithmetic, stable across torch versions and
implementations. `torch.Generator`-based derivations were deliberately
rejected (RNG algorithm is a torch implementation detail).

## Derivation (normative)

For a request `(block_hash, public_key, nonces, seq_len)` on a model with
vocabulary size `vocab` (from `hf_config.vocab_size`):

```
seed(nonce)  = int( sha256(f"{block_hash}_{public_key}_nonce{nonce}_input_ids")
                    .hexdigest()[:8], 16 )              # gpu_random._seed_from_string
keys         = [0, 1, ..., seq_len-1]                    # int32
ids[nonce,t] = murmur3_32(keys[t], seed(nonce)) % vocab  # gpu_random._batched_murmur3_32
dtype        = int32                                     # routing kernels expect Int
```

- The `_input_ids` suffix namespaces the seed away from the embedding stream
  (which uses the same string without the suffix).
- `murmur3_32` is the exact finalizer-included variant implemented in
  `gonka_poc/poc/gpu_random.py` (`_murmur3_32` / `_batched_murmur3_32`).
- Modulo bias at `vocab << 2^32` is negligible for routing purposes.
- Applied ONLY when `hf_config.model_type == "deepseek_v4"`; every other
  architecture receives `input_ids=None` (pre-existing behaviour).

## Reference vectors (frozen)

Inputs: `block_hash = "poc_ids_convention_v1_block"`,
`public_key = "poc_ids_convention_v1_pubkey"`, `vocab = 163840`
(DeepSeek-V4-Flash).

| nonce | seed (`_seed_from_string`) | ids[0..15] (seq_len=16) |
|---|---|---|
| 0 | 2980507924 | 119046, 160019, 98337, 39450, 94909, 163782, 59011, 57361, 156377, 36469, 139643, 43988, 50299, 147011, 12130, 86013 |
| 1 | 2917457512 | 137719, 146475, 40370, 95731, 105609, 72910, 88608, 153625, 112488, 156080, 83284, 17839, 70977, 48769, 15417, 65900 |
| 7 | 1725538525 | 132809, 151794, 33855, 107081, 64399, 133942, 119919, 3062, 136558, 105447, 18858, 162500, 55930, 32013, 122278, 101147 |

Integrity checksums for `seq_len = 256` (same inputs):

| nonce | sum(ids) | min | max |
|---|---|---|---|
| 0 | 20594062 | 181 | 163834 |
| 1 | 19523989 | 898 | 163477 |

These vectors are enforced by
`tests/unit/test_v4_metadata_layout.py::test_pseudo_ids_reference_vectors`;
an independent implementation (in-band PoC line) MUST reproduce them before
DeepSeek-V4 activation on the network.

## Change management

- Version this document (`v1`, `v2`, ...) and the seed suffix together: a new
  scheme MUST use a new suffix (e.g. `_input_ids_v2`) so mixed-version sets
  are never silently comparable.
- Nonce sets collected under different scheme versions are NOT comparable
  (measured drift ≈ 0.23 mean L2 — same order as honest cross-hardware
  distance).
