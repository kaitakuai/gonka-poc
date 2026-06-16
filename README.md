# gonka-poc

Out-of-tree vLLM plugin implementing **Gonka Proof-of-Compute v2** for stock
`vllm==0.23.*` wheels. Ships as a single `pip install gonka-poc` -- no fork,
no source patches.

## What it provides

1. **PoC API router** under `/api/v1/pow/*` (init/generate/status/stop) +
   503/abort gating against `/v1/chat/completions` and `/v1/completions`
   while PoC generation is active.
2. **PoCWorkerExtension** -- `execute_poc_forward` reachable through vLLM's
   public `collective_rpc` (replaces the previous AsyncLLM monkey-patch).
3. **Model defaults** -- Qwen3MoeForCausalLM PoC `custom_ops` defaults
   applied via the `vllm.general_plugins` entry point in every process.

The sampler-stack residual (enforced-token sampling, `logprobs_mode`) is
**not** part of this plugin -- it remains as a thin fork until vLLM grows a
sampler-stack hook. See `MIGRATION_FROM_FORK.md`.

## Quick start

The plugin requires the kaitakuai-org **sampler-residual wheel** of vLLM
(`vllm==0.23.0+gonka.sampler1`) on the import path -- stock `vllm/vllm-openai:v0.23.0-cu129`
does **not** carry the enforced-token / `logprobs_mode` sampler edits the
plugin relies on. Use either the prebuilt overlay image:

```bash
docker run --rm -it --gpus all \
  -p 8000:8000 \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  ghcr.io/kaitakuai/vllm-sampler-residual:0.23.0+gonka.sampler1 \
  sh -c "pip install gonka-poc && \
         gonka-vllm-serve \
           --model Qwen/Qwen3-235B-A22B-FP8 \
           --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
           --tensor-parallel-size 4 \
           --enforce-eager \
           --max-model-len 100000 \
           --dtype auto"
```

...or install the residual wheel into the stock image at runtime via the
kaitakuai-org private index (replace `<kaitakuai-index>` with the index URL
provisioned for your operator):

```bash
docker run --rm -it --gpus all \
  -p 8000:8000 \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  --build-arg KAITAKUAI_INDEX=<kaitakuai-index> \
  vllm/vllm-openai:v0.23.0-cu129 \
  sh -c "pip install vllm==0.23.0+gonka.sampler1 \
           --index-url <kaitakuai-index> \
           --extra-index-url https://pypi.org/simple && \
         pip install gonka-poc && \
         gonka-vllm-serve \
           --model Qwen/Qwen3-235B-A22B-FP8 \
           --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
           --tensor-parallel-size 4 \
           --enforce-eager \
           --max-model-len 100000 \
           --dtype auto"
```

The overlay image name above (`ghcr.io/kaitakuai/vllm-sampler-residual`) is
a placeholder; the canonical build lives in `mlnode-foundry` under
`profiles/gonka-poc/Dockerfile.overlay` (see `MIGRATION_FROM_FORK.md`
Section 2).

`gonka-vllm-serve` is a thin composition wrapper around
`vllm.entrypoints.openai.api_server`: it accepts every flag stock
`vllm serve` accepts (it re-uses `make_arg_parser` / `validate_parsed_serve_args`).
The PoC router and gating middleware are inserted between `build_app(...)`
and `serve_http(...)` -- no vLLM source is patched.

## Why the sampler-residual fork?

`gonka-poc` itself is a stock-wheels plugin, but it depends on a small set
of enforced-token / `logprobs_mode` edits in `vllm/v1/sample/*` that have no
extension point in `vllm==0.23.*`. Those edits live on a thin patch series
maintained by **kaitakuai** (not upstream `gonka-ai/gonka`, and not
`vllm-project/vllm`) at `kaitakuai/vllm @ poc-sampler-residual-v0.23`, and
ship as the wheel `vllm==0.23.0+gonka.sampler1`. For the per-commit
disposition, install order, and the upstream-PR backlog that retires this
fork, see `MIGRATION_FROM_FORK.md` Section 3 and ADR-0013.

## Layout

```
src/gonka_poc/
  poc/            -- PoC v2 module (callbacks, queue, gpu_random, manager, ...)
  worker/         -- PoCWorkerExtension (collective_rpc surface)
  entrypoint/     -- gonka-vllm-serve composer + 503 gating middleware
  models/         -- Qwen3MoE PoC custom_ops defaults
  _compat/        -- version-dispatched private-API shim (v0_23.py)
  plugin.py       -- vllm.general_plugins entry point
tests/
  contract/       -- vLLM private-surface drift detector (read-only)
  gonka/          -- PoC live + unit tests ported from the 0.15.1 fork
```

## Runtime requirements

* `vllm >= 0.23.0, < 0.24`
* `--enforce-eager` -- PoC forward MUST run eager (per
  `feedback_poc_eager_mandatory.md`). Compiled drift breaks cross-validator
  bit-compat.
* GPU profile: confirmed targets B200, B300, RTX PRO 6000 SE (Qwen3-235B-FP8).
  Validation gate per `feedback_hardware_validation_gate.md`.

## What's NOT here

See `MIGRATION_FROM_FORK.md` for:

* Foundry-profile deployment defaults (Dockerfile.overlay, engine-args.yaml,
  CI workflows) -- these live in **mlnode-foundry**, not the plugin.
* Sampler-stack residual (`vllm/v1/sample/*` edits) -- stays on the fork
  until upstream adds a sampler-stack hook.
* Structured-output graceful-degradation patch -- stays on the fork
  (private xgrammar internals; no plugin hook).
