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

```bash
docker run --rm -it --gpus all \
  -p 8000:8000 \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  vllm/vllm-openai:v0.23.0-cu129 \
  sh -c "pip install gonka-poc && \
         gonka-vllm-serve \
           --model Qwen/Qwen3-235B-A22B-FP8 \
           --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
           --tensor-parallel-size 4 \
           --enforce-eager \
           --max-model-len 100000 \
           --dtype auto"
```

`gonka-vllm-serve` is a thin composition wrapper around
`vllm.entrypoints.openai.api_server`: it accepts every flag stock
`vllm serve` accepts (it re-uses `make_arg_parser` / `validate_parsed_serve_args`).
The PoC router and gating middleware are inserted between `build_app(...)`
and `serve_http(...)` -- no vLLM source is patched.

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
