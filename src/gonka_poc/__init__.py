"""gonka-poc: out-of-tree vLLM plugin for Gonka Proof-of-Compute (PoC v2).

This package ships as a standalone pip-installable plugin that targets a stock
``vllm==0.23.*`` wheel. It provides three integration surfaces:

1. ``vllm.general_plugins`` entry point (:func:`gonka_poc.plugin.register`)
   that registers PoC-aware model configs (e.g. Qwen3MoeForCausalLM custom_ops
   defaults) in every vLLM process (engine core, workers, inspection
   subprocess).
2. ``--worker-extension-cls gonka_poc.worker.PoCWorkerExtension`` exposing
   ``execute_poc_forward`` to vLLM's ``collective_rpc`` (replaces the previous
   monkey-patch against ``vllm.v1.engine.async_llm.AsyncLLM``).
3. ``gonka-vllm-serve`` console script that composes a FastAPI app on top of
   the stock vLLM OpenAI-compatible server, attaching the PoC API router and a
   503/abort gating middleware.

The fork-residual changes (sampler-stack + structured-output) are NOT shipped
here -- see ``MIGRATION_FROM_FORK.md`` for the disposition of every commit
from the source branch.
"""

__version__ = "0.1.0a0"
