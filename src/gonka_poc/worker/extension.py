"""PoCWorkerExtension -- mixed into the vLLM v0.23 GPU Worker via ``--worker-extension-cls``.

Activation:
    vllm serve <model> --worker-extension-cls gonka_poc.worker.PoCWorkerExtension

How vLLM wires this in (v0.23.0, verified):
    ``vllm/v1/worker/worker_base.py:261-287`` (WorkerWrapperBase.init_worker)
    resolves the qualname, asserts no attribute collisions with the concrete
    Worker, then does ``worker_class.__bases__ += (PoCWorkerExtension,)``.
    There is NO __init__ -- methods just become attributes on the live Worker.

Inside any method on this class, ``self`` is the live GPU Worker. Available
attributes:
    self.model_runner           -- GPUModelRunner (gpu_model_runner.py)
    self.model_runner.model     -- the nn.Module
    self.model_runner.kv_caches -- list[torch.Tensor]   (declared L525)
    self.model_runner.attn_groups -- list[list[AttentionGroup]] (L530)
    self.device, self.rank, self.vllm_config

Invocation (from the API server / async engine):
    await async_llm.collective_rpc(
        "execute_poc_forward",
        args=(),
        kwargs={"block_hash": ..., "public_key": ..., "nonces": [...],
                "seq_len": int, "k_dim": int, "poc_stronger_rng": bool},
        timeout=POC_RPC_TIMEOUT_S,
    )

CONTRACT WARNINGS:
- Method names MUST NOT collide with any public Worker attribute -- vLLM
  asserts ``not hasattr(worker_class, attr)`` at init_worker time. Keep the
  ``execute_poc_*`` prefix unique.
- Return values must be msgpack-serialisable; do NOT return tensors. Return
  digests / dicts of bytes / ints (artifacts carry vectors as base64 strings
  via :func:`gonka_poc.poc.data.encode_vector`).
- Every TP/PP rank executes the method; the API server aggregates results
  across ranks (PP non-last ranks return ``{"artifacts": [], "rank": ...}``
  because the underlying forward returns None for them).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# NOTE: keep imports light at module scope -- this file is imported in every
# worker process during init_worker. Heavy imports (torch, gonka_poc.poc.*)
# are deferred into method bodies.


class PoCWorkerExtension:
    """Add-only methods reachable from ``collective_rpc``.

    See module docstring for the full contract.
    """

    # ------------------------------------------------------------------ #
    # PoC forward (the actual GPU work)
    # ------------------------------------------------------------------ #

    def execute_poc_forward(
        self,
        *,
        block_hash: str,
        public_key: str,
        nonces: List[int],
        seq_len: int,
        hidden_size: Optional[int] = None,
        k_dim: int = 12,
        poc_stronger_rng: bool = False,
    ) -> Dict[str, Any]:
        """Execute one PoC forward pass on this worker rank.

        Args:
            block_hash, public_key: PoC scope tags fed into the seeded RNG.
            nonces: list[int] of nonces to compute artifacts for; one batched
                forward processes all of them.
            seq_len: sequence length per nonce.
            hidden_size: model hidden size. If ``None`` it is resolved from
                ``self.vllm_config.model_config.get_hidden_size()`` (so the
                API server doesn't have to thread it through every call).
            k_dim: artifact vector dimensionality (default 12).
            poc_stronger_rng: if True, use murmur-concat RNG path; default
                False (legacy seeded normal path).

        Returns:
            ``{"artifacts": [{"nonce": int, "vector_b64": str}, ...],
               "rank": int}``

            On PP non-last ranks the underlying forward returns ``None``
            (intermediate tensors were forwarded inter-rank); we mirror that
            with an empty artifact list so the caller can aggregate uniformly.

            Keep the payload msgpack-friendly. Do NOT return torch tensors.
        """
        # Deferred imports: pulling gonka_poc.poc.* requires a configured
        # vllm runtime (vllm.logger), which is only available inside the
        # worker process.
        from gonka_poc.poc.data import encode_vector
        from gonka_poc.poc.poc_model_runner import (
            DEFAULT_K_DIM,
            execute_poc_forward as _execute_poc_forward,
        )

        if not nonces:
            return {"artifacts": [], "rank": int(getattr(self, "rank", -1))}

        if hidden_size is None:
            # vllm_config.model_config.get_hidden_size() is the v0.23
            # canonical accessor; fall back to model_config directly if a
            # future restructure splits the config tree.
            try:
                hidden_size = int(
                    self.vllm_config.model_config.get_hidden_size()
                )
            except AttributeError:
                hidden_size = int(self.model_config.get_hidden_size())

        result = _execute_poc_forward(
            self,  # the live Worker; matches the ``worker`` param in poc_model_runner
            block_hash,
            public_key,
            list(nonces),
            int(seq_len),
            int(hidden_size),
            k_dim=int(k_dim) if k_dim is not None else DEFAULT_K_DIM,
            poc_stronger_rng=bool(poc_stronger_rng),
        )

        rank = int(getattr(self, "rank", -1))

        # PP non-last ranks return None from the underlying forward.
        if result is None:
            return {"artifacts": [], "rank": rank}

        vectors = result.get("vectors")
        result_nonces = result.get("nonces", [])

        artifacts: List[Dict[str, Any]] = []
        if vectors is not None and len(result_nonces) > 0:
            for i, nonce in enumerate(result_nonces):
                artifacts.append({
                    "nonce": int(nonce),
                    "vector_b64": encode_vector(vectors[i]),
                })

        return {"artifacts": artifacts, "rank": rank}

    # ------------------------------------------------------------------ #
    # Diagnostic / liveness pings (cheap, no GPU)
    # ------------------------------------------------------------------ #

    def execute_poc_ping(self) -> Dict[str, Any]:
        """Cheap health probe; returns rank metadata without touching GPU."""
        # ``self.rank``, ``self.device``, ``self.vllm_config`` are all set by
        # the GPU Worker before extensions are usable.
        return {
            "rank": int(getattr(self, "rank", -1)),
            "device": str(getattr(self, "device", "?")),
            "ext_version": "gonka_poc/0.1.0a0",
        }

    def execute_poc_describe_kv(self) -> Dict[str, Any]:
        """Report KV-cache tensor shape/dtype for the compat shim to verify.

        Returns a small dict only; do NOT return the tensors themselves.
        """
        kv_caches = (
            getattr(self.model_runner, "kv_caches", None)
            if hasattr(self, "model_runner")
            else None
        )
        if kv_caches is None:
            return {"available": False}
        if not kv_caches:
            return {"available": True, "n_layers": 0}
        head = kv_caches[0]
        return {
            "available": True,
            "n_layers": len(kv_caches),
            "head_shape": list(getattr(head, "shape", ())),
            "head_dtype": str(getattr(head, "dtype", "?")),
            "head_device": str(getattr(head, "device", "?")),
        }


# Public alias used in the ``--worker-extension-cls`` CLI string.
__all__ = ["PoCWorkerExtension"]
