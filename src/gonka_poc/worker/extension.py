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
    self.device, self.rank, self.vllm_config

Invocation (from the API server / async engine):
    await async_llm.collective_rpc(
        "execute_poc_forward",
        args=(),
        kwargs={"payload": ..., ...},
        timeout=POC_RPC_TIMEOUT_S,
    )

CONTRACT WARNINGS:
- Method names MUST NOT collide with any public Worker attribute -- vLLM
  asserts ``not hasattr(worker_class, attr)`` at init_worker time. Keep the
  ``execute_poc_*`` prefix unique.
- Return values must be msgpack-serialisable; do NOT return tensors. Return
  digests / dicts of bytes / ints.
- Every TP/PP rank executes the method; the API server typically uses
  ``result[0]`` (driver). Wrap GPU work in rank-aware guards if needed.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# NOTE: keep imports light at module scope -- this file is imported in every
# worker process during init_worker. Heavy imports (torch, vllm.poc.*) are
# deferred into method bodies.


class PoCWorkerExtension:
    """Add-only methods reachable from ``collective_rpc``.

    See module docstring for the full contract.
    """

    # ------------------------------------------------------------------ #
    # Lifecycle / install
    # ------------------------------------------------------------------ #

    def execute_poc_install(self) -> Dict[str, Any]:
        """One-shot install: rebind ``model_runner.execute_model`` (or its
        forward) if PoC needs to intercept the live serving forward.

        Called once from the API server at startup via
        ``collective_rpc("execute_poc_install")`` BEFORE any inference traffic.

        NOTE(layer-2): wire to :mod:`gonka_poc.poc.poc_model_runner` install
        hook once the compat shim ``gonka_poc._compat.v0_23`` is settled
        (depends on the CommonAttentionMetadata constructor signature
        captured by ``tests/contract/test_v0_23_api_surface.py``).
        """
        # NOTE(hardware-validation): exercise on a real GPU worker and confirm
        # the model_runner.execute_model attribute is the right rebind point in
        # vLLM 0.23 (the v0.15 PoC fork patched a different symbol).
        return {"installed": False, "reason": "stub"}

    # ------------------------------------------------------------------ #
    # PoC forward (the actual GPU work)
    # ------------------------------------------------------------------ #

    def execute_poc_forward(
        self,
        payload: Dict[str, Any],
        *,
        batch_size: int = 32,
        timeout_ms: int = 60000,
    ) -> Dict[str, Any]:
        """Execute one PoC forward pass on this worker rank.

        Args:
            payload: dict with PoC params: nonces (list[int]), block_hash,
                public_key, seq_len, k_dim, poc_stronger_rng.
            batch_size: passes per forward.
            timeout_ms: soft deadline; advisory only -- the engine-side
                collective_rpc supplies the hard timeout.

        Returns:
            ``{"artifacts": [{"nonce": int, "vector_b64": str}, ...],
               "rank": int, "elapsed_ms": float}``

            Keep the payload msgpack-friendly. Do NOT return torch tensors.

        Bindings to private attributes (resolved via compat shim):
            self.model_runner.kv_caches[:N]   -- scratch space (PoC reuses
                blocks starting at index 0, see poc_model_runner.py)
            self.model_runner.model           -- nn.Module forward target
            self.model_runner.attn_metadata_builders -- for CommonAttentionMetadata
        """
        # NOTE(layer-1): port the body of vllm/poc/poc_model_runner.py
        # ``run_forward`` here. The current implementation in
        # ``gonka_poc.poc.poc_model_runner`` still references upstream
        # private symbols; the compat shim in
        # ``gonka_poc._compat.v0_23`` is where private-API touchpoints
        # (CommonAttentionMetadata constructor, kv_caches layout) get
        # normalised against the running vllm version.
        #
        # Pseudocode:
        #     from gonka_poc.poc.poc_model_runner import run_poc_forward
        #     from gonka_poc._compat import current as compat
        #     attn_md = compat.build_common_attention_metadata(
        #         model_runner=self.model_runner,
        #         seq_len=payload["seq_len"],
        #     )
        #     artifacts = run_poc_forward(
        #         model_runner=self.model_runner,
        #         attn_metadata=attn_md,
        #         payload=payload,
        #         batch_size=batch_size,
        #     )
        #     return {"artifacts": artifacts, "rank": int(self.rank), ...}
        raise NotImplementedError(
            "execute_poc_forward: not yet wired to gonka_poc.poc.poc_model_runner. "
            "Pending layer-1 port via gonka_poc._compat.v0_23 once the contract "
            "test pins the CommonAttentionMetadata constructor for the installed "
            "vllm version."
        )

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
        kv_caches = getattr(self.model_runner, "kv_caches", None) if hasattr(self, "model_runner") else None
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
