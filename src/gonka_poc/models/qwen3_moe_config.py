"""Qwen3MoeForCausalLM PoC defaults: custom_ops + compilation_config tail.

Ports the v0.15.1-fork commit 15ee09f11 ("feat(models): add
Qwen3MoeForCausalLMConfig with PoC custom_ops defaults") into the plugin
surface. The fork added an entry to vLLM's MODELS_CONFIG_MAP; we replicate
that as an idempotent seeding step that runs from
:func:`gonka_poc.plugin.register` in every vLLM process.

Approach: rather than patching MODELS_CONFIG_MAP, we mutate the live
``VllmConfig.compilation_config.custom_ops`` list once per process when the
loaded HF arch is Qwen3MoeForCausalLM. Detection happens at engine init via
the vllm.general_plugins entry point.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

# Custom-ops tail added by the v0.15.1 fork commit 15ee09f11. Each entry is a
# string in vLLM's ``+op`` / ``-op`` notation (custom_ops field semantics).
#
# NOTE(layer-1, hardware-validation): finalise the exact custom_ops list once
# we have a B200 / RTX PRO 6000 SE run on vllm 0.23.0 confirming which ops
# survive the upstream rewrites. Per project_b300_qwen235b_fp8 memory the
# upstream MoE backend selection has shifted (FlashInfer / TRITON / DeepGEMM)
# so the fork's set will likely need pruning.
QWEN3_MOE_POC_CUSTOM_OPS_DEFAULTS: tuple[str, ...] = (
    # Sentinel placeholders -- replace from the v0.15.1 fork's
    # MODELS_CONFIG_MAP[Qwen3MoeForCausalLM] entry once the contract test
    # confirms the v0.23.0 CustomOp registry shape.
    # "+rms_norm",
    # "+silu_and_mul",
)

# Architectures whose compilation_config we touch. List form so a future
# entry (MiniMax-M2.7, Kimi-K2.6) can be added without a code-shape change.
POC_AWARE_ARCHITECTURES: tuple[str, ...] = (
    "Qwen3MoeForCausalLM",
)


def merge_custom_ops(existing: Sequence[str], additions: Iterable[str]) -> List[str]:
    """Idempotent merge of custom_ops entries (preserve order, dedupe)."""
    seen = set(existing)
    merged: List[str] = list(existing)
    for op in additions:
        if op in seen:
            continue
        seen.add(op)
        merged.append(op)
    return merged


def apply_qwen3_moe_poc_defaults(vllm_config: object) -> bool:
    """Seed ``vllm_config.compilation_config.custom_ops`` for Qwen3MoE.

    Returns True if a mutation happened, False if the active arch is not
    PoC-aware (no-op).

    Bound late so we don't import vllm at plugin-registration time.
    """
    # NOTE(layer-1): swap to vllm.config import paths once contract test
    # pins ModelConfig.architecture and CompilationConfig.custom_ops names.
    model_cfg = getattr(vllm_config, "model_config", None)
    if model_cfg is None:
        return False
    arch_field = getattr(model_cfg, "architecture", None) or getattr(
        model_cfg, "architectures", None
    )
    archs: tuple[str, ...]
    if isinstance(arch_field, str):
        archs = (arch_field,)
    elif isinstance(arch_field, (list, tuple)):
        archs = tuple(str(a) for a in arch_field)
    else:
        archs = ()

    if not any(a in POC_AWARE_ARCHITECTURES for a in archs):
        return False

    compile_cfg = getattr(vllm_config, "compilation_config", None)
    if compile_cfg is None:
        return False
    custom_ops = getattr(compile_cfg, "custom_ops", None) or []
    merged = merge_custom_ops(custom_ops, QWEN3_MOE_POC_CUSTOM_OPS_DEFAULTS)
    try:
        compile_cfg.custom_ops = merged
    except AttributeError:
        # Some dataclass variants are frozen; fall back to setattr via
        # __dict__ as a last resort. NOTE(layer-1): confirm CompilationConfig
        # mutability in v0.23.0 contract test.
        compile_cfg.__dict__["custom_ops"] = merged
    return True


__all__ = [
    "POC_AWARE_ARCHITECTURES",
    "QWEN3_MOE_POC_CUSTOM_OPS_DEFAULTS",
    "apply_qwen3_moe_poc_defaults",
    "merge_custom_ops",
]
