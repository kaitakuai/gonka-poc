"""Unit tests for the borrowed-lease PoC KV layout (``_borrowed_layout``).

The lease design ("shared lease", port of gonka-ai/vllm
qd/combine-poc-and-inference): pool-unit block ids leased from the ONE
engine-wide BlockPool; per group the ids expand by the kernel-split ratio
``r = manager_block_size // kernel_block_size``. These tests pin the address
math CPU-only against a naive per-token walk, the slot↔table consistency
invariant, cross-sequence collision-freedom, and the fail-loud guards.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("vllm")  # poc_model_runner imports vllm.distributed
import torch  # noqa: E402

from gonka_poc.poc.poc_model_runner import _borrowed_layout  # noqa: E402


def _naive_slots(batch, seq_len, m_block, ids, stripe):
    """slot = L[i][t//m]*m + t%m — straight from the design note."""
    bps = math.ceil(seq_len / m_block)
    out = []
    for i in range(batch):
        lane = ids[i * stripe: i * stripe + bps]
        for t in range(seq_len):
            out.append(lane[t // m_block] * m_block + t % m_block)
    return torch.tensor(out, dtype=torch.long)


@pytest.mark.parametrize("g_block,m_block", [(16, 16), (64, 64), (64, 256), (8, 8), (4, 8)])
@pytest.mark.parametrize("seq_len", [1, 7, 16, 100])
def test_vectorized_matches_naive(g_block, m_block, seq_len):
    batch = 3
    bps = math.ceil(seq_len / m_block)
    stripe = bps + 1  # engine stripe may exceed this group's need
    # Non-contiguous, shuffled "pool" ids — nothing arange-like.
    ids = [513, 7, 1024, 99, 256, 3, 771, 42, 65, 1300, 11, 900][: batch * stripe]
    while len(ids) < batch * stripe:
        ids.append(2000 + len(ids))

    slot_mapping, block_table = _borrowed_layout(
        batch, seq_len, g_block, m_block, ids, stripe, device="cpu")

    assert torch.equal(slot_mapping, _naive_slots(batch, seq_len, m_block, ids, stripe))

    # Table shape & content: entry k of seq i == L[i][k//r]*r + k%r.
    r = m_block // g_block
    assert block_table.shape == (batch, bps * r)
    assert block_table.dtype == torch.int32
    for i in range(batch):
        lane = ids[i * stripe: i * stripe + bps]
        for k in range(bps * r):
            assert block_table[i, k].item() == lane[k // r] * r + k % r


@pytest.mark.parametrize("g_block,m_block,seq_len", [(64, 256, 300), (8, 8, 20), (16, 64, 65)])
def test_slot_table_consistency(g_block, m_block, seq_len):
    """The kernel-side address derived from the table must equal the slot.

    For token t: block_table[i][t//g]*g + t%g == slot_mapping[i*seq+t].
    This is the invariant the attention kernels rely on (write via
    slot_mapping, read via block_table) — if it breaks, KV reads miss the
    writes and hidden states silently degrade.
    """
    batch = 2
    bps = math.ceil(seq_len / m_block)
    stripe = bps
    ids = [i * 3 + 5 for i in range(batch * stripe)]
    slot_mapping, block_table = _borrowed_layout(
        batch, seq_len, g_block, m_block, ids, stripe, device="cpu")
    for i in range(batch):
        for t in range(seq_len):
            addr = block_table[i, t // g_block].item() * g_block + t % g_block
            assert addr == slot_mapping[i * seq_len + t].item(), (i, t)


def test_distinct_ids_no_slot_collisions():
    batch, seq_len, g, m = 4, 50, 16, 64
    stripe = math.ceil(seq_len / m)
    ids = [10, 3, 77, 21][: batch * stripe]
    slot_mapping, _ = _borrowed_layout(
        batch, seq_len, g, m, ids, stripe, device="cpu")
    assert slot_mapping.unique().numel() == batch * seq_len


def test_guard_non_divisible_split():
    with pytest.raises(ValueError, match="not a\n?.*multiple|not a"):
        _borrowed_layout(1, 16, 6, 16, [1, 2, 3], 3, device="cpu")


def test_guard_stripe_too_small():
    # seq 100 @ m_block 8 needs 13 blocks/seq; stripe 2 must fail loudly.
    with pytest.raises(ValueError, match="stripe"):
        _borrowed_layout(2, 100, 8, 8, list(range(1, 5)), 2, device="cpu")


def test_guard_lease_too_small():
    # stripe fits the group, but the id list is shorter than batch*stripe.
    with pytest.raises(ValueError, match="lease has"):
        _borrowed_layout(3, 16, 16, 16, [5, 6], 1, device="cpu")
