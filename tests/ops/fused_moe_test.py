# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the fused variable-width MoE forward.

Checked against a dense fp32 expert loop rather than against the block-sparse
path: both are bf16 approximations of the same thing, and the fused path is the
more accurate of the two (it keeps the gate/up intermediates in fp32 registers
instead of storing them as bf16 before the multiply), so requiring agreement
between them would be the wrong test.
"""

import pytest
import torch
import triton
import torch.nn.functional as F

from megablocks.backend.fused_moe import FusedMoEPlan, fused_moe_forward

# (widths, hidden, top_k, num_tokens)
_CASES = [
    ([128] * 8, 256, 2, 1),                       # single token
    ([128] * 8, 256, 8, 5),                       # top_k == num_experts
    ([128, 512, 256, 384], 256, 1, 64),           # top_k == 1
    ([512] * 4, 512, 2, 3),                       # wide experts, few tokens
    ([128] * 64, 512, 4, 17),                     # many narrow experts
    ([128, 256, 384, 512] * 16, 1024, 6, 200),    # ragged, mid batch
    ([128, 256, 384, 512] * 55, 2048, 8, 1),      # served shape, decode
    ([128, 256, 384, 512] * 55, 2048, 8, 64),     # served shape, batched decode
    ([128, 256, 384, 512] * 55, 2048, 8, 2048),   # served shape, prefill
]


def _reference(x, top_w, top_e, w_gate, w_up, w_down, widths):
    offsets = [0]
    for w in widths:
        offsets.append(offsets[-1] + w)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(len(widths)):
        lo, hi = offsets[e], offsets[e + 1]
        rows, slots = torch.where(top_e == e)
        if rows.numel() == 0:
            continue
        xi = xf[rows]
        h = F.silu(xi @ w_gate[:, lo:hi].float()) * (xi @ w_up[:, lo:hi].float())
        out.index_add_(
            0, rows,
            (h @ w_down[lo:hi].float()) * top_w[rows, slots, None].float(),
        )
    return out


def _run(widths, hidden, top_k, num_tokens, seed=0):
    torch.manual_seed(seed)
    dev = torch.device('cuda')
    total = sum(widths)
    w_gate = torch.randn(hidden, total, device=dev, dtype=torch.bfloat16) * 0.02
    w_up = torch.randn(hidden, total, device=dev, dtype=torch.bfloat16) * 0.02
    w_down = torch.randn(total, hidden, device=dev, dtype=torch.bfloat16) * 0.02
    x = torch.randn(num_tokens, hidden, device=dev, dtype=torch.bfloat16)

    logits = torch.randn(num_tokens, len(widths), device=dev)
    top_w, top_e = torch.topk(F.softmax(logits, dim=1, dtype=torch.float), top_k, -1)
    top_w = top_w.to(torch.bfloat16)

    plan = FusedMoEPlan(widths, hidden, dev)
    got = fused_moe_forward(x, top_w.flatten(), top_e.flatten().int(), plan,
                            top_k, w_gate, w_up, w_down)
    want = _reference(x, top_w, top_e, w_gate, w_up, w_down, widths)
    return (got.float() - want).norm() / want.norm()


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.parametrize('widths, hidden, top_k, num_tokens', _CASES)
def test_matches_dense_reference(widths, hidden, top_k, num_tokens):
    rel = _run(widths, hidden, top_k, num_tokens)
    assert rel < 5e-3, f'relative error {rel:.3e} too large for bf16'


@pytest.mark.gpu
@pytest.mark.world_size(1)
def test_row_bound_is_never_exceeded():
    """The buffer bound is computed on the host with no knowledge of routing.

    Worst case for padding is one token per expert, which forces a separate
    (mostly empty) row tile for every expert.
    """
    widths = [128] * 64
    plan = FusedMoEPlan(widths, 256, torch.device('cuda'))
    for num_tokens in (1, 3, 8, 64, 512, 4096):
        for top_k in (1, 2, 8):
            block_m, _, _ = plan.tile_config(num_tokens, top_k)
            m_tiles, row_bound = plan.bounds(num_tokens, top_k, block_m)
            # Exact worst case: every expert holds at least one token, so each
            # rounds a partial tile up.
            routes = num_tokens * top_k
            worst = 0
            remaining = routes
            for _ in range(min(len(widths), routes)):
                take = max(1, remaining // max(min(len(widths), routes), 1))
                worst += -(-take // block_m)
                remaining -= take
            assert m_tiles >= worst, (num_tokens, top_k, m_tiles, worst)
            assert row_bound == m_tiles * block_m


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.parametrize('num_tokens', [1, 7, 64, 512, 4096])
def test_counting_sort_is_a_bijection(num_tokens):
    """The permute step ranks routes with an atomic cursor, not a stable sort.

    Order inside an expert's range is therefore arbitrary, which is fine for the
    result -- but it must still be a bijection: every route gets exactly one
    row, inside its own expert's range, and no two routes collide.
    """
    from megablocks.backend import fused_moe as fm

    widths, hidden, top_k = [128, 256, 384, 512] * 12, 512, 4
    dev = torch.device('cuda')
    torch.manual_seed(num_tokens)
    top_e = torch.stack([
        torch.randperm(len(widths), device=dev)[:top_k] for _ in range(num_tokens)
    ]).int().contiguous()
    flat = top_e.flatten()

    plan = FusedMoEPlan(widths, hidden, dev)
    bm, _, _ = plan.tile_config(num_tokens, top_k)
    m_tiles, row_bound = plan.bounds(num_tokens, top_k, bm)
    tile_expert = torch.empty(m_tiles, dtype=torch.int32, device=dev)
    route_tokens = torch.empty(row_bound, dtype=torch.int32, device=dev)
    route_rows = torch.empty(flat.numel(), dtype=torch.int32, device=dev)

    plan.counts.zero_()
    fm._count_experts[(triton.cdiv(flat.numel(), 1024),)](
        flat, plan.counts, flat.numel(), NUM_EXPERTS=plan.num_experts,
        BINS=plan.count_bins, BLOCK=1024, num_warps=4)
    fm._plan_tiles[(1,)](
        plan.counts, plan.expert_row_start, plan.cursor, tile_expert,
        route_tokens, m_tiles, NUM_EXPERTS=plan.num_experts,
        BLOCK_E=plan.block_e, BLOCK_M=bm, FILL=1024, num_warps=4)
    fm._scatter_routes[(triton.cdiv(flat.numel(), 256),)](
        flat, route_tokens, route_rows, plan.cursor, plan.expert_row_start,
        flat.numel(), TOP_K=top_k, BLOCK_X=256, num_warps=4)

    # counts match a torch reference
    ref_counts = torch.bincount(flat.long(), minlength=len(widths))
    assert torch.equal(plan.counts.long(), ref_counts), 'histogram mismatch'

    # every route maps to a distinct row
    rows = route_rows.long()
    assert rows.numel() == torch.unique(rows).numel(), 'row collision'
    assert int(rows.min()) >= 0 and int(rows.max()) < row_bound, 'row out of range'

    # each route's row falls inside its own expert's range
    starts = plan.expert_row_start.long()
    lo = starts[flat.long()]
    hi = lo + ((ref_counts[flat.long()] + bm - 1) // bm) * bm
    assert bool(((rows >= lo) & (rows < hi)).all()), 'route left its expert range'

    # route_tokens agrees with the inverse map
    assert torch.equal(route_tokens[rows], (torch.arange(
        flat.numel(), device=dev) // top_k).int()), 'route_tokens mismatch'


@pytest.mark.gpu
@pytest.mark.world_size(1)
def test_output_is_deterministic_despite_atomic_ordering():
    """Intra-bin order varies run to run; the output must not."""
    ref = None
    for _ in range(8):
        rel = _run([128, 256, 384, 512] * 20, 1024, 6, 300, seed=7)
        if ref is None:
            ref = rel
        assert rel == ref, f'nondeterministic output: {rel} vs {ref}'
