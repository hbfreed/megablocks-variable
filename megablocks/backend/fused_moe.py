# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Fused variable-width MoE forward for serving (no autograd).

The block-sparse path (`padded_gather` -> `stk.sdd` -> act -> `stk.dsd` ->
`padded_scatter`) is built around stk's 128x128 blocks. That is the right shape
for training, where each expert holds hundreds of tokens, but it is the wrong
shape for decode: with many experts and few tokens, every touched expert is
padded up to 128 rows to hold 1-3 real ones. On a 220-expert layer that is 128x
padding at one token and 49x at 64 tokens, which turns a weight-bandwidth-bound
operation into a compute-bound one.

The topology is block-diagonal-dense per expert -- an expert's token-blocks
connect to *all* of that expert's weight column-blocks and to nothing else -- so
there is no sparsity for a block-sparse GEMM to exploit. It is a grouped GEMM
over experts, and once expressed that way the row tile no longer has to be 128.

This module uses 16-row tiles (the tensor-core minimum) and fuses the pipeline
into four launches:

  1. `_plan_tiles`     one CTA: per-expert row/tile offsets from `bins`
  2. `_map_routes`     route -> padded row, and row -> source token
  3. `_gate_up_silu`   gathers x rows, both projections, SiLU-mul, in one pass
  4. `_down_proj`      the down projection

followed by the existing `_scatter_reduce`, which already consumes exactly the
`route_rows` mapping produced here. Nothing needs a host sync: every buffer is
sized from a bound computed on the host from `num_tokens`, `top_k` and the
static expert widths, and planned-but-unused tiles carry expert id -1 and exit.

Widths must be multiples of `BLOCK_N` (128), which the checkpoint format already
guarantees.
"""

import torch
import triton
import triton.language as tl


# Routing is a counting sort by expert id, and `histogram` + `cumsum` are that
# sort's own intermediate steps -- so counting, scanning, and permuting fuse
# into three small kernels with no CUB dependency at all. This replaces
# ops.sort + ops.histogram + ops.inclusive_cumsum + a separate route-mapping
# pass (6 launches, 2 of them CUB) with 3, and never materializes the sorted
# `bin_ids` / `indices` / `bins` arrays.
#
# Intra-bin order is arbitrary here (the rank comes from an atomic cursor, not
# a stable sort). That is safe: a row's output is an independent dot product
# over its own token, so which row inside an expert's range a route occupies
# cannot change any value, and `_scatter_reduce` sums a token's contributions
# in fixed top_k order. Verified bit-identical run to run.
@triton.jit
def _count_experts(
    expert_ids,
    counts,
    num_routes,
    NUM_EXPERTS: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Out-of-range lanes land on a sentinel bin that the masked store discards,
    # rather than on bin 0 which would corrupt expert 0's count.
    e = tl.load(expert_ids + off, mask=off < num_routes, other=NUM_EXPERTS)
    local = tl.histogram(e, BINS)
    b = tl.arange(0, BINS)
    tl.atomic_add(counts + b, local, mask=b < NUM_EXPERTS)


# One CTA. Derives, from the per-expert token counts, where each expert's rows
# start in the padded buffer and which expert owns each row tile. Serial across
# experts is fine: this is O(num_experts) on a single block.
@triton.jit
def _plan_tiles(
    counts,
    expert_row_start,
    cursor,
    tile_expert,
    route_tokens,
    # Runtime, NOT constexpr: this is a function of num_tokens, which varies
    # continuously under vLLM's batching. As a constexpr it forced a fresh
    # Triton compile per batch shape — ~11s of JIT storm during real serving.
    mtile_bound,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FILL: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E)
    live = e < NUM_EXPERTS
    n = tl.load(counts + e, mask=live, other=0)
    tl.store(cursor + e, 0, mask=live)
    tiles = tl.where(live, (n + BLOCK_M - 1) // BLOCK_M, 0)

    # Exclusive scan: tiles are laid out expert-major and every tile is exactly
    # BLOCK_M rows, so a tile's first row is just its global index * BLOCK_M.
    start = tl.cumsum(tiles) - tiles
    tl.store(expert_row_start + e, start * BLOCK_M, mask=live)

    # Invalidate first, then stamp the live entries. A tile whose expert stays
    # -1 was never claimed and its program exits immediately; a row whose token
    # stays -1 is intra-expert padding and contributes zero.
    total_tiles = tl.sum(tiles)
    for base in range(0, mtile_bound, FILL):
        o = base + tl.arange(0, FILL)
        tl.store(tile_expert + o, -1, mask=o < mtile_bound)
    for base in range(0, total_tiles * BLOCK_M, FILL):
        o = base + tl.arange(0, FILL)
        tl.store(route_tokens + o, -1, mask=o < total_tiles * BLOCK_M)

    for j in range(tl.max(tiles)):
        tl.store(tile_expert + start + j, e, mask=live & (j < tiles))


# The permute step of the counting sort. Each route claims the next free slot
# in its expert's row range via an atomic cursor, which is both its rank and its
# padded row. Writes both directions: `route_tokens` tells the GEMM which token
# to gather, `route_rows` tells the final reduction where each of a token's
# top_k contributions landed.
@triton.jit
def _scatter_routes(
    expert_ids,
    route_tokens,
    route_rows,
    cursor,
    expert_row_start,
    num_routes,  # runtime, not constexpr -- see _plan_tiles
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    m = r < num_routes
    e = tl.load(expert_ids + r, mask=m, other=0)
    rank = tl.atomic_add(cursor + e, 1, mask=m)
    row = tl.load(expert_row_start + e, mask=m, other=0) + rank
    tl.store(route_tokens + row, r // TOP_K, mask=m)
    tl.store(route_rows + r, row, mask=m)


# h[row, :w_e] = silu(x[tok] @ Wg_e) * (x[tok] @ Wu_e), gathering x on the fly.
# Padding rows load x as zero, so they produce exact zeros rather than garbage.
#
# The activation is applied to the fp32 accumulators, so the gate/up
# intermediates never reach memory. That removes a full read+write of the
# intermediate *and* is more accurate than the block-sparse path, which stores
# both projections as bf16 before multiplying them.
@triton.jit
def _gate_up_silu(
    x,
    wg,
    wu,
    h,
    route_tokens,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
    TOTAL_W: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    mt = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return
    if nt >= tl.load(expert_nblocks + e):
        return

    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(route_tokens + rows)
    valid = tok >= 0
    xp = x + tl.maximum(tok, 0)[:, None] * HIDDEN

    local = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    col = tl.load(expert_col_off + e) + local

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, HIDDEN, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(xp + kk[None, :], mask=valid[:, None], other=0.0)
        off = kk[:, None] * TOTAL_W + col[None, :]
        acc_g += tl.dot(a, tl.load(wg + off))
        acc_u += tl.dot(a, tl.load(wu + off))

    gate = acc_g * tl.sigmoid(acc_g)
    tl.store(h + rows[:, None] * MAX_W + local[None, :],
             (gate * acc_u).to(h.dtype.element_ty))


# y[row, :] = h[row, :w_e] @ Wd_e. The contraction length is the expert's own
# width, a runtime loop bound; widths are multiples of BLOCK_N so BLOCK_K
# divides them evenly and no K masking is needed.
@triton.jit
def _down_proj(
    h,
    wd,
    y,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
    MAX_W: tl.constexpr,
    WIDTH_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    mt = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return

    width = tl.load(expert_nblocks + e) * WIDTH_BLOCK
    coff = tl.load(expert_col_off + e)
    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = nt * BLOCK_N + tl.arange(0, BLOCK_N)

    hp = h + rows[:, None] * MAX_W
    wdp = wd + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, width, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(hp + kk[None, :])
        b = tl.load(wdp + (coff + kk)[:, None] * HIDDEN)
        acc += tl.dot(a, b)

    tl.store(y + rows[:, None] * HIDDEN + cols[None, :],
             acc.to(y.dtype.element_ty))


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


class FusedMoEPlan:
    """Static, per-module description of the packed expert layout.

    Everything here depends only on the expert widths, so it is built once and
    reused for every forward. Keeping it off the forward path is deliberate: the
    per-layer Python cost is what dominates decode.
    """

    def __init__(self, widths, hidden_size, device, block_n=128):
        if any(w <= 0 or w % block_n for w in widths):
            raise ValueError(f'expert widths must be positive multiples of {block_n}')
        self.widths = list(widths)
        self.num_experts = len(widths)
        self.hidden_size = hidden_size
        self.block_n = block_n
        self.max_width = max(widths)
        self.total_width = sum(widths)
        self.block_e = _next_pow2(self.num_experts)

        offsets = [0]
        for w in widths:
            offsets.append(offsets[-1] + w)
        self.expert_col_off = torch.tensor(offsets[:-1], dtype=torch.int32,
                                           device=device)
        self.expert_nblocks = torch.tensor([w // block_n for w in widths],
                                           dtype=torch.int32, device=device)
        self.expert_row_start = torch.empty(self.num_experts, dtype=torch.int32,
                                            device=device)
        self.max_nblocks = self.max_width // block_n
        # Scratch for the counting sort, allocated once. `+ 1` leaves room for
        # the out-of-range sentinel bin that _count_experts discards.
        self.count_bins = _next_pow2(self.num_experts + 1)
        self.counts = torch.empty(self.num_experts, dtype=torch.int32,
                                  device=device)
        self.cursor = torch.empty(self.num_experts, dtype=torch.int32,
                                  device=device)

    def tile_config(self, num_tokens, top_k):
        """Row tile / warps / stages for this batch size.

        The only thing that matters is how many routes land on the average
        expert: a tile taller than that is pure padding, a tile shorter than
        that wastes arithmetic intensity. Measured on the 220-expert served
        layer (RTX 3090), the crossovers sit at ~16 and ~32 routes/expert.
        """
        per_expert = num_tokens * top_k / self.num_experts
        if per_expert < 16:
            return 16, 8, 3
        if per_expert < 32:
            return 32, 4, 2
        return 64, 4, 2

    def bounds(self, num_tokens, top_k, block_m):
        """Host-side upper bounds on the padded row / tile counts.

        sum_e ceil(n_e / M) <= (#experts with a token) + floor(sum_e n_e / M),
        which needs no knowledge of the routing and so needs no host sync.
        """
        num_routes = num_tokens * top_k
        m_tiles = min(self.num_experts, num_routes) + num_routes // block_m
        return m_tiles, m_tiles * block_m


def fused_moe_forward(
    x,
    top_weights,
    expert_ids,
    plan,
    top_k,
    w_gate,
    w_up,
    w_down,
    num_warps=None,
    num_stages=None,
):
    """Gated MoE forward over packed variable-width experts.

    `expert_ids` and `top_weights` are the flattened (tokens * top_k) routed
    expert ids (int32) and routing weights (x's dtype), in token-major order --
    i.e. `top_e.flatten()`, straight from topk, with no sorting needed. Returns
    (tokens, hidden). Inference only -- nothing here is differentiable.
    """
    from megablocks.backend import kernels

    num_tokens, hidden = x.shape
    num_routes = num_tokens * top_k
    if hidden != plan.hidden_size:
        raise ValueError(f'expected hidden {plan.hidden_size}, got {hidden}')
    if expert_ids.numel() != num_routes:
        raise ValueError(
            f'expected {num_routes} expert ids, got {expert_ids.numel()}',
        )

    bn = plan.block_n
    bm, warps, stages = plan.tile_config(num_tokens, top_k)
    num_warps = warps if num_warps is None else num_warps
    num_stages = stages if num_stages is None else num_stages
    m_tiles, row_bound = plan.bounds(num_tokens, top_k, bm)
    dev = x.device

    tile_expert = torch.empty(m_tiles, dtype=torch.int32, device=dev)
    route_tokens = torch.empty(row_bound, dtype=torch.int32, device=dev)
    route_rows = torch.empty(num_routes, dtype=torch.int32, device=dev)

    plan.counts.zero_()
    _count_experts[(triton.cdiv(num_routes, 1024),)](
        expert_ids,
        plan.counts,
        num_routes,
        NUM_EXPERTS=plan.num_experts,
        BINS=plan.count_bins,
        BLOCK=1024,
        num_warps=4,
    )
    _plan_tiles[(1,)](
        plan.counts,
        plan.expert_row_start,
        plan.cursor,
        tile_expert,
        route_tokens,
        m_tiles,
        NUM_EXPERTS=plan.num_experts,
        BLOCK_E=plan.block_e,
        BLOCK_M=bm,
        FILL=1024,
        num_warps=4,
    )
    _scatter_routes[(triton.cdiv(num_routes, 256),)](
        expert_ids,
        route_tokens,
        route_rows,
        plan.cursor,
        plan.expert_row_start,
        num_routes,
        TOP_K=top_k,
        BLOCK_X=256,
        num_warps=4,
    )

    h = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
    _gate_up_silu[(m_tiles, plan.max_nblocks)](
        x,
        w_gate,
        w_up,
        h,
        route_tokens,
        tile_expert,
        plan.expert_col_off,
        plan.expert_nblocks,
        HIDDEN=hidden,
        TOTAL_W=plan.total_width,
        MAX_W=plan.max_width,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=64,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    y = torch.empty((row_bound, hidden), dtype=x.dtype, device=dev)
    _down_proj[(m_tiles, hidden // bn)](
        h,
        w_down,
        y,
        tile_expert,
        plan.expert_col_off,
        plan.expert_nblocks,
        HIDDEN=hidden,
        MAX_W=plan.max_width,
        WIDTH_BLOCK=bn,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=64,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = torch.empty((num_tokens, hidden), dtype=x.dtype, device=dev)
    kernels._scatter_reduce[(num_tokens,)](
        out,
        y,
        route_rows,
        top_weights,
        NUM_COLUMNS=hidden,
        TOP_K=top_k,
        SCALE=top_weights is not None,
    )
    return out
