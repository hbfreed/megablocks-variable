# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Grouped variable-width MoE with autograd -- the training twin of fused_moe.

fused_moe.py established that the variable-width MoE is a grouped GEMM, not a
block-sparse one: an expert's token rows connect to all of that expert's weight
columns and to nothing else. This module extends that observation to training,
replacing `padded_gather -> stk.sdd -> SwiGLU -> stk.dsd -> padded_scatter`
(and with it the stk dependency and the nanomoe_ops topology extension) with
the same routing kernels fused_moe already uses plus grouped forward/backward
GEMMs.

Every Triton GEMM kernel here holds a SINGLE fp32 accumulator. That is a
deliberate trade measured on a 3090: fusing gate+up (or dWg+dWu) into one kernel
halves the gathered-x traffic, but the best fused gate+up tile still cost 0.13 ms
over the whole step, measured before the alignment note below. The CuTe backend
does fuse gate+up: with prefetch that kernel is 0.79 ms faster than the two
Triton launches (`cute_gate_up_rows`).

Forward routes with a stable counting sort -- `_count_blocks`, fused_moe's
`_plan_tiles`, then `_scatter_routes_stable`, which also puts each routing
weight w_r on its route's padded row. With token_rounding='nearest', a score
sort instead keeps each expert's count rounded half up to whole tiles,
dropping its lowest-scoring routes (`_count_route_keys`, `_round_counts`,
`_scatter_routes_ranked`). Then:

  `_proj_rows`      x @ W for one projection, gathering x rows on the fly;
                    called twice (gate, up), storing the pre-activations
                    (backward needs them -- the stk path stores the same two)
  h = w_r * silu(g) * u
                    fused into the up projection's output epilogue
  `_down_proj`      Triton or optional CuTe grouped GEMM on h
  `_scatter_reduce` unweighted top-k sum into the output (from kernels.py)

Folding the routing weight into h, as DeepGEMM's Mega MoE does, keeps the
(rows, hidden) down-projection output out of the saved tensors: backward
recovers dw_r from the intermediate instead, through the identity
<y_row, dout> = <silu(g) * u, dout @ Wd^T>.

Backward, all deterministic (fp32 accumulators, fixed reduction order, no
atomics, and a row layout that is a pure function of the routing):

  `_dgu`            da = gather(dout) @ Wd^T; per-tile partials of
                    dw_r = <da, silu(g) * u>; dh = w_r * da, then SwiGLU'
                    in-register -> dg, du; h again if it was not retained
  `_route_dw`       dw_r = sum of its row's partials, in tile order
  `_wgrad_rows`     Triton or CuTe h^T @ gather(dout) -> dWd
  `_wgrad_gather`   dW = gather(x)^T @ dg -> dWg, and with du -> dWu
  `_dx_rows_fused`  dx_rows = dg @ Wg^T + du @ Wu^T, using one fp32
                    accumulator in either Triton or CuTe
  `_scatter_reduce` (unweighted) top-k sum of dx_rows into dx

Padding is self-consistent everywhere: a padding row inside a claimed tile has
x gathered as zero in forward (so g = u = h = 0) and no token's dout in backward,
so with finite dout every wgrad contribution from padding is an exact zero. Rows
past the last claimed tile are never inside any expert's row range and are never
read. Weight gradients cover every weight block unconditionally, so untouched
experts get exact-zero grads rather than stale memory.
"""

import torch
import triton
import triton.language as tl

from megablocks.backend import kernels
from megablocks.backend.fused_moe import _down_proj, _plan_tiles

# Per-kernel launch configs, swept on a 3090 at the healing shape (keep50
# widths, 4096 tokens, top_k 8), then partly retuned at 275 W on a synthetic table
# with the same totals. Keys: (BLOCK_M, BLOCK_N, BLOCK_K, warps,
# stages). BLOCK_N on the width axis must be a multiple of 128 (the
# width-block granularity); on the hidden axis it must divide the hidden size.
_CFG = {
    'proj': (128, 128, 32, 4, 2),        # gate projection
    'proj_fused': (128, 128, 32, 4, 2),  # up projection + SwiGLU epilogue
    'down': (128, 256, 64, 8, 3),        # _down_proj on h
    'dgu': (128, 128, 64, 8, 3),         # _dgu
    'dx_fused': (128, 256, 32, 8, 3),    # gate + up input-gradient streams
    'wgrad_down': (128, 128, 32, 4, 2),  # _wgrad_rows: (BLOCK_W, BLOCK_H, K)
    'wgrad_gateup': (128, 128, 32, 4, 3),  # _wgrad_gather: (BLOCK_W, BLOCK_H, K)
}
# The row-tile layout is planned once with this block; every row-tile kernel
# uses the same BLOCK_M so tiles stay aligned to the planned buffers.
_ROW_BLOCK = 128

# Grid order sets which CTAs run side by side: program_id(0) advances fastest.
# The row-tile kernels put the width or hidden tile there, so a row tile's CTAs
# run together and share its rows through L2 instead of re-reading them from
# DRAM on a later sweep. _wgrad_rows does the same with a weight block's hidden
# tiles. _wgrad_gather already had that order. Reversing it there costs more
# than any other kernel gains.

# Triton sizes a load from the alignment it can prove, and a loaded
# expert_col_off has none. An operand indexed by it along its contiguous axis
# drops to one LDG per bf16 with no cp.async: W in _proj_rows and _dx_rows_fused,
# the rows in _wgrad_rows and _wgrad_gather. Offsets are multiples of the width
# block, so for any width block the K tiles allow `(off >> 3) << 3` changes no
# value, and it proves the 16 bytes cp.async needs. It saves 1.2 ms of a 10.4 ms CuTe step and 1.6 ms of 11.5 ms on Triton.

# Routes per counting-sort block; _count_blocks and _scatter_routes_stable must
# partition the routes identically.
_ROUTE_BLOCK = 1024


# Per-expert route counts for _plan_tiles, plus each block's own histogram for
# _scatter_routes_stable. Integer atomics on the totals are order-independent,
# so the counts are deterministic.
@triton.jit
def _count_blocks(
    expert_ids,
    counts,
    block_hist,
    num_routes,  # runtime, not constexpr -- see fused_moe._plan_tiles
    NUM_EXPERTS: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    off = b * BLOCK + tl.arange(0, BLOCK)
    # Out-of-range lanes land on the sentinel bin, as in
    # fused_moe._count_experts.
    e = tl.load(expert_ids + off, mask=off < num_routes, other=NUM_EXPERTS)
    local = tl.histogram(e, BINS)
    bins = tl.arange(0, BINS)
    tl.store(block_hist + b * BINS + bins, local)
    tl.atomic_add(counts + bins, local, mask=bins < NUM_EXPERTS)


# _plan_tiles, plus one exclusive scan of block_hist along the blocks, in place:
# row b becomes the routes to each bin in blocks before b, and a final row holds
# the totals, so row b + 1 - row b is still block b's own histogram. Scanning here
# adds no launch, and each permute program then reads two rows however many blocks
# there are. Integer sums, so the table is deterministic.
@triton.jit
def _plan_tiles_scan(
    counts,
    expert_row_start,
    cursor,
    tile_expert,
    route_tokens,
    block_hist,
    mtile_bound,  # runtime, not constexpr -- see fused_moe._plan_tiles
    route_blocks,  # runtime for the same reason
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    FILL: tl.constexpr,
    BINS: tl.constexpr,
):
    _plan_tiles(counts, expert_row_start, cursor, tile_expert, route_tokens,
                mtile_bound, NUM_EXPERTS, BLOCK_E, BLOCK_M, FILL)
    bins = tl.arange(0, BINS)
    earlier = tl.zeros((BINS,), dtype=tl.int32)
    for b in range(route_blocks):
        row = block_hist + b * BINS + bins
        hist = tl.load(row)
        tl.store(row, earlier)
        earlier += hist
    tl.store(block_hist + route_blocks * BINS + bins, earlier)


# The permute step of the counting sort, made stable: a route's rank within its
# expert is the number of earlier routes to that expert, so the padded layout
# -- and with it the summation order of every weight-gradient K loop -- is a
# pure function of expert_ids. fused_moe._scatter_routes claims rows with an
# atomic cursor, which is fine for inference, where rows are independent, but
# made dW differ bitwise between identical training steps. Earlier blocks
# contribute through _plan_tiles_scan's table; within the block, sorting on
# (expert, position) orders each expert's routes. Each route's weight goes onto
# its row; row_w is zero-initialized, so padding rows weigh 0.
@triton.jit
def _scatter_routes_stable(
    expert_ids,
    weights,
    block_hist,
    route_tokens,
    route_rows,
    row_w,
    expert_row_start,
    num_routes,  # runtime, not constexpr -- see fused_moe._plan_tiles
    NUM_EXPERTS: tl.constexpr,
    BINS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    bins = tl.arange(0, BINS)
    earlier = tl.load(block_hist + b * BINS + bins)
    hist = tl.load(block_hist + (b + 1) * BINS + bins) - earlier
    # Sorted position p of expert e's routes in this block is row base[e] + p.
    base = (tl.load(expert_row_start + bins, mask=bins < NUM_EXPERTS, other=0)
            + earlier - (tl.cumsum(hist, 0) - hist))

    pos = tl.arange(0, BLOCK)
    r = b * BLOCK + pos
    e = tl.load(expert_ids + r, mask=r < num_routes, other=NUM_EXPERTS)
    key = tl.sort(e * BLOCK + pos)  # ties broken by position: stable
    route = b * BLOCK + key % BLOCK
    live = route < num_routes
    row = tl.gather(base, key // BLOCK, 0) + pos
    tl.store(route_rows + route, row, mask=live)
    tl.store(route_tokens + row, route // TOP_K, mask=live)
    tl.store(row_w + row, tl.load(weights + route, mask=live).to(tl.float32),
             mask=live)


# Nearest token rounding, opt-in: each expert keeps its route count rounded
# half up to whole row tiles, dropping its lowest-scoring routes when that
# rounds down, but never rounding down to zero. Otherwise this is gpt_model's
# token_rounding='nearest' (and SonicMoE's token rounding). The keys are
# kernels._route_key's. Also accumulates the per-expert totals.
@triton.jit
def _count_route_keys(
    expert_ids,
    scores,
    counts,
    route_keys,
    num_routes,  # runtime, not constexpr -- see fused_moe._plan_tiles
    NUM_EXPERTS: tl.constexpr,
    BINS: tl.constexpr,
    SCORE_BITS: tl.constexpr,
    SCORE_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < num_routes
    e = tl.load(expert_ids + off, mask=m, other=NUM_EXPERTS)
    bins = tl.arange(0, BINS)
    tl.atomic_add(counts + bins, tl.histogram(e, BINS), mask=bins < NUM_EXPERTS)
    score = tl.load(scores + off, mask=m, other=0.0).to(tl.float32)
    tl.store(route_keys + off, kernels._route_key(e, score, SCORE_BITS, SCORE_SCALE),
             mask=m)


# One program: where each expert's routes start in the sorted keys, and how
# many of them it keeps -- min(count, count rounded half up to BLOCK_M), as in
# gpt_model's _block_tokens_per_expert, except that an expert under half a tile
# keeps its routes. Rounding those to zero stops the expert training for the step,
# and at small batches zeroes the whole layer, with no sign short of a host read.
# _plan_tiles then plans the kept counts; the original counts go to routed_counts
# for _route_stats.
@triton.jit
def _round_counts(
    counts,
    sorted_start,
    routed_counts,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E)
    live = e < NUM_EXPERTS
    n = tl.load(counts + e, mask=live, other=0)
    tl.store(routed_counts + e, n, mask=live)
    tl.store(sorted_start + e, tl.cumsum(n, 0) - n, mask=live)
    rounded = tl.maximum((n + BLOCK_M // 2) // BLOCK_M * BLOCK_M, BLOCK_M)
    tl.store(counts + e, tl.minimum(n, rounded), mask=live)


# Dropped routes and padding rows as fractions of all routes -- the quantities
# gpt_model.MoEMLP reports -- left on the device so logging them does not
# synchronize. An expert pads its kept routes up to whole BLOCK_M row tiles.
@triton.jit
def _route_stats(
    routed_counts,
    kept_counts,
    stats,
    num_routes,  # runtime, not constexpr -- see fused_moe._plan_tiles
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E)
    live = e < NUM_EXPERTS
    n = tl.load(routed_counts + e, mask=live, other=0)
    kept = tl.load(kept_counts + e, mask=live, other=0)
    rows = (kept + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    tl.store(stats, tl.sum(n - kept, 0).to(tl.float32) / num_routes)
    tl.store(stats + 1, tl.sum(rows - kept, 0).to(tl.float32) / num_routes)


# Rows for the kept routes, in descending-score order within each expert. A
# dropped route gets row -1, which _scatter_reduce and _route_dw skip, and no
# route_tokens entry, so it contributes nothing and receives no gradient.
@triton.jit
def _scatter_routes_ranked(
    sorted_keys,
    order,
    weights,
    sorted_start,
    kept,
    route_tokens,
    route_rows,
    row_w,
    expert_row_start,
    num_routes,  # runtime, not constexpr -- see fused_moe._plan_tiles
    SCORE_BITS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    m = p < num_routes
    e = tl.load(sorted_keys + p, mask=m, other=0) >> SCORE_BITS
    r = tl.load(order + p, mask=m, other=0).to(tl.int32)
    rank = p - tl.load(sorted_start + e, mask=m, other=0)
    keep = m & (rank < tl.load(kept + e, mask=m, other=0))
    row = tl.load(expert_row_start + e, mask=m, other=0) + rank
    tl.store(route_rows + r, tl.where(keep, row, -1), mask=m)
    tl.store(route_tokens + row, r // TOP_K, mask=keep)
    tl.store(row_w + row, tl.load(weights + r, mask=keep).to(tl.float32),
             mask=keep)


# One projection over gathered rows: out[row, :w_e] = x[tok(row)] @ W_e.
# Padding rows load x as zero, so they produce exact zeros.
@triton.jit
def _proj_rows(
    x,
    w,
    out,
    g_in,
    h_out,
    row_w,
    route_tokens,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
    TOTAL_W: tl.constexpr,
    MAX_W: tl.constexpr,
    WIDTH_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FUSE_SILU: tl.constexpr,
):
    nt = tl.program_id(0)
    mt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return
    if nt * BLOCK_N >= tl.load(expert_nblocks + e) * WIDTH_BLOCK:
        return

    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(route_tokens + rows)
    valid = tok >= 0
    xp = x + tl.maximum(tok, 0)[:, None] * HIDDEN

    local = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    col = ((tl.load(expert_col_off + e) >> 3) << 3) + local

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, HIDDEN, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(xp + kk[None, :], mask=valid[:, None], other=0.0)
        acc += tl.dot(a, tl.load(w + kk[:, None] * TOTAL_W + col[None, :]))

    dst = rows[:, None] * MAX_W + local[None, :]
    out_value = acc.to(out.dtype.element_ty)
    if FUSE_SILU:
        # Issued after the store below, this load cost ~450 B of epilogue
        # spills (ptxas, sm_86); here it costs ~50 B.
        ww = tl.load(row_w + rows[:, None])
    tl.store(out + dst, out_value)
    if FUSE_SILU:
        # Preserve the old numerics: the standalone kernel consumed the bf16
        # value after the projection store, not the fp32 accumulator.
        # _dgu repeats this operation order for recompute_activation.
        gg = tl.load(g_in + dst).to(tl.float32)
        uu = out_value.to(tl.float32)
        tl.store(h_out + dst,
                 (gg * tl.sigmoid(gg) * uu * ww).to(h_out.dtype.element_ty))


# da = dout[tok(row)] @ Wd^T for the expert's own width: the gradient with
# respect to the weighted activation w_r * h. h already carries the routing
# weight, so a row's output gradient is its token's own, unscaled. Padding rows
# load dout as zero, and a route dropped by token rounding has no row at all.
# The row dot of da with the unweighted h = silu(g) * u is this tile's share of
# dw_r, written to dw_part[row, nt] for _route_dw. dh = w_r * da then feeds the
# SwiGLU backward on the fp32 tile: dg = dh * u * silu'(g), du = dh * silu(g),
# with silu'(g) = sig(g) * (1 + g * (1 - sig(g))). Wd rows are loaded
# contiguously as (N, K) and transposed in-register. Gathering dout here and in
# the dWd kernels replaced a (rows, hidden) copy of its rows: at the primary
# shape, 0.18 ms (CuTe) and 0.14 ms (Triton) off a sustained step and 156 MiB
# off the peak. With FUSE_H it also writes h for recompute_activation, over the
# same real tiles the weight gradient reads.
@triton.jit
def _dgu(
    dout,
    wd,
    g_in,
    u_in,
    row_w,
    dg_out,
    du_out,
    dw_part,
    h_out,
    route_tokens,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
    MAX_W: tl.constexpr,
    NUM_NT: tl.constexpr,  # dw_part columns: the grid's width-tile count
    WIDTH_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FUSE_H: tl.constexpr,
):
    nt = tl.program_id(0)
    mt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return
    if nt * BLOCK_N >= tl.load(expert_nblocks + e) * WIDTH_BLOCK:
        return

    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    local = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    col = tl.load(expert_col_off + e) + local

    tok = tl.load(route_tokens + rows)
    valid = tok >= 0
    dyp = dout + tl.maximum(tok, 0)[:, None] * HIDDEN
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, HIDDEN, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(dyp + kk[None, :], mask=valid[:, None], other=0.0)
        b = tl.load(wd + col[:, None] * HIDDEN + kk[None, :])
        acc += tl.dot(a, tl.trans(b))

    src = rows[:, None] * MAX_W + local[None, :]
    gg = tl.load(g_in + src).to(tl.float32)
    uu = tl.load(u_in + src).to(tl.float32)
    sig = tl.sigmoid(gg)
    silu = gg * sig
    # Keeping the row sum 2D keeps the reduction in the accumulator's layout;
    # a 1D result made Triton convert the whole fp32 tile first and spill.
    tl.store(dw_part + rows[:, None] * NUM_NT + nt,
             tl.sum(acc * silu * uu, axis=1, keep_dims=True))
    dh = acc * tl.load(row_w + rows[:, None])
    dg = dh * uu * (sig * (1.0 + gg * (1.0 - sig)))
    du = dh * silu
    tl.store(dg_out + src, dg.to(dg_out.dtype.element_ty))
    tl.store(du_out + src, du.to(du_out.dtype.element_ty))
    if FUSE_H:
        # h = w_r * silu(g) * u, in the same operation order as _proj_rows'
        # epilogue. Read g and u again rather than sharing silu and uu above:
        # sharing took _dgu to 1.21-1.61 ms at the primary shape, depending on
        # where h went in the epilogue, against 1.18 ms for this pass.
        hh = tl.load(g_in + src).to(tl.float32)
        hh = hh * tl.sigmoid(hh)
        hh = hh * tl.load(u_in + src).to(tl.float32)
        hh = hh * tl.load(row_w + rows[:, None])
        tl.store(h_out + src, hh.to(h_out.dtype.element_ty))


# dw_r = the sum, in tile order, of the partials _dgu wrote for route r's row.
# Tiles past the expert's width never run and keep their zero initialization,
# so the sum needs no width mask. Dropped routes (row -1) get exact zeros.
@triton.jit
def _route_dw(
    dw,
    dw_part,
    route_rows,
    num_routes,  # runtime, not constexpr -- see fused_moe._plan_tiles
    NUM_NT: tl.constexpr,
    BLOCK_NT: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
    m = r < num_routes
    row = tl.load(route_rows + r, mask=m, other=-1)
    nt = tl.arange(0, BLOCK_NT)
    part = tl.load(dw_part + row[:, None] * NUM_NT + nt[None, :],
                   mask=(row >= 0)[:, None] & (nt[None, :] < NUM_NT),
                   other=0.0)
    tl.store(dw + r, tl.sum(part, axis=1), mask=m)


# dW[c, h] = sum over the expert's rows of a[row, c] * dout[tok(row), h] where a
# lives in padded-row layout (h for dWd). The grid tiles the whole weight, so
# the K loop runs over that block's expert's rows only, in order -- deterministic,
# and experts that received no tokens store exact zeros. Rows past the route
# count are exact zeros, so the loop stops at the count rounded up to the K tile.
# That count is a whole number of K tiles inside the expert's padded rows, so a
# lane needs a mask only where it gathers a token. The operand loads in MMA
# orientation: a masked load followed by tl.trans stays in registers on blocking
# LDGs instead of using cp.async.
@triton.jit
def _wgrad_rows(
    a_rows,
    dout,
    dw,
    route_tokens,
    wb_expert,
    expert_col_off,
    expert_row_start,
    counts,
    HIDDEN: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    hb = tl.program_id(0)
    wb = tl.program_id(1)
    e = tl.load(wb_expert + wb)
    row0 = tl.load(expert_row_start + e)
    n = tl.load(counts + e)
    n_k = ((n + BLOCK_K - 1) // BLOCK_K) * BLOCK_K

    gcol = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    lcol = gcol - ((tl.load(expert_col_off + e) >> 3) << 3)
    hcol = hb * BLOCK_H + tl.arange(0, BLOCK_H)

    acc = tl.zeros((BLOCK_W, BLOCK_H), dtype=tl.float32)
    for k in range(0, n_k, BLOCK_K):
        rows = row0 + k + tl.arange(0, BLOCK_K)
        tok = tl.load(route_tokens + rows)
        a = tl.load(a_rows + rows[None, :] * MAX_W + lcol[:, None])
        b = tl.load(dout + tl.maximum(tok, 0)[:, None] * HIDDEN + hcol[None, :],
                    mask=(tok >= 0)[:, None], other=0.0)
        acc += tl.dot(a, b)

    tl.store(dw + gcol[:, None] * HIDDEN + hcol[None, :],
             acc.to(dw.dtype.element_ty))


# dW[h, c] = sum over rows of x[token(row), h] * d[row, c] for d in {dg, du}.
# Padding rows gather x as zero, so they contribute exact zeros without any
# extra masking of d. Loop bound and load orientation as in _wgrad_rows.
@triton.jit
def _wgrad_gather(
    x,
    d_rows,
    dw,
    route_tokens,
    wb_expert,
    expert_col_off,
    expert_row_start,
    counts,
    HIDDEN: tl.constexpr,
    TOTAL_W: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    hb = tl.program_id(0)
    wb = tl.program_id(1)
    e = tl.load(wb_expert + wb)
    row0 = tl.load(expert_row_start + e)
    n = tl.load(counts + e)
    n_k = ((n + BLOCK_K - 1) // BLOCK_K) * BLOCK_K

    gcol = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    lcol = gcol - ((tl.load(expert_col_off + e) >> 3) << 3)
    hcol = hb * BLOCK_H + tl.arange(0, BLOCK_H)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for k in range(0, n_k, BLOCK_K):
        rows = row0 + k + tl.arange(0, BLOCK_K)
        tok = tl.load(route_tokens + rows)
        a = tl.load(x + tl.maximum(tok, 0)[None, :] * HIDDEN + hcol[:, None],
                    mask=(tok >= 0)[None, :], other=0.0)
        b = tl.load(d_rows + rows[:, None] * MAX_W + lcol[None, :])
        acc += tl.dot(a, b)

    tl.store(dw + hcol[:, None] * TOTAL_W + gcol[None, :],
             acc.to(dw.dtype.element_ty))


# Single-launch form of the two input-gradient GEMMs. It retains one fp32
# accumulator and streams gate then up operands through it, avoiding an
# intermediate bf16 dx write/read.
@triton.jit
def _dx_rows_fused(
    dg,
    du,
    wg,
    wu,
    dx,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
    TOTAL_W: tl.constexpr,
    MAX_W: tl.constexpr,
    WIDTH_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    nt = tl.program_id(0)
    mt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return

    width = tl.load(expert_nblocks + e) * WIDTH_BLOCK
    coff = (tl.load(expert_col_off + e) >> 3) << 3
    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    hcol = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    base = rows[:, None] * MAX_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, width, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        wsrc = hcol[:, None] * TOTAL_W + (coff + kk)[None, :]
        acc += tl.dot(
            tl.load(dg + base + kk[None, :]),
            tl.trans(tl.load(wg + wsrc)),
        )
        acc += tl.dot(
            tl.load(du + base + kk[None, :]),
            tl.trans(tl.load(wu + wsrc)),
        )

    dst = dx + rows[:, None] * HIDDEN + hcol[None, :]
    tl.store(dst, acc.to(dx.dtype.element_ty))


def _route_score_bits(num_experts):
    """Score bits in a packed int32 route key, as gpt_model.MoEMLP sizes them."""
    return min(20, 31 - max((num_experts - 1).bit_length(), 1))


def _route(expert_ids, top_weights, plan, top_k, m_tiles, row_bound,
           token_rounding='ceil', route_scores=None, stats=None):
    """Counting sort of the routes into the padded row-tile layout.

    Returns (tile_expert, route_tokens, route_rows, row_w): each row tile's
    expert (-1 if unclaimed), each padded row's token (-1 for padding), each
    route's padded row (-1 if dropped), and each padded row's routing weight
    (0 for padding). token_rounding='ceil' keeps every route, in route order
    within its expert. 'nearest' keeps each expert's count rounded half up to
    whole row tiles, but at least one tile's worth, highest route_scores first
    (top_weights when route_scores is None). Either layout is a pure function of the inputs. Leaves this
    call's kept counts in plan.counts and row starts in plan.expert_row_start.
    A float32 (2,) `stats` receives the dropped-route and padding-row
    fractions (see _route_stats).
    """
    num_routes = expert_ids.numel()
    dev = expert_ids.device
    tile_expert = torch.empty(m_tiles, dtype=torch.int32, device=dev)
    route_tokens = torch.empty(row_bound, dtype=torch.int32, device=dev)
    route_rows = torch.empty(num_routes, dtype=torch.int32, device=dev)

    route_blocks = triton.cdiv(num_routes, _ROUTE_BLOCK)
    plan.counts.zero_()
    if token_rounding == 'ceil':
        block_hist = torch.empty((route_blocks + 1, plan.count_bins),
                                 dtype=torch.int32, device=dev)
        _count_blocks[(route_blocks,)](
            expert_ids,
            plan.counts,
            block_hist,
            num_routes,
            NUM_EXPERTS=plan.num_experts,
            BINS=plan.count_bins,
            BLOCK=_ROUTE_BLOCK,
            num_warps=4,
        )
        routed_counts = plan.counts
    else:
        score_bits = _route_score_bits(plan.num_experts)
        route_keys = torch.empty_like(expert_ids)
        _count_route_keys[(route_blocks,)](
            expert_ids,
            top_weights if route_scores is None else route_scores,
            plan.counts,
            route_keys,
            num_routes,
            NUM_EXPERTS=plan.num_experts,
            BINS=plan.count_bins,
            SCORE_BITS=score_bits,
            SCORE_SCALE=(1 << score_bits) - 1,
            BLOCK=_ROUTE_BLOCK,
            num_warps=4,
        )
        sorted_keys, order = torch.sort(route_keys, stable=True)
        sorted_start = torch.empty_like(plan.counts)
        routed_counts = torch.empty_like(plan.counts)
        _round_counts[(1,)](
            plan.counts,
            sorted_start,
            routed_counts,
            NUM_EXPERTS=plan.num_experts,
            BLOCK_E=plan.block_e,
            BLOCK_M=_ROW_BLOCK,
            num_warps=4,
        )
    if token_rounding == 'ceil':
        _plan_tiles_scan[(1,)](
            plan.counts,
            plan.expert_row_start,
            plan.cursor,
            tile_expert,
            route_tokens,
            block_hist,
            m_tiles,
            route_blocks,
            NUM_EXPERTS=plan.num_experts,
            BLOCK_E=plan.block_e,
            BLOCK_M=_ROW_BLOCK,
            FILL=1024,
            BINS=plan.count_bins,
            num_warps=4,
        )
    else:
        _plan_tiles[(1,)](
            plan.counts,
            plan.expert_row_start,
            plan.cursor,
            tile_expert,
            route_tokens,
            m_tiles,
            NUM_EXPERTS=plan.num_experts,
            BLOCK_E=plan.block_e,
            BLOCK_M=_ROW_BLOCK,
            FILL=1024,
            num_warps=4,
        )
    row_w = torch.zeros(row_bound, dtype=torch.float32, device=dev)
    if token_rounding == 'ceil':
        _scatter_routes_stable[(route_blocks,)](
            expert_ids,
            top_weights,
            block_hist,
            route_tokens,
            route_rows,
            row_w,
            plan.expert_row_start,
            num_routes,
            NUM_EXPERTS=plan.num_experts,
            BINS=plan.count_bins,
            TOP_K=top_k,
            BLOCK=_ROUTE_BLOCK,
            num_warps=4,
        )
    else:
        _scatter_routes_ranked[(triton.cdiv(num_routes, 256),)](
            sorted_keys,
            order,
            top_weights,
            sorted_start,
            plan.counts,
            route_tokens,
            route_rows,
            row_w,
            plan.expert_row_start,
            num_routes,
            SCORE_BITS=score_bits,
            TOP_K=top_k,
            BLOCK_X=256,
            num_warps=4,
        )
    if stats is not None:
        _route_stats[(1,)](
            routed_counts,
            plan.counts,
            stats,
            num_routes,
            NUM_EXPERTS=plan.num_experts,
            BLOCK_E=plan.block_e,
            BLOCK_M=_ROW_BLOCK,
            num_warps=4,
        )
    return tile_expert, route_tokens, route_rows, row_w


def _wb_expert_table(plan, block_w):
    """Column-block -> expert id, cached per (plan, block size)."""
    cache = getattr(plan, '_wb_expert', None)
    if cache is None:
        cache = plan._wb_expert = {}
    table = cache.get(block_w)
    if table is None:
        table = torch.repeat_interleave(
            torch.arange(plan.num_experts, dtype=torch.int32),
            torch.tensor([w // block_w for w in plan.widths]),
        ).to(plan.expert_col_off.device)
        cache[block_w] = table
    return table


class _GroupedMoE(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        top_weights,
        expert_ids,
        w_gate,
        w_up,
        w_down,
        plan,
        top_k,
        down_proj_backend,
        recompute_activation,
        token_rounding,
        route_scores,
        return_route_stats,
    ):
        # Every kernel addresses these with computed offsets.
        x, top_weights, expert_ids = (x.contiguous(), top_weights.contiguous(),
                                      expert_ids.contiguous())
        w_gate, w_up, w_down = (w_gate.contiguous(), w_up.contiguous(),
                                w_down.contiguous())
        if route_scores is not None:
            route_scores = route_scores.contiguous()
        num_tokens, hidden = x.shape
        num_routes = num_tokens * top_k
        if hidden != plan.hidden_size:
            raise ValueError(f'expected hidden {plan.hidden_size}, got {hidden}')
        # The hidden tiles in _CFG must divide it, or part of the output and the
        # weight gradients is never written.
        if hidden != 128 and hidden % 256:
            raise ValueError(f'hidden must be 128 or a multiple of 256, got {hidden}')
        if expert_ids.numel() != num_routes:
            raise ValueError(
                f'expected {num_routes} expert ids, got {expert_ids.numel()}',
            )
        if down_proj_backend not in ('triton', 'cute'):
            raise ValueError(
                "down_proj_backend must be 'triton' or 'cute', got "
                f'{down_proj_backend!r}',
            )
        if not isinstance(recompute_activation, bool):
            raise TypeError('recompute_activation must be a bool')
        if token_rounding not in ('ceil', 'nearest'):
            raise ValueError(
                "token_rounding must be 'ceil' or 'nearest', got "
                f'{token_rounding!r}',
            )
        if route_scores is not None and route_scores.numel() != num_routes:
            raise ValueError(
                f'expected {num_routes} route scores, got {route_scores.numel()}',
            )

        bn = plan.block_n
        bm = _ROW_BLOCK
        m_tiles, row_bound = plan.bounds(num_tokens, top_k, bm)
        dev = x.device

        stats = (torch.empty(2, dtype=torch.float32, device=dev)
                 if return_route_stats else None)
        tile_expert, route_tokens, route_rows, row_w = _route(
            expert_ids, top_weights, plan, top_k, m_tiles, row_bound,
            token_rounding, route_scores, stats)
        # plan.counts / plan.expert_row_start are shared scratch that the next
        # forward overwrites; backward needs this call's values.
        counts = plan.counts.clone()
        row_start = plan.expert_row_start.clone()

        g = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
        u = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
        h = torch.empty_like(g)
        # 16 byte cp.async needs 16 byte aligned x rows and weights, which only
        # hand-built tensors can miss. The Triton kernels take that case and
        # other width blocks.
        if (down_proj_backend == 'cute' and bn == 128
                and all(t.data_ptr() % 16 == 0 for t in (x, w_gate, w_up))):
            from megablocks.backend.cute_down_proj import cute_gate_up_rows

            cute_gate_up_rows(
                x,
                w_gate,
                w_up,
                g,
                u,
                h,
                row_w,
                route_tokens,
                tile_expert,
                plan.expert_col_off,
                plan.expert_nblocks,
                bn,
            )
        else:
            for w, out, fuse_silu in (
                (w_gate, g, False),
                (w_up, u, True),
            ):
                pm, pn, pk, pw, ps = _CFG[
                    'proj_fused' if fuse_silu else 'proj'
                ]
                proj_grid = ((plan.max_width + pn - 1) // pn, m_tiles)
                _proj_rows[proj_grid](
                    x,
                    w,
                    out,
                    g,
                    h,
                    row_w,
                    route_tokens,
                    tile_expert,
                    plan.expert_col_off,
                    plan.expert_nblocks,
                    HIDDEN=hidden,
                    TOTAL_W=plan.total_width,
                    MAX_W=plan.max_width,
                    WIDTH_BLOCK=bn,
                    BLOCK_M=pm,
                    BLOCK_N=pn,
                    BLOCK_K=pk,
                    FUSE_SILU=fuse_silu,
                    num_warps=pw,
                    num_stages=ps,
                )

        dm, dn, dk, dw_, ds = _CFG['down']
        dn = min(dn, hidden)
        y = torch.empty((row_bound, hidden), dtype=x.dtype, device=dev)
        if down_proj_backend == 'cute':
            # Lazy import keeps CUTLASS DSL entirely optional for the default
            # Triton path (and avoids its import cost during normal startup).
            from megablocks.backend.cute_down_proj import cute_down_proj

            cute_down_proj(
                h,
                w_down,
                y,
                tile_expert,
                plan.expert_col_off,
                plan.expert_nblocks,
                bn,
            )
        elif down_proj_backend == 'triton':
            _down_proj[(m_tiles, hidden // dn)](
                h,
                w_down,
                y,
                tile_expert,
                plan.expert_col_off,
                plan.expert_nblocks,
                HIDDEN=hidden,
                MAX_W=plan.max_width,
                WIDTH_BLOCK=bn,
                BLOCK_M=dm,
                BLOCK_N=dn,
                BLOCK_K=dk,
                num_warps=dw_,
                num_stages=ds,
            )
        saved_h = None if recompute_activation else h
        if recompute_activation:
            del h

        out = torch.empty((num_tokens, hidden), dtype=x.dtype, device=dev)
        kernels._scatter_reduce[(num_tokens,)](
            out,
            y,
            route_rows,
            None,
            NUM_COLUMNS=hidden,
            TOP_K=top_k,
            SCALE=False,
        )

        saved = (x, top_weights, w_gate, w_up, w_down,
                 route_tokens, route_rows, tile_expert,
                 counts, row_start, g, u, row_w)
        if saved_h is not None:
            saved += (saved_h,)
        ctx.save_for_backward(*saved)
        ctx.plan = plan
        ctx.top_k = top_k
        ctx.launch = (m_tiles, row_bound)
        ctx.down_proj_backend = down_proj_backend
        ctx.recompute_activation = recompute_activation
        if stats is None:
            return out
        ctx.mark_non_differentiable(stats)
        return out, stats

    @staticmethod
    def backward(ctx, dout, *_):
        (x, top_weights, w_gate, w_up, w_down,
         route_tokens, route_rows, tile_expert,
         counts, row_start, g, u, row_w) = ctx.saved_tensors[:13]
        plan, top_k = ctx.plan, ctx.top_k
        m_tiles, row_bound = ctx.launch
        num_tokens, hidden = x.shape
        num_routes = num_tokens * top_k
        bn = plan.block_n
        dev = x.device
        dout = dout.contiguous()

        gm, gn, gk, gw, gs = _CFG['dgu']
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        dgu_nt = (plan.max_width + gn - 1) // gn
        dw_part = torch.zeros((row_bound, dgu_nt), dtype=torch.float32,
                              device=dev)
        # Transient when recomputed. Only the weight gradient reads h, and only
        # inside each expert's rows and width, which is what _dgu tiles over.
        h = (
            torch.empty_like(g)
            if ctx.recompute_activation
            else ctx.saved_tensors[13]
        )
        _dgu[(dgu_nt, m_tiles)](
            dout,
            w_down,
            g,
            u,
            row_w,
            dg,
            du,
            dw_part,
            h,
            route_tokens,
            tile_expert,
            plan.expert_col_off,
            plan.expert_nblocks,
            HIDDEN=hidden,
            MAX_W=plan.max_width,
            NUM_NT=dgu_nt,
            WIDTH_BLOCK=bn,
            BLOCK_M=gm,
            BLOCK_N=gn,
            BLOCK_K=gk,
            FUSE_H=ctx.recompute_activation,
            num_warps=gw,
            num_stages=gs,
        )
        dw_routes = torch.empty(num_routes, dtype=torch.float32, device=dev)
        _route_dw[(triton.cdiv(num_routes, 256),)](
            dw_routes,
            dw_part,
            route_rows,
            num_routes,
            NUM_NT=dgu_nt,
            BLOCK_NT=triton.next_power_of_2(dgu_nt),
            BLOCK_X=256,
            num_warps=4,
        )

        ww, wh, wk, wwar, wst = _CFG['wgrad_down']
        dwd = torch.empty_like(w_down)
        if ctx.down_proj_backend == 'cute':
            from megablocks.backend.cute_down_proj import cute_wgrad_rows

            cute_wgrad_rows(
                h,
                dout,
                route_tokens,
                dwd,
                _wb_expert_table(plan, 128),
                plan.expert_col_off,
                counts,
                row_start,
            )
        else:
            _wgrad_rows[(hidden // wh, plan.total_width // ww)](
                h,
                dout,
                dwd,
                route_tokens,
                _wb_expert_table(plan, ww),
                plan.expert_col_off,
                row_start,
                counts,
                HIDDEN=hidden,
                MAX_W=plan.max_width,
                BLOCK_W=ww,
                BLOCK_H=wh,
                BLOCK_K=wk,
                num_warps=wwar,
                num_stages=wst,
            )
        del h

        ww, wh, wk, wwar, wst = _CFG['wgrad_gateup']
        wb_expert = _wb_expert_table(plan, ww)
        dwg = torch.empty_like(w_gate)
        dwu = torch.empty_like(w_up)
        for d_rows, dw in ((dg, dwg), (du, dwu)):
            _wgrad_gather[(hidden // wh, plan.total_width // ww)](
                x,
                d_rows,
                dw,
                route_tokens,
                wb_expert,
                plan.expert_col_off,
                row_start,
                counts,
                HIDDEN=hidden,
                TOTAL_W=plan.total_width,
                MAX_W=plan.max_width,
                BLOCK_W=ww,
                BLOCK_H=wh,
                BLOCK_K=wk,
                num_warps=wwar,
                num_stages=wst,
            )

        dx_rows = torch.empty((row_bound, hidden), dtype=x.dtype, device=dev)
        if ctx.down_proj_backend == 'cute':
            from megablocks.backend.cute_down_proj import cute_dx_proj_fused

            cute_dx_proj_fused(
                dg,
                du,
                w_gate,
                w_up,
                dx_rows,
                tile_expert,
                plan.expert_col_off,
                plan.expert_nblocks,
                bn,
            )
        else:
            xm, xn, xk, xw, xs = _CFG['dx_fused']
            xn = min(xn, hidden)
            _dx_rows_fused[(hidden // xn, m_tiles)](
                dg,
                du,
                w_gate,
                w_up,
                dx_rows,
                tile_expert,
                plan.expert_col_off,
                plan.expert_nblocks,
                HIDDEN=hidden,
                TOTAL_W=plan.total_width,
                MAX_W=plan.max_width,
                WIDTH_BLOCK=bn,
                BLOCK_M=xm,
                BLOCK_N=xn,
                BLOCK_K=xk,
                num_warps=xw,
                num_stages=xs,
            )
        dx = torch.empty_like(x)
        kernels._scatter_reduce[(num_tokens,)](
            dx,
            dx_rows,
            route_rows,
            None,
            NUM_COLUMNS=hidden,
            TOP_K=top_k,
            SCALE=False,
        )

        return (dx, dw_routes.to(top_weights.dtype), None,
                dwg, dwu, dwd, None, None, None, None, None, None, None)


def grouped_moe(
    x,
    top_weights,
    expert_ids,
    plan,
    top_k,
    w_gate,
    w_up,
    w_down,
    down_proj_backend='triton',
    recompute_activation=True,
    token_rounding='ceil',
    route_scores=None,
    return_route_stats=False,
):
    """Differentiable gated MoE over packed variable-width experts.

    Same calling convention as fused_moe_forward: `expert_ids` and
    `top_weights` are the flattened (tokens * top_k) routed expert ids (int32)
    and routing weights, token-major, straight from topk. Returns
    (tokens, hidden), differentiable w.r.t. x, top_weights and all three
    weights. ``down_proj_backend='cute'`` opts into the experimental Ampere
    CuTe DSL gate, up and down projections, input gradients, and
    down-projection weight gradient. The remaining kernels use Triton. Set
    ``recompute_activation=False`` to retain the forward SwiGLU activation and
    trade activation memory for a faster backward pass.

    ``token_rounding='nearest'`` rounds each expert's route count half up to
    whole 128-row tiles, like gpt_model's ``token_rounding='nearest'``: an
    expert that rounds down drops its lowest-scoring routes, which then
    contribute nothing and receive no gradient. Unlike gpt_model, an expert with
    routes never rounds down to zero, so an expert under 64 routes keeps them
    all. The surviving weights are not renormalized. ``route_scores`` (flattened like ``top_weights``, in [0, 1],
    e.g. the router probabilities before top-k renormalization) ranks the
    routes; it defaults to ``top_weights``. The default ``'ceil'`` keeps every
    route and pads each expert to whole tiles instead.

    ``return_route_stats=True`` returns ``(output, stats)``, where ``stats``
    maps ``'dropped_route_fraction'`` and ``'padding_route_fraction'`` --
    gpt_model.MoEMLP's names -- to 0-dim float32 tensors left on the device,
    so logging them does not synchronize.
    """
    result = _GroupedMoE.apply(x, top_weights, expert_ids,
                               w_gate, w_up, w_down, plan, top_k,
                               down_proj_backend, recompute_activation,
                               token_rounding, route_scores,
                               return_route_stats)
    if not return_route_stats:
        return result
    out, stats = result
    return out, {
        'dropped_route_fraction': stats[0],
        'padding_route_fraction': stats[1],
    }
