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

Every GEMM kernel here holds a SINGLE fp32 accumulator. That is a deliberate
trade measured on a 3090: fusing gate+up (or dWg+dWu) into one kernel halves
the gathered-x traffic but caps the tile at 128x64 before registers spill,
and the win from running every GEMM at its best single-tile config (128x128
class) is larger than the cost of gathering x twice -- weight re-reads, not x
re-reads, dominate the traffic at training shapes.

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

  `_route_bwd`      dy_rows[row] = dout[token]
  `_dgu`            da = dy_rows @ Wd^T; per-tile partials of
                    dw_r = <da, silu(g) * u>; dh = w_r * da, then SwiGLU'
                    in-register -> dg, du
  `_route_dw`       dw_r = sum of its row's partials, in tile order
  `_wgrad_rows`     Triton or CuTe rows^T @ cols for (h, dy_rows) -> dWd
  `_wgrad_gather`   dW = gather(x)^T @ dg -> dWg, and with du -> dWu
  `_dx_rows_fused`  dx_rows = dg @ Wg^T + du @ Wu^T, using one fp32
                    accumulator in either Triton or CuTe
  `_scatter_reduce` (unweighted) top-k sum of dx_rows into dx

Padding is self-consistent everywhere: a padding row inside a claimed tile has
x gathered as zero in forward (so g = u = 0) and dy_rows zero-initialized in
backward, so every wgrad contribution from padding is an exact zero. Rows past
the last claimed tile are never inside any expert's row range and are never
read. Weight gradients cover every weight block unconditionally, so untouched
experts get exact-zero grads rather than stale memory.
"""

import torch
import triton
import triton.language as tl

from megablocks.backend import kernels
from megablocks.backend.fused_moe import _down_proj, _plan_tiles

# Per-kernel launch configs, swept on a 3090 at the healing shape (keep50
# widths, 4096 tokens, top_k 8). Keys: (BLOCK_M, BLOCK_N, BLOCK_K, warps,
# stages). BLOCK_N on the width axis must be a multiple of 128 (the
# width-block granularity); on the hidden axis it must divide the hidden size.
_CFG = {
    'proj': (128, 128, 16, 4, 2),        # gate projection
    'proj_fused': (128, 128, 32, 4, 2),  # up projection + SwiGLU epilogue
    'down': (128, 256, 64, 8, 3),        # _down_proj on h
    'dgu': (128, 128, 64, 8, 3),         # _dgu
    'dx_fused': (128, 256, 32, 8, 2),    # gate + up input-gradient streams
    'wgrad_down': (64, 128, 32, 4, 2),   # _wgrad_rows: (BLOCK_W, BLOCK_H, K)
    'wgrad_gateup': (128, 64, 32, 4, 2),  # _wgrad_gather: (BLOCK_W, BLOCK_H, K)
}
# The row-tile layout is planned once with this block; every row-tile kernel
# uses the same BLOCK_M so tiles stay aligned to the planned buffers.
_ROW_BLOCK = 128

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


# The permute step of the counting sort, made stable: a route's rank within its
# expert is the number of earlier routes to that expert, so the padded layout
# -- and with it the summation order of every weight-gradient K loop -- is a
# pure function of expert_ids. fused_moe._scatter_routes claims rows with an
# atomic cursor, which is fine for inference, where rows are independent, but
# made dW differ bitwise between identical training steps. Earlier blocks
# contribute through their histograms; within the block, sorting on
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
    PREFIX_BLOCKS: tl.constexpr,
):
    b = tl.program_id(0)
    bins = tl.arange(0, BINS)
    earlier = tl.zeros((BINS,), dtype=tl.int32)
    for b0 in range(0, b, PREFIX_BLOCKS):
        prev = b0 + tl.arange(0, PREFIX_BLOCKS)
        earlier += tl.sum(
            tl.load(block_hist + prev[:, None] * BINS + bins[None, :],
                    mask=(prev < b)[:, None], other=0),
            axis=0,
        )
    hist = tl.load(block_hist + b * BINS + bins)
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
# rounds down. This is gpt_model's token_rounding='nearest' (and SonicMoE's
# token rounding). The keys match kernels._pack_route_keys -- expert-major,
# descending quantized score -- so a stable sort ranks each expert's routes
# with ties kept in route order. Also accumulates the per-expert totals.
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
    descending = ((1.0 - score) * SCORE_SCALE).to(tl.int32)
    descending = tl.maximum(0, tl.minimum(descending, SCORE_SCALE))
    tl.store(route_keys + off, (e << SCORE_BITS) | descending, mask=m)


# One program: where each expert's routes start in the sorted keys, and how
# many of them it keeps -- min(count, count rounded half up to BLOCK_M), as in
# gpt_model's _block_tokens_per_expert. _plan_tiles then plans the kept counts.
@triton.jit
def _round_counts(
    counts,
    sorted_start,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    e = tl.arange(0, BLOCK_E)
    live = e < NUM_EXPERTS
    n = tl.load(counts + e, mask=live, other=0)
    tl.store(sorted_start + e, tl.cumsum(n, 0) - n, mask=live)
    rounded = (n + BLOCK_M // 2) // BLOCK_M * BLOCK_M
    tl.store(counts + e, tl.minimum(n, rounded), mask=live)


# Rows for the kept routes, in descending-score order within each expert. A
# dropped route gets row -1, which _scatter_reduce, _route_bwd, and _route_dw
# skip, so it contributes nothing and receives no gradient.
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
    NBLOCK_N: tl.constexpr,  # width blocks (128) per BLOCK_N
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FUSE_SILU: tl.constexpr,
):
    mt = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return
    if nt * NBLOCK_N >= tl.load(expert_nblocks + e):
        return

    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(route_tokens + rows)
    valid = tok >= 0
    xp = x + tl.maximum(tok, 0)[:, None] * HIDDEN

    local = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    col = tl.load(expert_col_off + e) + local

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
        # _silu_mul repeats this operation order for recompute_activation.
        gg = tl.load(g_in + dst).to(tl.float32)
        uu = out_value.to(tl.float32)
        tl.store(h_out + dst,
                 (gg * tl.sigmoid(gg) * uu * ww).to(h_out.dtype.element_ty))


# One program per route: dy_rows[row] = dout[token]. h already carries the
# routing weight, so the down projection's output gradient is the token's own,
# unscaled. Every kept route owns exactly one padded row, so there are no write
# conflicts; padding rows are never a route's destination and stay at their
# zero initialization. A route dropped by token rounding has row -1 and no
# gradient.
@triton.jit
def _route_bwd(
    dout,
    dy_rows,
    route_rows,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    r = tl.program_id(0)
    row = tl.load(route_rows + r)
    if row < 0:
        return
    src = dout + (r // TOP_K) * NUM_COLUMNS
    dst = dy_rows + row * NUM_COLUMNS
    for col_base in range(0, NUM_COLUMNS, BLOCK_X):
        cols = col_base + tl.arange(0, BLOCK_X)
        mask = cols < NUM_COLUMNS
        tl.store(dst + cols, tl.load(src + cols, mask=mask), mask=mask)


# da = dy_rows @ Wd^T for the expert's own width: the gradient with respect to
# the weighted activation w_r * h. Its row dot with the unweighted
# h = silu(g) * u is this tile's share of dw_r, written to dw_part[row, nt]
# for _route_dw. dh = w_r * da then feeds the SwiGLU backward on the fp32
# tile: dg = dh * u * silu'(g), du = dh * silu(g), with
# silu'(g) = sig(g) * (1 + g * (1 - sig(g))). Wd rows are loaded contiguously
# as (N, K) and transposed in-register.
@triton.jit
def _dgu(
    dy_rows,
    wd,
    g_in,
    u_in,
    row_w,
    dg_out,
    du_out,
    dw_part,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
    MAX_W: tl.constexpr,
    NUM_NT: tl.constexpr,  # dw_part columns: the grid's width-tile count
    NBLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    mt = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return
    if nt * NBLOCK_N >= tl.load(expert_nblocks + e):
        return

    rows = mt * BLOCK_M + tl.arange(0, BLOCK_M)
    local = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    col = tl.load(expert_col_off + e) + local

    dyp = dy_rows + rows[:, None] * HIDDEN
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, HIDDEN, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(dyp + kk[None, :])
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


# dW[c, h] = sum over the expert's padded rows of a[row, c] * b[row, h] where
# a lives in padded-row layout (h for dWd). The grid tiles the whole weight,
# so the K loop runs over that block's expert's rows only, in order --
# deterministic, and experts that received no tokens store exact zeros.
@triton.jit
def _wgrad_rows(
    a_rows,
    b_rows,
    dw,
    wb_expert,
    expert_col_off,
    expert_row_start,
    counts,
    row_block,  # runtime: rows pad to a multiple of the forward row tile
    HIDDEN: tl.constexpr,
    MAX_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    wb = tl.program_id(0)
    hb = tl.program_id(1)
    e = tl.load(wb_expert + wb)
    row0 = tl.load(expert_row_start + e)
    n = tl.load(counts + e)
    n_pad = ((n + row_block - 1) // row_block) * row_block

    gcol = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    lcol = gcol - tl.load(expert_col_off + e)
    hcol = hb * BLOCK_H + tl.arange(0, BLOCK_H)

    acc = tl.zeros((BLOCK_W, BLOCK_H), dtype=tl.float32)
    for k in range(0, n_pad, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        live = kk < n_pad
        rows = row0 + kk
        a = tl.load(a_rows + rows[:, None] * MAX_W + lcol[None, :],
                    mask=live[:, None], other=0.0)
        b = tl.load(b_rows + rows[:, None] * HIDDEN + hcol[None, :],
                    mask=live[:, None], other=0.0)
        acc += tl.dot(tl.trans(a), b)

    tl.store(dw + gcol[:, None] * HIDDEN + hcol[None, :],
             acc.to(dw.dtype.element_ty))


# dW[h, c] = sum over rows of x[token(row), h] * d[row, c] for d in {dg, du}.
# Padding rows gather x as zero, so they contribute exact zeros without any
# extra masking of d.
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
    row_block,  # runtime -- see _wgrad_rows
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
    n_pad = ((n + row_block - 1) // row_block) * row_block

    gcol = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    lcol = gcol - tl.load(expert_col_off + e)
    hcol = hb * BLOCK_H + tl.arange(0, BLOCK_H)

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for k in range(0, n_pad, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        live = kk < n_pad
        rows = row0 + kk
        tok = tl.load(route_tokens + rows, mask=live, other=-1)
        valid = live & (tok >= 0)
        a = tl.load(x + tl.maximum(tok, 0)[:, None] * HIDDEN + hcol[None, :],
                    mask=valid[:, None], other=0.0)
        b = tl.load(d_rows + rows[:, None] * MAX_W + lcol[None, :],
                    mask=live[:, None], other=0.0)
        acc += tl.dot(tl.trans(a), b)

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
    mt = tl.program_id(0)
    nt = tl.program_id(1)
    e = tl.load(tile_expert + mt)
    if e < 0:
        return

    width = tl.load(expert_nblocks + e) * WIDTH_BLOCK
    coff = tl.load(expert_col_off + e)
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


# h = w_r * silu(g) * u in one pass, in the same operation order as
# _proj_rows' epilogue. F.silu(g) * u is two eager kernels and an extra
# intermediate write; this is one read of g/u and one write of h.
@triton.jit
def _silu_mul(
    g_in,
    u_in,
    row_w,
    h_out,
    numel,
    NUM_COLUMNS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < numel
    gg = tl.load(g_in + off, mask=m, other=0.0).to(tl.float32)
    uu = tl.load(u_in + off, mask=m, other=0.0).to(tl.float32)
    ww = tl.load(row_w + off // NUM_COLUMNS, mask=m, other=0.0)
    tl.store(h_out + off,
             (gg * tl.sigmoid(gg) * uu * ww).to(h_out.dtype.element_ty),
             mask=m)


# Zero only the padding rows of dy_rows (route_tokens < 0): the full-buffer
# memset costs ~0.4 ms at training shapes and >85% of it is immediately
# overwritten by _route_bwd.
@triton.jit
def _zero_pad_rows(
    buf,
    route_tokens,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(route_tokens + row) >= 0:
        return
    zeros = tl.zeros((BLOCK_X,), dtype=buf.dtype.element_ty)
    for col_base in range(0, NUM_COLUMNS, BLOCK_X):
        cols = col_base + tl.arange(0, BLOCK_X)
        tl.store(buf + row * NUM_COLUMNS + cols, zeros,
                 mask=cols < NUM_COLUMNS)


def _silu_mul_launch(g, u, row_w):
    h = torch.empty_like(g)
    n = g.numel()
    _silu_mul[(triton.cdiv(n, 4096),)](g, u, row_w, h, n,
                                       NUM_COLUMNS=g.shape[1],
                                       BLOCK=4096, num_warps=8)
    return h


def _route_score_bits(num_experts):
    """Score bits in a packed int32 route key, as gpt_model.MoEMLP sizes them."""
    return min(20, 31 - max((num_experts - 1).bit_length(), 1))


def _route(expert_ids, top_weights, plan, top_k, m_tiles, row_bound,
           token_rounding='ceil', route_scores=None):
    """Counting sort of the routes into the padded row-tile layout.

    Returns (tile_expert, route_tokens, route_rows, row_w): each row tile's
    expert (-1 if unclaimed), each padded row's token (-1 for padding), each
    route's padded row (-1 if dropped), and each padded row's routing weight
    (0 for padding). token_rounding='ceil' keeps every route, in route order
    within its expert. 'nearest' keeps each expert's count rounded half up to
    whole row tiles, highest route_scores first (top_weights when route_scores
    is None). Either layout is a pure function of the inputs. Leaves this
    call's kept counts in plan.counts and row starts in plan.expert_row_start.
    """
    num_routes = expert_ids.numel()
    dev = expert_ids.device
    tile_expert = torch.empty(m_tiles, dtype=torch.int32, device=dev)
    route_tokens = torch.empty(row_bound, dtype=torch.int32, device=dev)
    route_rows = torch.empty(num_routes, dtype=torch.int32, device=dev)

    route_blocks = triton.cdiv(num_routes, _ROUTE_BLOCK)
    plan.counts.zero_()
    if token_rounding == 'ceil':
        block_hist = torch.empty((route_blocks, plan.count_bins),
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
        _round_counts[(1,)](
            plan.counts,
            sorted_start,
            NUM_EXPERTS=plan.num_experts,
            BLOCK_E=plan.block_e,
            BLOCK_M=_ROW_BLOCK,
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
            PREFIX_BLOCKS=8,
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
    ):
        num_tokens, hidden = x.shape
        num_routes = num_tokens * top_k
        if hidden != plan.hidden_size:
            raise ValueError(f'expected hidden {plan.hidden_size}, got {hidden}')
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

        tile_expert, route_tokens, route_rows, row_w = _route(
            expert_ids, top_weights, plan, top_k, m_tiles, row_bound,
            token_rounding, route_scores)
        # plan.counts / plan.expert_row_start are shared scratch that the next
        # forward overwrites; backward needs this call's values.
        counts = plan.counts.clone()
        row_start = plan.expert_row_start.clone()

        g = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
        u = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
        h = torch.empty_like(g)
        for w, out, fuse_silu in (
            (w_gate, g, False),
            (w_up, u, True),
        ):
            pm, pn, pk, pw, ps = _CFG[
                'proj_fused' if fuse_silu else 'proj'
            ]
            proj_grid = (m_tiles, (plan.max_width + pn - 1) // pn)
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
                NBLOCK_N=pn // bn,
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
        ctx.launch = (bm, m_tiles, row_bound)
        ctx.down_proj_backend = down_proj_backend
        ctx.recompute_activation = recompute_activation
        return out

    @staticmethod
    def backward(ctx, dout):
        (x, top_weights, w_gate, w_up, w_down,
         route_tokens, route_rows, tile_expert,
         counts, row_start, g, u, row_w) = ctx.saved_tensors[:13]
        plan, top_k = ctx.plan, ctx.top_k
        bm, m_tiles, row_bound = ctx.launch
        num_tokens, hidden = x.shape
        num_routes = num_tokens * top_k
        bn = plan.block_n
        dev = x.device
        dout = dout.contiguous()

        # Padding rows inside claimed tiles are read by the weight-gradient K
        # loops but never written by _route_bwd, so they must be zero; zeroing
        # just those rows beats a full-buffer memset.
        dy_rows = torch.empty((row_bound, hidden), dtype=x.dtype, device=dev)
        _zero_pad_rows[(row_bound,)](
            dy_rows,
            route_tokens,
            NUM_COLUMNS=hidden,
            BLOCK_X=1024,
            num_warps=4,
        )
        _route_bwd[(num_routes,)](
            dout,
            dy_rows,
            route_rows,
            NUM_COLUMNS=hidden,
            TOP_K=top_k,
            BLOCK_X=1024,
            num_warps=4,
        )

        gm, gn, gk, gw, gs = _CFG['dgu']
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        dgu_nt = (plan.max_width + gn - 1) // gn
        dw_part = torch.zeros((row_bound, dgu_nt), dtype=torch.float32,
                              device=dev)
        _dgu[(m_tiles, dgu_nt)](
            dy_rows,
            w_down,
            g,
            u,
            row_w,
            dg,
            du,
            dw_part,
            tile_expert,
            plan.expert_col_off,
            plan.expert_nblocks,
            HIDDEN=hidden,
            MAX_W=plan.max_width,
            NUM_NT=dgu_nt,
            NBLOCK_N=gn // bn,
            BLOCK_M=gm,
            BLOCK_N=gn,
            BLOCK_K=gk,
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

        # One elementwise pass beats recomputing w_r * silu(g) * u inside
        # every K iteration of the weight-gradient kernel. Transient.
        h = (
            _silu_mul_launch(g, u, row_w)
            if ctx.recompute_activation
            else ctx.saved_tensors[13]
        )
        ww, wh, wk, wwar, wst = _CFG['wgrad_down']
        dwd = torch.empty_like(w_down)
        if ctx.down_proj_backend == 'cute':
            from megablocks.backend.cute_down_proj import cute_wgrad_rows

            cute_wgrad_rows(
                h,
                dy_rows,
                dwd,
                _wb_expert_table(plan, 128),
                plan.expert_col_off,
                counts,
                row_start,
                bm,
            )
        else:
            _wgrad_rows[(plan.total_width // ww, hidden // wh)](
                h,
                dy_rows,
                dwd,
                _wb_expert_table(plan, ww),
                plan.expert_col_off,
                row_start,
                counts,
                bm,
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
                bm,
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
            _dx_rows_fused[(m_tiles, hidden // xn)](
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
                dwg, dwu, dwd, None, None, None, None, None, None)


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
):
    """Differentiable gated MoE over packed variable-width experts.

    Same calling convention as fused_moe_forward: `expert_ids` and
    `top_weights` are the flattened (tokens * top_k) routed expert ids (int32)
    and routing weights, token-major, straight from topk. Returns
    (tokens, hidden), differentiable w.r.t. x, top_weights and all three
    weights. ``down_proj_backend='cute'`` opts into the experimental Ampere
    CuTe DSL down projection, input gradients, and down-projection weight
    gradient. The remaining kernels use Triton. Set
    ``recompute_activation=False`` to retain the forward SwiGLU activation and
    trade activation memory for a faster backward pass.

    ``token_rounding='nearest'`` rounds each expert's route count half up to
    whole 128-row tiles, like gpt_model's ``token_rounding='nearest'``: an
    expert that rounds down drops its lowest-scoring routes, which then
    contribute nothing and receive no gradient. The surviving weights are not
    renormalized. ``route_scores`` (flattened like ``top_weights``, in [0, 1],
    e.g. the router probabilities before top-k renormalization) ranks the
    routes; it defaults to ``top_weights``. The default ``'ceil'`` keeps every
    route and pads each expert to whole tiles instead.
    """
    return _GroupedMoE.apply(x, top_weights, expert_ids,
                             w_gate, w_up, w_down, plan, top_k,
                             down_proj_backend, recompute_activation,
                             token_rounding, route_scores)
