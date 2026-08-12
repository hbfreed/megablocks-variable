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

Forward reuses fused_moe's counting-sort routing verbatim, then:

  `_gate_up_gu`     gathers x, both projections; stores the g/u pre-activations
                    (backward needs them -- the stk path stores the same two)
  `_down_from_gu`   recomputes h = silu(g)*u in registers, down-projects
  `_scatter_reduce` weighted top-k reduction into the output (from kernels.py)

Backward is five kernels, all deterministic (fp32 tile accumulators, fixed
reduction order, no atomics):

  `_route_bwd`      dy_rows[row] = w_r * dout[token]; dw_r = <y_row, dout>
  `_dgu`            dh = dy_rows @ Wd^T, then SwiGLU' in-register -> dg, du
  `_wgrad_down`     dWd = h^T @ dy_rows, h recomputed from g/u
  `_wgrad_gateup`   dWg = x^T @ dg and dWu = x^T @ du, sharing the x gather
  `_dx_rows`        dx_rows = dg @ Wg^T + du @ Wu^T
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
from megablocks.backend.fused_moe import (
    _count_experts,
    _gate_up_silu,  # noqa: F401  (re-exported for symmetry/debugging)
    _plan_tiles,
    _scatter_routes,
)


# Forward gate/up that keeps the pre-activations instead of fusing the SwiGLU:
# backward needs g and u to differentiate through silu(g) * u. Same tiling and
# gather as fused_moe._gate_up_silu.
@triton.jit
def _gate_up_gu(
    x,
    wg,
    wu,
    g_out,
    u_out,
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

    dst = rows[:, None] * MAX_W + local[None, :]
    tl.store(g_out + dst, acc_g.to(g_out.dtype.element_ty))
    tl.store(u_out + dst, acc_u.to(u_out.dtype.element_ty))


# fused_moe._down_proj with h = silu(g) * u recomputed in registers from the
# stored pre-activations, so the training forward never materializes h at all.
@triton.jit
def _down_from_gu(
    g_in,
    u_in,
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

    base = rows[:, None] * MAX_W
    wdp = wd + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, width, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        gg = tl.load(g_in + base + kk[None, :]).to(tl.float32)
        uu = tl.load(u_in + base + kk[None, :]).to(tl.float32)
        a = (gg * tl.sigmoid(gg) * uu).to(wd.dtype.element_ty)
        b = tl.load(wdp + (coff + kk)[:, None] * HIDDEN)
        acc += tl.dot(a, b)

    tl.store(y + rows[:, None] * HIDDEN + cols[None, :],
             acc.to(y.dtype.element_ty))


# One program per route. Splits dout at the reduction boundary: the row grad
# picks up the routing weight (dy_row = w_r * dout[token]) and the routing
# weight grad is the unweighted dot (dw_r = <y_row, dout[token]>). Every route
# owns exactly one padded row, so there are no write conflicts; padding rows
# are never a route's destination and stay at their zero initialization.
@triton.jit
def _route_bwd(
    dout,
    y,
    weights,
    dy_rows,
    dw,
    route_rows,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    r = tl.program_id(0)
    row = tl.load(route_rows + r)
    tok = r // TOP_K
    w = tl.load(weights + r).to(tl.float32)

    src = dout + tok * NUM_COLUMNS
    yp = y + row * NUM_COLUMNS
    dst = dy_rows + row * NUM_COLUMNS
    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    for col_base in range(0, NUM_COLUMNS, BLOCK_X):
        cols = col_base + tl.arange(0, BLOCK_X)
        mask = cols < NUM_COLUMNS
        d = tl.load(src + cols, mask=mask, other=0.0).to(tl.float32)
        yv = tl.load(yp + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst + cols, (w * d).to(dy_rows.dtype.element_ty), mask=mask)
        acc += yv * d
    tl.store(dw + r, tl.sum(acc))


# dh = dy_rows @ Wd^T for the expert's own width, then the SwiGLU backward on
# the fp32 tile: dg = dh * u * silu'(g), du = dh * silu(g), with
# silu'(g) = sig(g) * (1 + g * (1 - sig(g))). Same grid/early-exit as forward.
@triton.jit
def _dgu(
    dy_rows,
    wd,
    g_in,
    u_in,
    dg_out,
    du_out,
    tile_expert,
    expert_col_off,
    expert_nblocks,
    HIDDEN: tl.constexpr,
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
    local = nt * BLOCK_N + tl.arange(0, BLOCK_N)
    col = tl.load(expert_col_off + e) + local

    dyp = dy_rows + rows[:, None] * HIDDEN
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, HIDDEN, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        a = tl.load(dyp + kk[None, :])
        # Load Wd rows contiguously as (N, K) and transpose in-register: the
        # (K, N) layout would stride HIDDEN between adjacent lanes.
        b = tl.load(wd + col[:, None] * HIDDEN + kk[None, :])
        acc += tl.dot(a, tl.trans(b))

    src = rows[:, None] * MAX_W + local[None, :]
    gg = tl.load(g_in + src).to(tl.float32)
    uu = tl.load(u_in + src).to(tl.float32)
    sig = tl.sigmoid(gg)
    silu = gg * sig
    dg = acc * uu * (sig * (1.0 + gg * (1.0 - sig)))
    du = acc * silu
    tl.store(dg_out + src, dg.to(dg_out.dtype.element_ty))
    tl.store(du_out + src, du.to(du_out.dtype.element_ty))


# dWd[c, h] = sum over the expert's padded rows of h[row, c] * dy_rows[row, h],
# h recomputed from g/u. The grid tiles the whole weight, so the K loop runs
# over that block's expert's rows only, in order -- deterministic, and experts
# that received no tokens store exact zeros.
@triton.jit
def _wgrad_down(
    g_in,
    u_in,
    dy_rows,
    dwd,
    wb_expert,
    expert_col_off,
    expert_row_start,
    counts,
    row_block,  # runtime: forward's BLOCK_M (rows pad to a multiple of it)
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
        src = rows[:, None] * MAX_W + lcol[None, :]
        gg = tl.load(g_in + src, mask=live[:, None], other=0.0).to(tl.float32)
        uu = tl.load(u_in + src, mask=live[:, None], other=0.0).to(tl.float32)
        a = (gg * tl.sigmoid(gg) * uu).to(dy_rows.dtype.element_ty)
        b = tl.load(dy_rows + rows[:, None] * HIDDEN + hcol[None, :],
                    mask=live[:, None], other=0.0)
        acc += tl.dot(tl.trans(a), b)

    tl.store(dwd + gcol[:, None] * HIDDEN + hcol[None, :],
             acc.to(dwd.dtype.element_ty))


# dWg[h, c] = sum over rows of x[token(row), h] * dg[row, c], and dWu likewise
# with du, sharing the gathered x tile. Padding rows gather x as zero, so they
# contribute exact zeros without any extra masking of dg/du.
@triton.jit
def _wgrad_gateup(
    x,
    dg_in,
    du_in,
    dwg,
    dwu,
    route_tokens,
    wb_expert,
    expert_col_off,
    expert_row_start,
    counts,
    row_block,  # runtime -- see _wgrad_down
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

    acc_g = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for k in range(0, n_pad, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        live = kk < n_pad
        rows = row0 + kk
        tok = tl.load(route_tokens + rows, mask=live, other=-1)
        valid = live & (tok >= 0)
        # a[k, h] = x[token_k, hcol_h], zero for padding rows.
        a = tl.load(x + tl.maximum(tok, 0)[:, None] * HIDDEN + hcol[None, :],
                    mask=valid[:, None], other=0.0)
        src = rows[:, None] * MAX_W + lcol[None, :]
        bg = tl.load(dg_in + src, mask=live[:, None], other=0.0)
        bu = tl.load(du_in + src, mask=live[:, None], other=0.0)
        at = tl.trans(a)
        acc_g += tl.dot(at, bg)
        acc_u += tl.dot(at, bu)

    dst = hcol[:, None] * TOTAL_W + gcol[None, :]
    tl.store(dwg + dst, acc_g.to(dwg.dtype.element_ty))
    tl.store(dwu + dst, acc_u.to(dwu.dtype.element_ty))


# dx_rows = dg @ Wg^T + du @ Wu^T over the expert's own width; the top-k sum
# back into token space is kernels._scatter_reduce with SCALE=False.
@triton.jit
def _dx_rows(
    dg_in,
    du_in,
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
        ag = tl.load(dg_in + base + kk[None, :])
        au = tl.load(du_in + base + kk[None, :])
        # Load W rows contiguously as (H, K) and transpose in-register: the
        # (K, H) layout would stride TOTAL_W between adjacent lanes.
        wsrc = hcol[:, None] * TOTAL_W + (coff + kk)[None, :]
        acc += tl.dot(ag, tl.trans(tl.load(wg + wsrc)))
        acc += tl.dot(au, tl.trans(tl.load(wu + wsrc)))

    tl.store(dx + rows[:, None] * HIDDEN + hcol[None, :],
             acc.to(dx.dtype.element_ty))


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


# Weight-gradient tile sizes. 64x64 fp32 accumulators (two of them in
# _wgrad_gateup) stay within Ampere's register budget at num_warps=8.
# Swept on a 3090 at the served keep50 shape (4096 tokens, top_k 8): 128x64
# tiles at 4 warps were 1.24x faster overall than the 64x64/8-warp starting
# point. The two fp32 accumulators in _wgrad_gateup put 128x64 at the register
# ceiling; wider tiles spill.
_WGRAD_BLOCK_W = 128
_WGRAD_BLOCK_H = 64
_WGRAD_BLOCK_K = 32
_WGRAD_WARPS = 4
_WGRAD_STAGES = 3


class _GroupedMoE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, top_weights, expert_ids, w_gate, w_up, w_down, plan, top_k):
        num_tokens, hidden = x.shape
        num_routes = num_tokens * top_k
        if hidden != plan.hidden_size:
            raise ValueError(f'expected hidden {plan.hidden_size}, got {hidden}')
        if expert_ids.numel() != num_routes:
            raise ValueError(
                f'expected {num_routes} expert ids, got {expert_ids.numel()}',
            )

        bn = plan.block_n
        bm, num_warps, num_stages = plan.tile_config(num_tokens, top_k)
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
        # plan.counts / plan.expert_row_start are shared scratch that the next
        # forward overwrites; backward needs this call's values.
        counts = plan.counts.clone()
        row_start = plan.expert_row_start.clone()

        g = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
        u = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
        _gate_up_gu[(m_tiles, plan.max_nblocks)](
            x,
            w_gate,
            w_up,
            g,
            u,
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
        _down_from_gu[(m_tiles, hidden // bn)](
            g,
            u,
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
            SCALE=True,
        )

        ctx.save_for_backward(x, top_weights, w_gate, w_up, w_down,
                              route_tokens, route_rows, tile_expert,
                              counts, row_start, g, u, y)
        ctx.plan = plan
        ctx.top_k = top_k
        ctx.launch = (bm, num_warps, num_stages, m_tiles, row_bound)
        return out

    @staticmethod
    def backward(ctx, dout):
        (x, top_weights, w_gate, w_up, w_down,
         route_tokens, route_rows, tile_expert,
         counts, row_start, g, u, y) = ctx.saved_tensors
        plan, top_k = ctx.plan, ctx.top_k
        bm, num_warps, num_stages, m_tiles, row_bound = ctx.launch
        num_tokens, hidden = x.shape
        num_routes = num_tokens * top_k
        bn = plan.block_n
        dev = x.device
        dout = dout.contiguous()

        # Zero-initialized: padding rows inside claimed tiles are read by the
        # weight-gradient K loops but never written by _route_bwd.
        dy_rows = torch.zeros((row_bound, hidden), dtype=x.dtype, device=dev)
        dw_routes = torch.empty(num_routes, dtype=torch.float32, device=dev)
        _route_bwd[(num_routes,)](
            dout,
            y,
            top_weights,
            dy_rows,
            dw_routes,
            route_rows,
            NUM_COLUMNS=hidden,
            TOP_K=top_k,
            BLOCK_X=1024,
            num_warps=4,
        )

        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        _dgu[(m_tiles, plan.max_nblocks)](
            dy_rows,
            w_down,
            g,
            u,
            dg,
            du,
            tile_expert,
            plan.expert_col_off,
            plan.expert_nblocks,
            HIDDEN=hidden,
            MAX_W=plan.max_width,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=64,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        wb_expert = _wb_expert_table(plan, _WGRAD_BLOCK_W)
        total_wb = plan.total_width // _WGRAD_BLOCK_W

        dwd = torch.empty_like(w_down)
        _wgrad_down[(total_wb, hidden // _WGRAD_BLOCK_H)](
            g,
            u,
            dy_rows,
            dwd,
            wb_expert,
            plan.expert_col_off,
            row_start,
            counts,
            bm,
            HIDDEN=hidden,
            MAX_W=plan.max_width,
            BLOCK_W=_WGRAD_BLOCK_W,
            BLOCK_H=_WGRAD_BLOCK_H,
            BLOCK_K=_WGRAD_BLOCK_K,
            num_warps=_WGRAD_WARPS,
            num_stages=_WGRAD_STAGES,
        )

        dwg = torch.empty_like(w_gate)
        dwu = torch.empty_like(w_up)
        _wgrad_gateup[(hidden // _WGRAD_BLOCK_H, total_wb)](
            x,
            dg,
            du,
            dwg,
            dwu,
            route_tokens,
            wb_expert,
            plan.expert_col_off,
            row_start,
            counts,
            bm,
            HIDDEN=hidden,
            TOTAL_W=plan.total_width,
            MAX_W=plan.max_width,
            BLOCK_W=_WGRAD_BLOCK_W,
            BLOCK_H=_WGRAD_BLOCK_H,
            BLOCK_K=_WGRAD_BLOCK_K,
            num_warps=_WGRAD_WARPS,
            num_stages=_WGRAD_STAGES,
        )

        dx_rows = torch.empty((row_bound, hidden), dtype=x.dtype, device=dev)
        _dx_rows[(m_tiles, hidden // bn)](
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
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=64,
            num_warps=num_warps,
            num_stages=num_stages,
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
                dwg, dwu, dwd, None, None)


def grouped_moe(x, top_weights, expert_ids, plan, top_k, w_gate, w_up, w_down):
    """Differentiable gated MoE over packed variable-width experts.

    Same calling convention as fused_moe_forward: `expert_ids` and
    `top_weights` are the flattened (tokens * top_k) routed expert ids (int32)
    and routing weights, token-major, straight from topk. Returns
    (tokens, hidden), differentiable w.r.t. x, top_weights and all three
    weights.
    """
    return _GroupedMoE.apply(x, top_weights, expert_ids,
                             w_gate, w_up, w_down, plan, top_k)
