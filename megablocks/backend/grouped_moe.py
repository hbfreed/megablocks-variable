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

Forward reuses fused_moe's counting-sort routing verbatim, then:

  `_proj_rows`      x @ W for one projection, gathering x rows on the fly;
                    called twice (gate, up), storing the pre-activations
                    (backward needs them -- the stk path stores the same two)
  h = silu(g) * u   one elementwise pass (transient, freed after the forward)
  `_down_proj`      fused_moe's serving kernel on h
  `_scatter_reduce` weighted top-k reduction into the output (from kernels.py)

Backward, all deterministic (fp32 accumulators, fixed reduction order, no
atomics):

  `_route_bwd`      dy_rows[row] = w_r * dout[token]; dw_r = <y_row, dout>
  `_dgu`            dh = dy_rows @ Wd^T, then SwiGLU' in-register -> dg, du
  `_wgrad_rows`     dW = rows^T @ cols for (h, dy_rows) -> dWd
  `_wgrad_gather`   dW = gather(x)^T @ dg -> dWg, and with du -> dWu
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
    _down_proj,
    _plan_tiles,
    _scatter_routes,
)

# Per-kernel launch configs, swept on a 3090 at the healing shape (keep50
# widths, 4096 tokens, top_k 8). Keys: (BLOCK_M, BLOCK_N, BLOCK_K, warps,
# stages). BLOCK_N on the width axis must be a multiple of 128 (the
# width-block granularity); on the hidden axis it must divide the hidden size.
_CFG = {
    'proj': (128, 128, 32, 4, 3),        # _proj_rows (gate and up)
    'down': (128, 256, 64, 8, 3),        # _down_proj on h
    'dgu': (128, 128, 64, 8, 3),         # _dgu
    'dx': (128, 128, 64, 8, 3),          # _dx_rows
    'wgrad_down': (64, 128, 32, 4, 2),   # _wgrad_rows: (BLOCK_W, BLOCK_H, K)
    'wgrad_gateup': (128, 128, 64, 4, 3),  # _wgrad_gather: (BLOCK_W, BLOCK_H, K)
}
# The row-tile layout is planned once with this block; every row-tile kernel
# uses the same BLOCK_M so tiles stay aligned to the planned buffers.
_ROW_BLOCK = 128


# One projection over gathered rows: out[row, :w_e] = x[tok(row)] @ W_e.
# Padding rows load x as zero, so they produce exact zeros.
@triton.jit
def _proj_rows(
    x,
    w,
    out,
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

    tl.store(out + rows[:, None] * MAX_W + local[None, :],
             acc.to(out.dtype.element_ty))


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
# silu'(g) = sig(g) * (1 + g * (1 - sig(g))). Wd rows are loaded contiguously
# as (N, K) and transposed in-register.
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
    dg = acc * uu * (sig * (1.0 + gg * (1.0 - sig)))
    du = acc * silu
    tl.store(dg_out + src, dg.to(dg_out.dtype.element_ty))
    tl.store(du_out + src, du.to(du_out.dtype.element_ty))


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


# One term of dx_rows = dg @ Wg^T + du @ Wu^T over the expert's own width;
# launched twice, the second launch accumulating onto the first (ACCUM).
# Splitting the two weight streams lets each launch run a clean single-stream
# GEMM; the top-k sum back into token space is kernels._scatter_reduce with
# SCALE=False. W rows are loaded contiguously as (H, K), transposed in-register.
@triton.jit
def _dx_rows(
    d_in,
    w,
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
    ACCUM: tl.constexpr,
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
        a = tl.load(d_in + base + kk[None, :])
        wsrc = hcol[:, None] * TOTAL_W + (coff + kk)[None, :]
        acc += tl.dot(a, tl.trans(tl.load(w + wsrc)))

    dst = dx + rows[:, None] * HIDDEN + hcol[None, :]
    if ACCUM:
        acc += tl.load(dst).to(tl.float32)
    tl.store(dst, acc.to(dx.dtype.element_ty))


# h = silu(g) * u in one pass. F.silu(g) * u is two eager kernels and an
# extra intermediate write; this is one read of g/u and one write of h.
@triton.jit
def _silu_mul(
    g_in,
    u_in,
    h_out,
    numel,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < numel
    gg = tl.load(g_in + off, mask=m, other=0.0).to(tl.float32)
    uu = tl.load(u_in + off, mask=m, other=0.0).to(tl.float32)
    tl.store(h_out + off, (gg * tl.sigmoid(gg) * uu).to(h_out.dtype.element_ty),
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


def _silu_mul_launch(g, u):
    h = torch.empty_like(g)
    n = g.numel()
    _silu_mul[(triton.cdiv(n, 4096),)](g, u, h, n, BLOCK=4096, num_warps=8)
    return h


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
        bm = _ROW_BLOCK
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

        pm, pn, pk, pw, ps = _CFG['proj']
        g = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
        u = torch.empty((row_bound, plan.max_width), dtype=x.dtype, device=dev)
        proj_grid = (m_tiles, (plan.max_width + pn - 1) // pn)
        for w, out in ((w_gate, g), (w_up, u)):
            _proj_rows[proj_grid](
                x,
                w,
                out,
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
                num_warps=pw,
                num_stages=ps,
            )

        # Transient: freed after the down projection; backward recomputes it.
        h = _silu_mul_launch(g, u)

        dm, dn, dk, dw_, ds = _CFG['down']
        dn = min(dn, hidden)
        y = torch.empty((row_bound, hidden), dtype=x.dtype, device=dev)
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
        del h

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
        ctx.launch = (bm, m_tiles, row_bound)
        return out

    @staticmethod
    def backward(ctx, dout):
        (x, top_weights, w_gate, w_up, w_down,
         route_tokens, route_rows, tile_expert,
         counts, row_start, g, u, y) = ctx.saved_tensors
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

        gm, gn, gk, gw, gs = _CFG['dgu']
        dg = torch.empty_like(g)
        du = torch.empty_like(u)
        _dgu[(m_tiles, (plan.max_width + gn - 1) // gn)](
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
            NBLOCK_N=gn // bn,
            BLOCK_M=gm,
            BLOCK_N=gn,
            BLOCK_K=gk,
            num_warps=gw,
            num_stages=gs,
        )

        # One elementwise pass beats recomputing silu(g)*u inside every K
        # iteration of the weight-gradient kernel. Transient; freed after.
        h = _silu_mul_launch(g, u)
        ww, wh, wk, wwar, wst = _CFG['wgrad_down']
        dwd = torch.empty_like(w_down)
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

        xm, xn, xk, xw, xs = _CFG['dx']
        xn = min(xn, hidden)
        dx_rows = torch.empty((row_bound, hidden), dtype=x.dtype, device=dev)
        for accum, (d_rows, w) in enumerate(((dg, w_gate), (du, w_up))):
            _dx_rows[(m_tiles, hidden // xn)](
                d_rows,
                w,
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
                ACCUM=bool(accum),
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
