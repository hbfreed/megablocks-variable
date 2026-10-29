# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the grouped variable-width MoE forward AND backward.

Checked against a dense fp32 expert loop through torch autograd: the leaves
are the same bf16 tensors the kernels see, the reference math runs in fp32,
and every gradient (x, routing weights, all three expert weights) must agree
with the kernel gradients to bf16 tolerance. Training must also be bitwise
reproducible: the routing layout is checked exactly, and repeated identical
steps must produce identical outputs and gradients.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

from megablocks.backend import grouped_moe as grouped_moe_module
from megablocks.backend.fused_moe import FusedMoEPlan
from megablocks.backend.grouped_moe import grouped_moe

# (widths, hidden, top_k, num_tokens)
_CASES = [
    ([128] * 8, 256, 2, 1),                       # single token
    ([128] * 8, 256, 8, 5),                       # top_k == num_experts
    ([128, 512, 256, 384], 256, 1, 64),           # top_k == 1
    ([512] * 4, 512, 2, 3),                       # wide experts, few tokens
    ([128] * 64, 512, 4, 17),                     # many narrow experts
    ([128, 256, 384, 512] * 16, 1024, 6, 200),    # ragged, mid batch
    ([128, 256, 384, 512] * 55, 2048, 8, 64),     # served shape, small batch
    ([128, 256, 384, 512] * 55, 2048, 8, 1024),   # served shape, train batch
]
_GRAD_NAMES = ('dx', 'dw_route', 'dwg', 'dwu', 'dwd')


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
        out = out.index_add(
            0, rows,
            (h @ w_down[lo:hi].float()) * top_w[rows, slots, None].float(),
        )
    return out


def _rel(a, b):
    return (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)


def _inputs(widths, hidden, top_k, num_tokens, seed=0, zero_last_weight=False):
    torch.manual_seed(seed)
    dev = torch.device('cuda')
    total = sum(widths)

    def leaf(*shape, scale=1.0):
        t = torch.randn(*shape, device=dev, dtype=torch.bfloat16) * scale
        return t.requires_grad_()

    w_gate = leaf(hidden, total, scale=0.02)
    w_up = leaf(hidden, total, scale=0.02)
    w_down = leaf(total, hidden, scale=0.02)
    x = leaf(num_tokens, hidden)

    logits = torch.randn(num_tokens, len(widths), device=dev)
    top_w0, top_e = torch.topk(F.softmax(logits, dim=1, dtype=torch.float), top_k, -1)
    if zero_last_weight:
        top_w0[:, -1] = 0.0
    top_w = top_w0.to(torch.bfloat16).requires_grad_()

    dout = torch.randn(num_tokens, hidden, device=dev, dtype=torch.bfloat16)
    plan = FusedMoEPlan(widths, hidden, dev)
    return x, top_w, top_e, w_gate, w_up, w_down, dout, plan


def _step(inputs, top_k, down_proj_backend='triton', recompute_activation=True):
    """One forward + backward; returns (out, grads) and clears the leaf grads."""
    x, top_w, top_e, w_gate, w_up, w_down, dout, plan = inputs
    out = grouped_moe(x, top_w.flatten(), top_e.flatten().int(), plan,
                      top_k, w_gate, w_up, w_down, down_proj_backend,
                      recompute_activation)
    out.backward(dout)
    leaves = (x, top_w, w_gate, w_up, w_down)
    grads = [t.grad for t in leaves]
    for t in leaves:
        t.grad = None
    return out.detach(), grads


def _run(
    widths,
    hidden,
    top_k,
    num_tokens,
    seed=0,
    down_proj_backend='triton',
    recompute_activation=True,
    zero_last_weight=False,
):
    inputs = _inputs(widths, hidden, top_k, num_tokens, seed, zero_last_weight)
    got, grads = _step(inputs, top_k, down_proj_backend, recompute_activation)

    x, top_w, top_e, w_gate, w_up, w_down, dout, _ = inputs
    want = _reference(x, top_w, top_e, w_gate, w_up, w_down, widths)
    want.backward(dout.float())
    ref_grads = [t.grad for t in (x, top_w, w_gate, w_up, w_down)]

    errs = {'out': _rel(got, want)}
    for name, ga, gb in zip(_GRAD_NAMES, grads, ref_grads):
        errs[name] = _rel(ga, gb)
    return errs


def _assert_close(errs):
    assert errs['out'] < 5e-3, f'forward error {errs["out"]:.3e}'
    for name in _GRAD_NAMES:
        assert errs[name] < 2e-2, f'{name} error {errs[name]:.3e} ({errs})'


def _require_cute():
    if importlib.util.find_spec('cutlass') is None:
        pytest.skip('nvidia-cutlass-dsl is not installed')
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip('the experimental CuTe backend currently targets Ampere')


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.parametrize('recompute_activation', [True, False])
@pytest.mark.parametrize('widths, hidden, top_k, num_tokens', _CASES)
def test_matches_dense_autograd_reference(
    widths,
    hidden,
    top_k,
    num_tokens,
    recompute_activation,
):
    _assert_close(_run(widths, hidden, top_k, num_tokens,
                       recompute_activation=recompute_activation))


@pytest.mark.gpu
@pytest.mark.world_size(1)
def test_zero_routing_weights():
    # A route can carry weight 0 (masked tokens, renormalized routers); its
    # weight gradient is still <y_row, dout>, which a recovery that divides
    # by w_r would turn into 0/0.
    _assert_close(_run([128, 512, 256, 384], 256, 3, 40, zero_last_weight=True))


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.parametrize(
    'widths, hidden, top_k, num_tokens, recompute_activation',
    [
        ([128] * 8, 256, 2, 1, True),
        ([128, 512, 256, 384], 512, 2, 257, False),
    ],
)
def test_cute_backend_matches_dense_autograd_reference(
    widths,
    hidden,
    top_k,
    num_tokens,
    recompute_activation,
):
    _require_cute()
    _assert_close(_run(
        widths,
        hidden=hidden,
        top_k=top_k,
        num_tokens=num_tokens,
        down_proj_backend='cute',
        recompute_activation=recompute_activation,
    ))


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_training_step_is_bitwise_reproducible(down_proj_backend):
    # 8192 routes span eight routing blocks, and every expert takes rows from
    # several of them, so any order-dependent row assignment shows up as
    # weight-gradient bit differences between identical steps. Retaining the
    # activation instead of recomputing it must not change a bit either.
    if down_proj_backend == 'cute':
        _require_cute()
    top_k = 8
    inputs = _inputs([128, 256, 384, 512] * 16, 1024, top_k, 1024)
    first_out, first_grads = _step(inputs, top_k, down_proj_backend)
    for recompute_activation in (True, True, False):
        out, grads = _step(inputs, top_k, down_proj_backend, recompute_activation)
        assert torch.equal(out, first_out)
        for name, got, want in zip(_GRAD_NAMES, grads, first_grads):
            assert torch.equal(got, want), (
                f'{name} differs bitwise (recompute_activation='
                f'{recompute_activation})')


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.parametrize(
    'widths, top_k, num_tokens',
    [
        ([128] * 8, 2, 5),
        ([128, 256, 384, 512] * 16, 6, 200),    # two routing blocks
        ([128, 256, 384, 512] * 16, 8, 1500),   # twelve: chunked prefix sum
        ([128, 256, 384, 512] * 55, 8, 4096),   # 32 blocks, 220 experts
    ],
)
def test_routing_layout_is_stable(widths, top_k, num_tokens):
    # Each expert's routes fill its rows in route order; padding rows carry
    # token -1 and weight 0; each route's weight sits on its own row.
    torch.manual_seed(0)
    dev = torch.device('cuda')
    plan = FusedMoEPlan(widths, 256, dev)
    logits = torch.randn(num_tokens, len(widths), device=dev)
    top_w, top_e = torch.topk(F.softmax(logits, dim=1), top_k, -1)
    expert_ids = top_e.flatten().int()
    weights = top_w.flatten().to(torch.bfloat16)
    block = grouped_moe_module._ROW_BLOCK
    m_tiles, row_bound = plan.bounds(num_tokens, top_k, block)
    _, route_tokens, route_rows, row_w = grouped_moe_module._route(
        expert_ids, weights, plan, top_k, m_tiles, row_bound)

    # Expected layout, computed on the CPU.
    e = expert_ids.long().cpu()
    routes = torch.arange(e.numel())
    onehot = F.one_hot(e, len(widths))
    tiles = (onehot.sum(0) + block - 1) // block
    start = (torch.cumsum(tiles, 0) - tiles) * block
    want_rows = start[e] + (onehot.cumsum(0) - onehot)[routes, e]
    route_tokens, row_w = route_tokens.cpu(), row_w.cpu()
    assert torch.equal(route_rows.long().cpu(), want_rows)
    assert torch.equal(route_tokens.long()[want_rows], routes // top_k)
    assert torch.equal(row_w[want_rows], weights.float().cpu())
    claimed = int(tiles.sum()) * block
    padding = torch.ones(claimed, dtype=torch.bool)
    padding[want_rows] = False
    assert bool((route_tokens[:claimed][padding] == -1).all())
    assert bool((row_w[:claimed][padding] == 0).all())


@pytest.mark.gpu
@pytest.mark.world_size(1)
def test_forward_saves_no_hidden_wide_route_buffer():
    # With w_r folded into h, backward derives the routing-weight gradient
    # from the intermediate, so the (rows, hidden) down-projection output is
    # not kept. Nothing hidden-wide may be saved besides the input leaves.
    hidden, top_k = 1024, 4
    x, top_w, top_e, w_gate, w_up, w_down, _, plan = _inputs(
        [128, 256] * 8, hidden, top_k, 300)
    out = grouped_moe(x, top_w.flatten(), top_e.flatten().int(), plan, top_k,
                      w_gate, w_up, w_down)
    leaves = {t.data_ptr() for t in (x, top_w, w_gate, w_up, w_down)}
    wide = [
        tuple(t.shape) for t in out.grad_fn.saved_tensors
        if t.data_ptr() not in leaves and t.dim() == 2 and t.shape[1] == hidden
    ]
    assert not wide, f'saved (rows, hidden) buffers: {wide}'
