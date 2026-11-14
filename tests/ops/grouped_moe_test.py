# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the grouped variable-width MoE forward AND backward.

Checked against a dense fp32 expert loop through torch autograd: the leaves
are the same bf16 tensors the kernels see, the reference math runs in fp32,
and every gradient (x, routing weights, all three expert weights) must agree
with the kernel gradients to bf16 tolerance. Training must also be bitwise
reproducible: the routing layout is checked exactly, and repeated identical
steps must produce identical outputs and gradients. Nearest token rounding is
checked against the same reference restricted to the routes it keeps.
"""

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


def _reference(x, top_w, top_e, w_gate, w_up, w_down, widths, keep=None):
    offsets = [0]
    for w in widths:
        offsets.append(offsets[-1] + w)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    xf = x.float()
    for e in range(len(widths)):
        lo, hi = offsets[e], offsets[e + 1]
        routed = top_e == e if keep is None else (top_e == e) & keep
        rows, slots = torch.where(routed)
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


def _nearest_rounding(top_e, scores, num_experts):
    """(keep, rank) under nearest token rounding, by the kernels' own rule.

    Keys are expert-major with descending quantized score, as in
    kernels._route_key; a stable sort ranks each expert's routes, and an expert
    keeps min(count, count rounded half up to the row tile, but at least one
    tile) of them.
    """
    block = grouped_moe_module._ROW_BLOCK
    bits = grouped_moe_module._route_score_bits(num_experts)
    scale = (1 << bits) - 1
    e = top_e.flatten().long().cpu()
    descending = ((1.0 - scores.flatten().float().cpu()) * scale).to(torch.int32)
    keys = (e << bits) | descending.clamp(0, scale).long()
    order = torch.sort(keys, stable=True).indices
    counts = torch.bincount(e, minlength=num_experts)
    rounded = ((counts + block // 2) // block * block).clamp_min(block)
    kept = torch.minimum(counts, rounded)
    start = torch.cumsum(counts, 0) - counts
    rank = torch.empty_like(e)
    rank[order] = torch.arange(e.numel()) - start[e[order]]
    return (rank < kept[e]).view(top_e.shape), rank.view(top_e.shape)


def _inputs(
    widths,
    hidden,
    top_k,
    num_tokens,
    seed=0,
    zero_last_weight=False,
    renormalize=False,
    block_n=128,
):
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
    # The router probabilities rank routes for token rounding; renormalizing
    # the weights, as gpt_model.MoEMLP does, makes the two orders differ.
    scores = top_w0
    if renormalize:
        top_w0 = top_w0 / top_w0.sum(dim=-1, keepdim=True)
    top_w = top_w0.to(torch.bfloat16).requires_grad_()

    dout = torch.randn(num_tokens, hidden, device=dev, dtype=torch.bfloat16)
    plan = FusedMoEPlan(widths, hidden, dev, block_n=block_n)
    return x, top_w, top_e, w_gate, w_up, w_down, dout, plan, scores


def _step(
    inputs,
    top_k,
    down_proj_backend='auto',
    recompute_activation=True,
    token_rounding='ceil',
):
    """One forward + backward; returns (out, grads) and clears the leaf grads."""
    x, top_w, top_e, w_gate, w_up, w_down, dout, plan, scores = inputs
    out = grouped_moe(x, top_w.flatten(), top_e.flatten().int(), plan,
                      top_k, w_gate, w_up, w_down, down_proj_backend,
                      recompute_activation, token_rounding,
                      scores.flatten() if token_rounding == 'nearest' else None)
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
    token_rounding='ceil',
    block_n=128,
):
    inputs = _inputs(widths, hidden, top_k, num_tokens, seed, zero_last_weight,
                     renormalize=token_rounding == 'nearest', block_n=block_n)
    got, grads = _step(inputs, top_k, down_proj_backend, recompute_activation,
                       token_rounding)

    x, top_w, top_e, w_gate, w_up, w_down, dout, _, scores = inputs
    keep = None
    if token_rounding == 'nearest':
        keep, _ = _nearest_rounding(top_e, scores, len(widths))
        keep = keep.to(top_e.device)
    want = _reference(x, top_w, top_e, w_gate, w_up, w_down, widths, keep)
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
    if grouped_moe_module._cute() is None:
        pytest.skip('the cute extra is not installed')
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip('the experimental CuTe backend currently targets Ampere')


@pytest.mark.gpu
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
def test_zero_routing_weights():
    # A route can carry weight 0 (masked tokens, renormalized routers); its
    # weight gradient is still <y_row, dout>, which a recovery that divides
    # by w_r would turn into 0/0.
    _assert_close(_run([128, 512, 256, 384], 256, 3, 40, zero_last_weight=True))


@pytest.mark.gpu
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
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
@pytest.mark.parametrize(
    'widths, hidden, top_k, num_tokens',
    [
        ([128] * 8, 256, 2, 700),                     # ~175 routes per expert
        ([128, 256, 384, 512] * 16, 1024, 6, 1500),   # ~140 routes per expert
    ],
)
def test_nearest_token_rounding_matches_reference(
    widths,
    hidden,
    top_k,
    num_tokens,
    down_proj_backend,
):
    # The reference drops the same routes and runs the dense expert loop over
    # the rest; dropped routes must get zero routing-weight gradient.
    if down_proj_backend == 'cute':
        _require_cute()
    _assert_close(_run(widths, hidden, top_k, num_tokens,
                       down_proj_backend=down_proj_backend,
                       token_rounding='nearest'))


@pytest.mark.gpu
@pytest.mark.parametrize(
    'down_proj_backend, token_rounding',
    [('triton', 'ceil'), ('cute', 'ceil'), ('triton', 'nearest')],
)
def test_training_step_is_bitwise_reproducible(down_proj_backend, token_rounding):
    # 8192 routes span eight routing blocks, and every expert takes rows from
    # several of them, so any order-dependent row assignment shows up as
    # weight-gradient bit differences between identical steps. Retaining the
    # activation instead of recomputing it must not change a bit either.
    if down_proj_backend == 'cute':
        _require_cute()
    top_k = 8
    inputs = _inputs([128, 256, 384, 512] * 16, 1024, top_k, 1024)
    first_out, first_grads = _step(inputs, top_k, down_proj_backend,
                                   token_rounding=token_rounding)
    for recompute_activation in (True, True, False):
        out, grads = _step(inputs, top_k, down_proj_backend,
                           recompute_activation, token_rounding)
        assert torch.equal(out, first_out)
        for name, got, want in zip(_GRAD_NAMES, grads, first_grads):
            assert torch.equal(got, want), (
                f'{name} differs bitwise (recompute_activation='
                f'{recompute_activation})')


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_unaligned_dout(down_proj_backend):
    # The backward gathers read the caller's dout directly, and a contiguous
    # view of it need not be 16-byte aligned.
    if down_proj_backend == 'cute':
        _require_cute()
    top_k = 8
    inputs = _inputs([128, 256, 384, 512] * 16, 1024, top_k, 1024)
    _, want = _step(inputs, top_k, down_proj_backend)
    dout = inputs[6]
    flat = torch.cat((dout.new_zeros(1), dout.flatten()))
    unaligned = flat[1:].view(dout.shape)
    assert unaligned.data_ptr() % 16
    _, got = _step(inputs[:6] + (unaligned,) + inputs[7:], top_k,
                   down_proj_backend)
    for name, g, w in zip(_GRAD_NAMES, got, want):
        assert torch.equal(g, w), f'{name} differs bitwise'


@pytest.mark.gpu
@pytest.mark.parametrize(
    'widths, top_k, num_tokens',
    [
        ([128] * 8, 2, 5),
        ([128, 256, 384, 512] * 16, 6, 200),    # two routing blocks
        ([128] * 8, 8, 128),                    # exactly one routing block
        ([128, 256, 384, 512] * 16, 8, 1024),   # exactly eight
        ([128, 256, 384, 512] * 16, 8, 1500),   # twelve
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
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_forward_saves_no_hidden_wide_route_buffer(down_proj_backend):
    # With w_r folded into h, backward derives the routing-weight gradient
    # from the intermediate, so the (rows, hidden) down-projection output is
    # not kept. Nothing hidden-wide may be saved besides the input leaves.
    if down_proj_backend == 'cute':
        _require_cute()
    hidden, top_k = 1024, 4
    x, top_w, top_e, w_gate, w_up, w_down, _, plan, _ = _inputs(
        [128, 256] * 8, hidden, top_k, 300)
    out = grouped_moe(x, top_w.flatten(), top_e.flatten().int(), plan, top_k,
                      w_gate, w_up, w_down, down_proj_backend)
    leaves = {t.data_ptr() for t in (x, top_w, w_gate, w_up, w_down)}
    wide = [
        tuple(t.shape) for t in out.grad_fn.saved_tensors
        if t.data_ptr() not in leaves and t.dim() == 2 and t.shape[1] == hidden
    ]
    assert not wide, f'saved (rows, hidden) buffers: {wide}'


@pytest.mark.gpu
def test_nearest_rounding_layout():
    # Exactly the expected routes are dropped (row -1); each expert's kept
    # routes fill its rows in descending-score order; an expert that rounded
    # down has no padding; padding rows carry token -1 and weight 0.
    torch.manual_seed(0)
    dev = torch.device('cuda')
    widths, top_k, num_tokens = [128, 256, 384, 512] * 16, 6, 1500
    plan = FusedMoEPlan(widths, 256, dev)
    probs = F.softmax(torch.randn(num_tokens, len(widths), device=dev), dim=1)
    scores, top_e = torch.topk(probs, top_k, -1)
    weights = (scores / scores.sum(dim=-1, keepdim=True)).flatten().to(torch.bfloat16)
    block = grouped_moe_module._ROW_BLOCK
    m_tiles, row_bound = plan.bounds(num_tokens, top_k, block)
    _, route_tokens, route_rows, row_w = grouped_moe_module._route(
        top_e.flatten().int(), weights, plan, top_k, m_tiles, row_bound,
        'nearest', scores.flatten())

    keep, rank = _nearest_rounding(top_e, scores, len(widths))
    keep, rank = keep.flatten(), rank.flatten()
    e = top_e.flatten().long().cpu()
    kept = torch.bincount(e[keep], minlength=len(widths))
    counts = torch.bincount(e, minlength=len(widths))
    assert bool((kept < counts).any()), 'no expert rounded down'
    assert bool((kept % block != 0).any()), 'no expert rounded up'
    tiles = (kept + block - 1) // block
    start = (torch.cumsum(tiles, 0) - tiles) * block
    routes = torch.arange(e.numel())
    rows = route_rows.long().cpu()
    assert torch.equal(rows < 0, ~keep)
    assert torch.equal(rows[keep], (start[e] + rank)[keep])

    route_tokens, row_w = route_tokens.cpu(), row_w.cpu()
    assert torch.equal(route_tokens.long()[rows[keep]], routes[keep] // top_k)
    assert torch.equal(row_w[rows[keep]], weights.float().cpu()[keep])
    claimed = int(tiles.sum()) * block
    padding = torch.ones(claimed, dtype=torch.bool)
    padding[rows[keep]] = False
    assert bool((route_tokens[:claimed][padding] == -1).all())
    assert bool((row_w[:claimed][padding] == 0).all())


@pytest.mark.gpu
@pytest.mark.parametrize('token_rounding', ['ceil', 'nearest'])
def test_route_stats(token_rounding):
    # The fractions gpt_model.MoEMLP reports, left on the device. Asking for
    # them must not change the output or any gradient.
    widths, top_k = [128, 256, 384, 512] * 16, 6
    inputs = _inputs(widths, 1024, top_k, 1500, renormalize=True)
    x, top_w, top_e, w_gate, w_up, w_down, dout, plan, scores = inputs
    want_out, want_grads = _step(inputs, top_k, token_rounding=token_rounding)

    nearest = token_rounding == 'nearest'
    out, stats = grouped_moe(
        x, top_w.flatten(), top_e.flatten().int(), plan, top_k, w_gate, w_up,
        w_down, token_rounding=token_rounding,
        route_scores=scores.flatten() if nearest else None,
        return_route_stats=True)
    out.backward(dout)
    leaves = (x, top_w, w_gate, w_up, w_down)
    assert torch.equal(out.detach(), want_out)
    for name, leaf, want in zip(_GRAD_NAMES, leaves, want_grads):
        assert torch.equal(leaf.grad, want), f'{name} changed'
        leaf.grad = None

    e = top_e.flatten().long().cpu()
    counts = torch.bincount(e, minlength=len(widths))
    kept = counts
    if nearest:
        keep, _ = _nearest_rounding(top_e, scores, len(widths))
        kept = torch.bincount(e[keep.flatten()], minlength=len(widths))
    block = grouped_moe_module._ROW_BLOCK
    rows = (kept + block - 1) // block * block
    dropped = float((counts - kept).sum()) / e.numel()
    padding = float((rows - kept).sum()) / e.numel()
    assert (dropped > 0) == nearest
    for name, want in (('dropped_route_fraction', dropped),
                       ('padding_route_fraction', padding)):
        got = stats[name]
        assert got.device.type == 'cuda' and got.dim() == 0
        assert abs(got.item() - want) < 1e-6, (name, got.item(), want)


@pytest.mark.gpu
def test_cute_short_k_loop():
    # One K tile with three pipeline stages: the prologue must still commit one
    # cp.async group per stage, or its wait passes with data in flight. A 32-wide
    # expert under a 32 width block reaches that, and it returned 32% error
    # before. grouped_moe needs 128-wide blocks, so this drives the kernel directly.
    _require_cute()
    from megablocks.backend.cute_down_proj import cute_down_proj

    torch.manual_seed(0)
    dev = torch.device('cuda')
    widths, hidden = [32, 128], 256
    plan = FusedMoEPlan(widths, hidden, dev, block_n=32)
    w_down = torch.randn(sum(widths), hidden, device=dev, dtype=torch.bfloat16) * 0.05
    # One 128-row tile per expert; columns past an expert's width hold values
    # the kernel must not read.
    h = torch.randn(256, plan.max_width, device=dev, dtype=torch.bfloat16)
    tile_expert = torch.tensor([0, 1], device=dev, dtype=torch.int32)
    want = torch.cat((h[:128, :32].float() @ w_down[:32].float(),
                      h[128:].float() @ w_down[32:].float()))
    for _ in range(3):
        y = torch.empty(256, hidden, device=dev, dtype=torch.bfloat16)
        cute_down_proj(h, w_down, y, tile_expert, plan.expert_col_off,
                       plan.expert_nblocks, plan.block_n)
        assert _rel(y, want) < 1e-2


@pytest.mark.gpu
def test_cute_one_row_view():
    # A one-row view is contiguous but keeps its parent's stride, which compiling
    # a CuTe kernel used to reject. It must match the same data in a fresh tensor.
    _require_cute()
    torch.manual_seed(0)
    dev = torch.device('cuda')
    widths, hidden, total = [256, 128], 512, 384
    big_x = torch.randn(4, 1024, device=dev, dtype=torch.bfloat16)
    big_dout = torch.randn(4, 1024, device=dev, dtype=torch.bfloat16)
    w_gate = (torch.randn(hidden, total, device=dev, dtype=torch.bfloat16) * 0.05).requires_grad_()
    w_up = (torch.randn(hidden, total, device=dev, dtype=torch.bfloat16) * 0.05).requires_grad_()
    w_down = (torch.randn(total, hidden, device=dev, dtype=torch.bfloat16) * 0.05).requires_grad_()
    top_e = torch.tensor([0], device=dev, dtype=torch.int32)
    plan = FusedMoEPlan(widths, hidden, dev)

    def step(x, dout):
        top_w = torch.ones(1, device=dev, dtype=torch.bfloat16)
        out = grouped_moe(x, top_w, top_e, plan, 1, w_gate, w_up, w_down,
                          down_proj_backend='cute')
        grads = torch.autograd.grad(out, (w_gate, w_up, w_down), dout)
        return out.detach(), grads

    view = step(big_x[2:3, :512], big_dout[1:2, :512])
    fresh = step(big_x[2:3, :512].clone(), big_dout[1:2, :512].clone())
    assert torch.equal(view[0], fresh[0])
    for got, want in zip(view[1], fresh[1]):
        assert torch.equal(got, want)


@pytest.mark.gpu
@pytest.mark.parametrize('hidden', [64, 192, 384])
def test_unsupported_hidden_raises(hidden):
    # These sizes would leave part of the output or weight gradients unwritten.
    dev = torch.device('cuda')
    widths = [128, 256]
    x = torch.randn(4, hidden, device=dev, dtype=torch.bfloat16)
    w = torch.randn(hidden, sum(widths), device=dev, dtype=torch.bfloat16)
    plan = FusedMoEPlan(widths, hidden, dev)
    top_e = torch.zeros(4, device=dev, dtype=torch.int32)
    top_w = torch.ones(4, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='hidden must be'):
        grouped_moe(x, top_w, top_e, plan, 1, w, w, w.T)


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_runs_on_current_stream(down_proj_backend):
    # Every kernel must launch on torch's current stream. CuTe used to launch on
    # the default stream and race the Triton kernels around it.
    if down_proj_backend == 'cute':
        _require_cute()
    top_k = 8
    inputs = _inputs([128, 256, 384, 512] * 16, 1024, top_k, 1024)
    want_out, want_grads = _step(inputs, top_k, down_proj_backend)
    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(8):
            out, grads = _step(inputs, top_k, down_proj_backend)
            assert torch.equal(out, want_out)
            for name, got, want in zip(_GRAD_NAMES, grads, want_grads):
                assert torch.equal(got, want), f'{name} differs bitwise'


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
@pytest.mark.parametrize(
    'widths, block_n',
    [([256, 512, 768, 256], 256), ([512, 1024], 512)],
)
def test_width_block_above_128(widths, block_n, down_proj_backend):
    # Tiles past an expert's width must not run, or their partials corrupt dw_r.
    if down_proj_backend == 'cute':
        _require_cute()
    _assert_close(_run(widths, 1024, 2, 300, down_proj_backend=down_proj_backend,
                       block_n=block_n))


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_noncontiguous_inputs(down_proj_backend):
    # Strided x and weights must give what the same data gives in fresh tensors.
    if down_proj_backend == 'cute':
        _require_cute()
    top_k = 2
    inputs = _inputs([128, 256] * 4, 512, top_k, 100)
    want_out, want_grads = _step(inputs, top_k, down_proj_backend)
    x, top_w, top_e, w_gate, w_up, w_down, dout, plan, _ = inputs

    def strided(t):
        # A leaf holding t's data in every other column, and that view of it.
        leaf = torch.zeros(t.shape[0], 2 * t.shape[1], dtype=t.dtype, device=t.device)
        leaf[:, ::2] = t.detach()
        leaf.requires_grad_()
        return leaf, leaf[:, ::2]

    leaves, views = zip(*(strided(t) for t in (x, w_gate, w_up, w_down)))
    assert not any(v.is_contiguous() for v in views)
    out = grouped_moe(views[0], top_w.flatten(), top_e.flatten().int(), plan, top_k,
                      views[1], views[2], views[3],
                      down_proj_backend=down_proj_backend)
    out.backward(dout)
    assert torch.equal(out.detach(), want_out)
    # want_grads is (dx, dw_route, dwg, dwu, dwd).
    for leaf, want in zip(leaves, (want_grads[0], *want_grads[2:])):
        assert torch.equal(leaf.grad[:, ::2], want)
    assert torch.equal(top_w.grad, want_grads[1])


@pytest.mark.gpu
@pytest.mark.parametrize(
    'widths, hidden, top_k, num_tokens',
    [
        ([128, 256, 384, 512] * 55, 256, 2, 100),  # every expert under half a tile
        ([128, 256] * 4, 256, 2, 100),             # some under, some over
    ],
)
def test_nearest_keeps_experts_under_half_a_tile(widths, hidden, top_k, num_tokens):
    # Rounding them to zero stopped those experts training for the step, and with
    # every expert under half a tile zeroed the whole layer without any error.
    inputs = _inputs(widths, hidden, top_k, num_tokens, renormalize=True)
    x, top_w, top_e, w_gate, w_up, w_down, _, plan, scores = inputs
    out, stats = grouped_moe(x, top_w.flatten(), top_e.flatten().int(), plan, top_k,
                             w_gate, w_up, w_down, token_rounding='nearest',
                             route_scores=scores.flatten(), return_route_stats=True)
    assert bool((out != 0).any())
    counts = torch.bincount(top_e.flatten(), minlength=len(widths))
    if int(counts.max()) < 64:
        assert stats['dropped_route_fraction'].item() == 0.0
    _assert_close(_run(widths, hidden, top_k, num_tokens, token_rounding='nearest'))


@pytest.mark.gpu
@pytest.mark.parametrize('widths, block_n', [([32, 128], 32), ([64, 128, 192], 64)])
def test_narrow_width_block_raises(widths, block_n):
    # The row-tile and weight-gradient kernels tile the width axis in 128s. Under
    # a narrower block they returned a wrong output and garbage weight gradients.
    x, top_w, top_e, w_gate, w_up, w_down, _, plan, _ = _inputs(
        widths, 256, 2, 20, block_n=block_n)
    with pytest.raises(ValueError, match='multiple of 128'):
        grouped_moe(x, top_w.flatten(), top_e.flatten().int(), plan, 2,
                    w_gate, w_up, w_down)


@pytest.mark.gpu
def test_int64_expert_ids_raise():
    # topk returns int64 indices, and the routing kernels failed to compile on
    # them with a Triton compiler dump instead of an error.
    x, top_w, top_e, w_gate, w_up, w_down, _, plan, _ = _inputs(
        [128, 256], 256, 2, 20)
    with pytest.raises(TypeError, match='int32'):
        grouped_moe(x, top_w.flatten(), top_e.flatten(), plan, 2,
                    w_gate, w_up, w_down)


@pytest.mark.gpu
def test_auto_picks_cute():
    # Where the CuTe kernels run, 'auto' must run them, bit for bit.
    _require_cute()
    top_k = 2
    inputs = _inputs([128, 256] * 4, 512, top_k, 100)
    want_out, want_grads = _step(inputs, top_k, 'cute')
    out, grads = _step(inputs, top_k, 'auto')
    assert torch.equal(out, want_out)
    for name, got, want in zip(_GRAD_NAMES, grads, want_grads):
        assert torch.equal(got, want), f'{name} differs bitwise'


@pytest.mark.gpu
def test_auto_falls_back_to_triton(monkeypatch):
    def resolve(hidden=512, dtype=torch.bfloat16):
        x = torch.empty(4, hidden, device='cuda', dtype=dtype)
        w = torch.empty(hidden, 128, device='cuda', dtype=dtype)
        return grouped_moe_module._resolve_backend(x, w, w, w.T)

    assert resolve(hidden=128) == 'triton'
    assert resolve(dtype=torch.float16) == 'triton'
    monkeypatch.setattr(grouped_moe_module, '_cute', lambda: None)
    assert resolve() == 'triton'


@pytest.mark.gpu
def test_unknown_backend_raises():
    x, top_w, top_e, w_gate, w_up, w_down, _, plan, _ = _inputs(
        [128, 256], 256, 2, 20)
    with pytest.raises(ValueError, match="'auto', 'triton' or 'cute'"):
        grouped_moe(x, top_w.flatten(), top_e.flatten().int(), plan, 2,
                    w_gate, w_up, w_down, down_proj_backend='cuda')


@pytest.mark.gpu
@pytest.mark.parametrize('recompute_activation', [True, False])
@pytest.mark.parametrize('token_rounding', ['ceil', 'nearest'])
def test_custom_op_registration(token_rounding, recompute_activation):
    # Schema, fake implementation and autograd registration of the forward op.
    # opcheck's AOT test compares every output, and the row buffers the forward
    # hands to backward hold unwritten padding rows, so the compile tests below
    # check the outputs that matter instead.
    top_k = 4
    x, top_w, top_e, w_gate, w_up, w_down, _, plan, scores = _inputs(
        [128, 256, 384, 512] * 4, 256, top_k, 33)
    torch.library.opcheck(
        grouped_moe_module._grouped_moe_fwd,
        (x, top_w.flatten(), top_e.flatten().int(), w_gate, w_up, w_down,
         scores.flatten() if token_rounding == 'nearest' else None,
         plan.op_key, top_k, 'triton', recompute_activation, token_rounding,
         True, -1),
        test_utils=('test_schema', 'test_autograd_registration',
                    'test_faketensor'),
    )


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
@pytest.mark.parametrize('token_rounding', ['ceil', 'nearest'])
def test_compiled_matches_eager(down_proj_backend, token_rounding):
    # torch.compile traces through grouped_moe without a graph break, and the
    # compiled step computes exactly what the eager one does.
    if down_proj_backend == 'cute':
        _require_cute()
    top_k = 8
    inputs = _inputs([128, 256, 384, 512] * 16, 1024, top_k, 1024,
                     renormalize=token_rounding == 'nearest')
    x, top_w, top_e, w_gate, w_up, w_down, dout, plan, scores = inputs

    def layer(x, top_w, top_e, scores):
        # Elementwise work on both sides, which compile fuses around the op.
        out = grouped_moe(x * 2, top_w.flatten(), top_e.flatten().int(), plan,
                          top_k, w_gate, w_up, w_down, down_proj_backend,
                          token_rounding=token_rounding,
                          route_scores=(scores.flatten()
                                        if token_rounding == 'nearest' else None))
        return out + x

    compiled = torch.compile(layer, fullgraph=True, dynamic=False)
    leaves = (x, top_w, w_gate, w_up, w_down)
    results = []
    for fn in (layer, compiled):
        out = fn(x, top_w, top_e, scores)
        out.backward(dout)
        results.append((out.detach(), [t.grad for t in leaves]))
        for t in leaves:
            t.grad = None
    (want, want_grads), (got, got_grads) = results
    assert torch.equal(got, want)
    for name, g, w in zip(_GRAD_NAMES, got_grads, want_grads):
        assert torch.equal(g, w), f'{name} differs bitwise'


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_cuda_graph_replay(down_proj_backend):
    # A captured forward + backward replays correctly after its inputs change in
    # place, including the routing, so no size or launch depends on the host.
    if down_proj_backend == 'cute':
        _require_cute()
    top_k = 8
    inputs = _inputs([128, 256, 384, 512] * 16, 1024, top_k, 1024)
    x, top_w, top_e, w_gate, w_up, w_down, dout, plan, scores = inputs
    static_e = top_e.flatten().int()
    leaves = (x, top_w, w_gate, w_up, w_down)

    def step():
        out = grouped_moe(x, top_w.flatten(), static_e, plan, top_k,
                          w_gate, w_up, w_down, down_proj_backend)
        out.backward(dout)
        return out

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):  # compiles every kernel outside the capture
            step()
            for t in leaves:
                t.grad = None
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = step()

    for seed in (1, 2):
        new = _inputs([128, 256, 384, 512] * 16, 1024, top_k, 1024, seed=seed)
        with torch.no_grad():
            for t, n in zip(leaves, (new[0], new[1], new[3], new[4], new[5])):
                t.copy_(n)
            static_e.copy_(new[2].flatten().int())
            dout.copy_(new[6])
        graph.replay()
        got = static_out.clone(), [t.grad.clone() for t in leaves]
        want_out, want_grads = _step(new, top_k, down_proj_backend)
        assert torch.equal(got[0], want_out)
        for name, g, w in zip(_GRAD_NAMES, got[1], want_grads):
            assert torch.equal(g, w), f'{name} differs bitwise (seed {seed})'


# Gradient-accumulation fusion (register_weight_grads / weight_grad_key).
_ACCUM_CASE = ([128, 256, 384, 512] * 16, 1024, 8, 1024)
_WEIGHT_NAMES = ('dwg', 'dwu', 'dwd')


def _micro_batches(n, seed=0, weight_dtype=torch.bfloat16):
    """n micro-batches of fresh tokens, routing and dout over one set of weights."""
    widths, hidden, top_k, num_tokens = _ACCUM_CASE
    batches = [_inputs(widths, hidden, top_k, num_tokens, seed=seed + i)
               for i in range(n)]
    weights = [w.detach().to(weight_dtype).requires_grad_()
               for w in batches[0][3:6]]
    plan = batches[0][7]
    return [(b[0], b[1], b[2], b[6]) for b in batches], weights, plan


def _accumulate(batches, weights, plan, backend, key=None, fn=None):
    """Forward + backward over each micro-batch, without clearing the weight
    grads in between. Returns per-micro-batch (out, dx, dtw) and weight-grad
    snapshots."""
    top_k = _ACCUM_CASE[2]
    fn = fn or (lambda x, top_w, top_e, wg, wu, wd: grouped_moe(
        x, top_w.flatten(), top_e.flatten().int(), plan, top_k, wg, wu, wd,
        backend, weight_grad_key=key))
    acts, snaps = [], []
    for x, top_w, top_e, dout in batches:
        out = fn(x, top_w, top_e,
                 *(w.to(torch.bfloat16) for w in weights))
        out.backward(dout)
        acts.append((out.detach(), x.grad, top_w.grad))
        x.grad = top_w.grad = None
        snaps.append([None if w.grad is None else w.grad.clone()
                      for w in weights])
    return acts, snaps


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_weight_grad_accumulation_matches_autograd(down_proj_backend):
    # Three micro-batches into the weights' .grad: the first writes it (it is
    # None) bitwise as autograd would, the later ones add in fp32 and round
    # once, so they differ from autograd's separate bf16 add by bf16 rounding
    # only. Every other output and gradient is bitwise unchanged.
    if down_proj_backend == 'cute':
        _require_cute()
    batches, weights, plan = _micro_batches(3)
    want_acts, want = _accumulate(batches, weights, plan, down_proj_backend)
    for w in weights:
        w.grad = None
    key = grouped_moe_module.register_weight_grads(*weights)
    got_acts, got = _accumulate(batches, weights, plan, down_proj_backend, key)

    for (go, gdx, gdtw), (wo, wdx, wdtw) in zip(got_acts, want_acts):
        assert torch.equal(go, wo) and torch.equal(gdx, wdx)
        assert torch.equal(gdtw, wdtw)
    for name, g, w in zip(_WEIGHT_NAMES, got[0], want[0]):
        assert torch.equal(g, w), f'{name} differs bitwise after one micro-batch'
    for step in (1, 2):
        for name, g, w in zip(_WEIGHT_NAMES, got[step], want[step]):
            err = _rel(g, w)
            assert err < 4e-3, f'{name} after {step + 1} micro-batches: {err:.3e}'


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_weight_grad_accumulation_in_place(down_proj_backend):
    # A zeroed .grad (zero_grad(set_to_none=False)) is accumulated into in
    # place, and acc + 0 rounds to the unfused gradient bitwise.
    if down_proj_backend == 'cute':
        _require_cute()
    batches, weights, plan = _micro_batches(1)
    _, want = _accumulate(batches, weights, plan, down_proj_backend)
    for w in weights:
        w.grad = torch.zeros_like(w)
    ptrs = [w.grad.data_ptr() for w in weights]
    key = grouped_moe_module.register_weight_grads(*weights)
    _, got = _accumulate(batches, weights, plan, down_proj_backend, key)
    assert [w.grad.data_ptr() for w in weights] == ptrs
    for name, g, w in zip(_WEIGHT_NAMES, got[0], want[0]):
        assert torch.equal(g, w), f'{name} differs bitwise'


@pytest.mark.gpu
def test_weight_grad_accumulation_fp32_weights():
    # fp32 leaves cast to bf16 for the op cannot take the bf16 kernels' output
    # in place, so it is added the way AccumulateGrad adds it: bitwise as before.
    batches, weights, plan = _micro_batches(2, weight_dtype=torch.float32)
    _, want = _accumulate(batches, weights, plan, 'triton')
    for w in weights:
        w.grad = None
    key = grouped_moe_module.register_weight_grads(*weights)
    _, got = _accumulate(batches, weights, plan, 'triton', key)
    for step in range(2):
        for name, g, w in zip(_WEIGHT_NAMES, got[step], want[step]):
            assert g.dtype == torch.float32
            assert torch.equal(g, w), f'{name} differs bitwise'


@pytest.mark.gpu
def test_weight_grad_accumulation_skips_frozen_weight():
    batches, weights, plan = _micro_batches(1)
    _, want = _accumulate(batches, weights, plan, 'triton')
    for w in weights:
        w.grad = None
    weights[2].requires_grad_(False)
    key = grouped_moe_module.register_weight_grads(*weights)
    _, got = _accumulate(batches, weights, plan, 'triton', key)
    assert got[0][2] is None
    for name, g, w in zip(_WEIGHT_NAMES[:2], got[0][:2], want[0][:2]):
        assert torch.equal(g, w), f'{name} differs bitwise'


@pytest.mark.gpu
@pytest.mark.parametrize('down_proj_backend', ['triton', 'cute'])
def test_weight_grad_accumulation_compiled(down_proj_backend):
    # torch.compile(fullgraph=True) traces the fused path without a graph break
    # and accumulates exactly what eager does.
    if down_proj_backend == 'cute':
        _require_cute()
    top_k = _ACCUM_CASE[2]
    batches, weights, plan = _micro_batches(2)
    key = grouped_moe_module.register_weight_grads(*weights)

    def layer(x, top_w, top_e, wg, wu, wd):
        out = grouped_moe(x * 2, top_w.flatten(), top_e.flatten().int(), plan,
                          top_k, wg, wu, wd, down_proj_backend,
                          weight_grad_key=key)
        return out + x

    compiled = torch.compile(layer, fullgraph=True, dynamic=False)
    results = []
    for fn in (layer, compiled):
        results.append(_accumulate(batches, weights, plan, down_proj_backend,
                                   fn=fn))
        for w in weights:
            w.grad = None
    (want_acts, want), (got_acts, got) = results
    for g, w in zip(got_acts, want_acts):
        assert all(torch.equal(a, b) for a, b in zip(g, w))
    for step in range(2):
        for name, g, w in zip(_WEIGHT_NAMES, got[step], want[step]):
            assert torch.equal(g, w), f'{name} differs bitwise (step {step})'


@pytest.mark.gpu
def test_weight_grad_key_custom_op_registration():
    top_k = 4
    x, top_w, top_e, w_gate, w_up, w_down, _, plan, _ = _inputs(
        [128, 256, 384, 512] * 4, 256, top_k, 33)
    key = grouped_moe_module.register_weight_grads(w_gate, w_up, w_down)
    torch.library.opcheck(
        grouped_moe_module._grouped_moe_fwd,
        (x, top_w.flatten(), top_e.flatten().int(), w_gate, w_up, w_down,
         None, plan.op_key, top_k, 'triton', True, 'ceil', False, key),
        test_utils=('test_schema', 'test_autograd_registration',
                    'test_faketensor'),
    )
