# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""Correctness of the grouped variable-width MoE forward AND backward.

Checked against a dense fp32 expert loop through torch autograd: the leaves
are the same bf16 tensors the kernels see, the reference math runs in fp32,
and every gradient (x, routing weights, all three expert weights) must agree
with the kernel gradients to bf16 tolerance.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

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


def _run(
    widths,
    hidden,
    top_k,
    num_tokens,
    seed=0,
    down_proj_backend='triton',
    recompute_activation=True,
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
    top_w = top_w0.to(torch.bfloat16).requires_grad_()

    dout = torch.randn(num_tokens, hidden, device=dev, dtype=torch.bfloat16)

    plan = FusedMoEPlan(widths, hidden, dev)
    got = grouped_moe(x, top_w.flatten(), top_e.flatten().int(), plan,
                      top_k, w_gate, w_up, w_down, down_proj_backend,
                      recompute_activation)
    got.backward(dout)
    grads = [t.grad for t in (x, top_w, w_gate, w_up, w_down)]
    for t in (x, top_w, w_gate, w_up, w_down):
        t.grad = None

    want = _reference(x, top_w, top_e, w_gate, w_up, w_down, widths)
    want.backward(dout.float())
    ref_grads = [t.grad for t in (x, top_w, w_gate, w_up, w_down)]

    errs = {'out': _rel(got, want)}
    for name, ga, gb in zip(('dx', 'dw_route', 'dwg', 'dwu', 'dwd'),
                            grads, ref_grads):
        errs[name] = _rel(ga, gb)
    return errs


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.parametrize('widths, hidden, top_k, num_tokens', _CASES)
def test_matches_dense_autograd_reference(widths, hidden, top_k, num_tokens):
    errs = _run(widths, hidden, top_k, num_tokens)
    assert errs['out'] < 5e-3, f'forward error {errs["out"]:.3e}'
    for name in ('dx', 'dw_route', 'dwg', 'dwu', 'dwd'):
        assert errs[name] < 2e-2, f'{name} error {errs[name]:.3e} ({errs})'


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.skipif(
    importlib.util.find_spec('cutlass') is None,
    reason='nvidia-cutlass-dsl is not installed',
)
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
    if torch.cuda.get_device_capability()[0] != 8:
        pytest.skip('the experimental CuTe backend currently targets Ampere')
    errs = _run(
        widths,
        hidden=hidden,
        top_k=top_k,
        num_tokens=num_tokens,
        down_proj_backend='cute',
        recompute_activation=recompute_activation,
    )
    assert errs['out'] < 5e-3, f'forward error {errs["out"]:.3e}'
    for name in ('dx', 'dw_route', 'dwg', 'dwu', 'dwd'):
        assert errs[name] < 2e-2, f'{name} error {errs[name]:.3e} ({errs})'
