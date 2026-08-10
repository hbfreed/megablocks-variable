# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gpt_model import GPTConfig, MoEMLP
from megablocks import ops


@pytest.mark.gpu
@pytest.mark.world_size(1)
def test_score_sorted_nearest_rounding_drops_lowest_route():
    moe = MoEMLP(
        GPTConfig(
            n_embd=128,
            expert_sizes=[(2, 128)],
            num_active_experts=1,
            token_rounding="nearest",
        )
    ).cuda()

    experts = torch.tensor([0] * 129 + [1] * 127, device="cuda")
    scores = torch.linspace(0.1, 0.9, experts.numel(), device="cuda")
    scores[3] = 0.0

    bin_ids, indices, tokens_per_expert = moe._sort_tokens_by_expert(
        experts, scores
    )
    block_tokens = moe._block_tokens_per_expert(tokens_per_expert, "nearest")
    bins = ops.inclusive_cumsum(tokens_per_expert, 0)
    padded_bins = ops.inclusive_cumsum(block_tokens, 0)
    route_mask = ops.padded_route_mask(indices, bin_ids, bins, padded_bins)

    assert tokens_per_expert.tolist() == [129, 127]
    assert block_tokens.tolist() == [128, 128]
    assert route_mask.sum().item() == 255
    assert not route_mask[3].item()

    sparse_counts = torch.tensor([1, 63, 64, 65], device="cuda", dtype=torch.int32)
    assert moe._block_tokens_per_expert(sparse_counts, "nearest").tolist() == [
        0,
        0,
        128,
        128,
    ]


@pytest.mark.gpu
@pytest.mark.world_size(1)
@pytest.mark.parametrize("top_k", [1, 2])
def test_dropped_routes_are_zero_in_forward_and_backward(top_k: int):
    num_routes = 260
    tokens = num_routes // top_k
    hidden = 16

    # Expert-major and score-sorted: 129 -> 128 and 131 -> 128.
    bin_ids = torch.tensor([0] * 129 + [1] * 131, device="cuda", dtype=torch.int32)
    indices = torch.arange(num_routes, device="cuda", dtype=torch.int32)
    bins = torch.tensor([129, 260], device="cuda", dtype=torch.int32)
    padded_bins = torch.tensor([128, 256], device="cuda", dtype=torch.int32)
    expected_mask = torch.ones(num_routes, device="cuda", dtype=torch.bool)
    expected_mask[[128, 257, 258, 259]] = False

    route_mask = ops.padded_route_mask(indices, bin_ids, bins, padded_bins)
    torch.testing.assert_close(route_mask, expected_mask)

    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.float16, requires_grad=True)
    weights = torch.randn(num_routes, device="cuda", dtype=torch.float16, requires_grad=True)
    gathered = ops.padded_gather(
        x, indices, bin_ids, bins, padded_bins, top_k, output_rows=256
    )
    out = ops.padded_scatter(
        gathered, indices, bin_ids, weights, bins, padded_bins, top_k
    )

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weights = weights.detach().clone().requires_grad_(True)
    route_scale = (
        ref_weights.view(tokens, top_k)
        * expected_mask.view(tokens, top_k).to(ref_weights.dtype)
    ).sum(dim=1)
    expected = ref_x * route_scale[:, None]

    torch.testing.assert_close(out, expected, rtol=5e-3, atol=5e-3)

    grad = torch.randn_like(out)
    out.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        weights.grad, ref_weights.grad, rtol=5e-3, atol=5e-3
    )
    assert torch.count_nonzero(weights.grad[~expected_mask]).item() == 0


@pytest.mark.gpu
@pytest.mark.world_size(1)
def test_nearest_rounding_runs_full_moe_forward_and_backward():
    """Exercise score sorting, rounded topology, and dropped-route gradients together."""
    torch.manual_seed(0)
    moe = MoEMLP(
        GPTConfig(
            n_embd=128,
            expert_sizes=[(2, 128)],
            num_active_experts=1,
            token_rounding="nearest",
        )
    ).cuda().train()
    with torch.no_grad():
        moe.router.weight[0].fill_(1)
        moe.router.weight[1].fill_(-1)

    # Every token chooses expert 0; nearest-block rounding keeps 128/129 routes.
    x = torch.ones(1, 129, 128, device="cuda", requires_grad=True)
    out, aux, _ = moe(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    torch.testing.assert_close(
        aux["dropped_route_fraction"],
        torch.tensor(1 / 129, device="cuda"),
    )
    assert torch.count_nonzero(out.abs().sum(dim=-1) == 0).item() == 1

    out.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert moe.w1.grad is not None and torch.isfinite(moe.w1.grad).all()
    assert moe.w2.grad is not None and torch.isfinite(moe.w2.grad).all()
