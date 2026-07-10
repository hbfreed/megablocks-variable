# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f'Expected {ndim}-tensor but got {x.ndim}-tensor')


def assert_is_matrix(x):
    assert_is_tensor(x, 2)


def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f'Expected 1-tensor but got {x.ndim}-tensor')


def assert_equal(a, b):
    if a != b:
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.',)


# Autotune search space shared by the copy kernels. These kernels are purely
# memory-bound (one row copied/reduced per program), so the goal is to move each
# row in as few vectorized iterations as possible while keeping enough warps to
# hide memory latency.
_COPY_CONFIGS = [
    triton.Config({'BLOCK_X': 64}, num_warps=2),
    triton.Config({'BLOCK_X': 128}, num_warps=2),
    triton.Config({'BLOCK_X': 256}, num_warps=2),
    triton.Config({'BLOCK_X': 128}, num_warps=4),
    triton.Config({'BLOCK_X': 256}, num_warps=4),
    triton.Config({'BLOCK_X': 512}, num_warps=4),
    triton.Config({'BLOCK_X': 1024}, num_warps=8),
]


# a: (padded_rows, hidden), the destination of padded_gather.
# bins:        (num_experts,) inclusive cumsum of *real* tokens per expert.
# padded_bins: (num_experts,) inclusive cumsum of *padded* tokens per expert.
#
# For each expert the copy writes its real tokens into the first rows of that
# expert's padded region; the trailing rows (up to the padded bin boundary) are
# padding that must be zero. Rather than zero the whole output and immediately
# overwrite ~94% of it, we zero only these per-expert padding gaps.
@triton.jit
def _zero_padding(
    out,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    expert = tl.program_id(0)

    # Real-token and padded row bounds for this expert.
    prev_real = tl.load(bins + expert - 1, mask=expert > 0, other=0)
    real_end = tl.load(bins + expert)
    prev_pad = tl.load(padded_bins + expert - 1, mask=expert > 0, other=0)
    pad_end = tl.load(padded_bins + expert)

    # The real tokens occupy [prev_pad, prev_pad + num_real); zero the rest.
    num_real = real_end - prev_real
    start = (prev_pad + num_real) * NUM_COLUMNS
    end = pad_end * NUM_COLUMNS

    col = tl.arange(0, BLOCK_X)
    zeros = tl.zeros((BLOCK_X,), dtype=out.dtype.element_ty)
    for base in range(start, end, BLOCK_X):
        offsets = base + col
        tl.store(out + offsets, zeros, mask=offsets < end)


# a: (tokens, hidden_size), real.
# indices: (tokens * top_k), integer.
# bin_ids: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.
# padded_bins: (num_experts), integer.
@triton.autotune(
    configs=_COPY_CONFIGS,
    key=['NUM_COLUMNS'],
)
@triton.jit
def _padded_copy(
    a,
    b,
    indices,
    bin_ids,
    weights,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Our index into array 'a'.
    index_a = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Now we know what bin we're assigned to, but we need to know how
    # many threadblocks were assigned to earlier bins so we can offset
    # in our bin properly.
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Load the starting index of our bin in array 'b'.
    index_b = offset_in_bin
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a).to(tl.float32) if SCALE else 1

    # Swap the pointers depending on the direction.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)

        # When not scaling this is a pure copy: skipping the fp32 round-trip is
        # bit-identical (bf16 -> fp32 -> bf16 is the identity) and avoids the
        # conversion instructions on this memory-bound path.
        if SCALE:
            x = (x.to(tl.float32) * scale).to(optr.dtype.element_ty)

        tl.store(optr + offsets, x, mask=mask)

        offsets += BLOCK_X


def padded_gather(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    assert_equal(bin_ids.shape[0], x.shape[0] * top_k)
    assert_equal(bins.size(), padded_bins.size())

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    # NOTE: Because of the padding, the output size is dynamic.
    # We load the final padded bin bound to get the output rows.
    output_rows = padded_bins[-1].cpu().item()
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)

    # The copy below writes every real token row, so we only need to zero the
    # per-expert padding gaps rather than the whole (much larger) output.
    _zero_padding[(bins.shape[0],)](
        out,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        BLOCK_X=1024,
        num_warps=4,
    )
    _padded_copy[(indices.shape[0],)](
        x,
        out,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


def gather(x, indices, bin_ids, weights, bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    assert_equal(bin_ids.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    # NOTE: There is no padding so the output rows equals the
    # input rows multiplied by top_k.
    output_rows = x.shape[0] * top_k
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](
        x,
        out,
        indices,
        bin_ids,
        weights,
        bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


# One threadblock per output token. Each accumulates that token's TOP_K scattered
# rows directly in fp32 registers, so we never materialize the (tokens, top_k,
# hidden) intermediate that a separate .sum(dim=1) would then have to re-read.
# inv is the inverse of `indices`: inv[dest] = sorted position holding it.
@triton.autotune(
    configs=_COPY_CONFIGS,
    key=['NUM_COLUMNS'],
)
@triton.jit
def _scatter_reduce(
    out,
    x,
    inv,
    bin_ids,
    weights,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    SCALE: tl.constexpr,
):
    token = tl.program_id(0)
    out_ptr = out + token * NUM_COLUMNS
    for col_base in range(0, NUM_COLUMNS, BLOCK_X):
        cols = col_base + tl.arange(0, BLOCK_X)
        mask = cols < NUM_COLUMNS
        acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
        for k in range(TOP_K):
            dest = token * TOP_K + k
            pid = tl.load(inv + dest)
            b = tl.load(bin_ids + pid)
            prev_bin = tl.load(bins + b - 1, mask=b > 0, other=0)
            prev_pad = tl.load(padded_bins + b - 1, mask=b > 0, other=0)
            src_row = (pid - prev_bin) + prev_pad
            v = tl.load(x + src_row * NUM_COLUMNS + cols, mask=mask).to(tl.float32)
            if SCALE:
                v = v * tl.load(weights + dest).to(tl.float32)
            acc += v
        tl.store(out_ptr + cols, acc.to(out.dtype.element_ty), mask=mask)


def padded_scatter(x, indices, bin_ids, weights, bins, padded_bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.size(), padded_bins.size())

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    tokens = indices.shape[0] // top_k

    if top_k > 1:
        # Fused scatter + top-k reduction (see _scatter_reduce). This both scales
        # (when weights are given) and sums the top_k contributions per token,
        # accumulating in fp32, so it replaces the copy + .sum(dim=1) and avoids
        # the 8x (top_k) intermediate. Accumulating in fp32 is also slightly more
        # accurate than summing the bf16 intermediate the old path produced.
        inv = torch.empty_like(indices)
        inv[indices.long()] = torch.arange(
            indices.shape[0], device=indices.device, dtype=indices.dtype
        )
        out = torch.empty((tokens, x.shape[1]), dtype=x.dtype, device=x.device)
        _scatter_reduce[(tokens,)](
            out,
            x,
            inv,
            bin_ids,
            weights,
            bins,
            padded_bins,
            NUM_COLUMNS=x.shape[1],
            TOP_K=top_k,
            SCALE=weights is not None,
        )
        return out

    # top_k == 1: no reduction, so the direct copy is fastest.
    out = torch.empty((tokens, 1, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](
        out,
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out.view(tokens, x.shape[1])


def scatter(x, indices, bin_ids, weights, bins, top_k):
    return padded_scatter(x, indices, bin_ids, weights, bins, bins, top_k)


# x: (tokens, top_k, hidden_size), real
# grad: (tokens, hidden_size), real.
# wgrad: (tokens, top_k), real.
# indices: (tokens * top_k), integer.
# bin_ids: (tokens * top_k), integer.
# bins: (num_experts), integer.
# padded_bins: (num_experts), integer.
@triton.autotune(
    configs=_COPY_CONFIGS,
    key=['NUM_COLUMNS'],
)
@triton.jit
def _padded_copy_wgrad(
    x,
    grad,
    wgrad,
    indices,
    bin_ids,
    bins,
    padded_bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Our index into 'tokens * top_k'.
    index_out = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Now we know what bin we're assigned to, but we need to know how
    # many threadblocks were assigned to earlier bins so we can offset
    # in our bin properly.
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Load the starting index of our bin in array 'x'.
    index_x = offset_in_bin
    if bin_idx > 0:
        index_x += tl.load(padded_bins + bin_idx - 1)

    # Offset the input and output pointers.
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X

    # Reduce to get the final result and store.
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


def padded_scatter_wgrad(x, grad, indices, bin_ids, bins, padded_bins, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.size(), padded_bins.size())

    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens * top_k), dtype=x.dtype, device=x.device)
    _padded_copy_wgrad[(indices.shape[0],)](
        x,
        grad,
        out,
        indices,
        bin_ids,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
    )
    return out


def scatter_wgrad(x, grad, indices, bin_ids, bins, top_k):
    return padded_scatter_wgrad(x, grad, indices, bin_ids, bins, bins, top_k)


# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.
@triton.autotune(
    configs=_COPY_CONFIGS,
    key=['NUM_COLUMNS'],
)
@triton.jit
def _binned_copy(
    a,
    b,
    num_experts,
    expert_capacity,
    indices,
    weights,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_b = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_a = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a).to(tl.float32) if SCALE else 1

    # Swap the pointers depending on the direction.
    #
    # NOTE: We need to zero the output in both directions.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)

        # See _padded_copy: skip the fp32 round-trip on the pure-copy path.
        if SCALE:
            x = (x.to(tl.float32) * scale).to(optr.dtype.element_ty)

        tl.store(optr + offsets, x, mask=mask)

        offsets += BLOCK_X


def binned_gather(x, indices, weights, bins, expert_capacity, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    num_experts = bins.shape[0]
    out = torch.zeros((num_experts, expert_capacity, x.shape[1]), dtype=x.dtype, device=x.device)

    _binned_copy[(num_experts, expert_capacity)](
        x,
        out,
        num_experts,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


def binned_scatter(x, indices, weights, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, top_k, hidden_size), dtype=x.dtype, device=x.device)
    _binned_copy[(num_experts, expert_capacity)](
        out,
        x,
        num_experts,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=hidden_size,
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None,
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, hidden_size)


# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.
@triton.autotune(
    configs=_COPY_CONFIGS,
    key=['NUM_COLUMNS'],
)
@triton.jit
def _binned_copy_wgrad(
    x,
    grad,
    wgrad,
    num_experts,
    expert_capacity,
    indices,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_x = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_out = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X

    # Reduce to get the final result and store.
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


def binned_scatter_wgrad(x, grad, indices, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens * top_k), dtype=x.dtype, device=x.device)
    _binned_copy_wgrad[(num_experts, expert_capacity)](
        x,
        grad,
        out,
        num_experts,
        expert_capacity,
        indices,
        bins,
        NUM_COLUMNS=hidden_size,
        TOP_K=top_k,
    )
    return out
