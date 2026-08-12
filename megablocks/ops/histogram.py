# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


# Same shape as backend.fused_moe._count_experts: block-local tl.histogram,
# then one atomic add per bin. Integer atomics are associative, so the result
# is deterministic even though the add order is not. Out-of-range lanes land
# on a sentinel bin that the masked store discards.
@triton.jit
def _histogram(
    x,
    counts,
    num_elements,
    MAX_VAL: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(x + off, mask=off < num_elements, other=MAX_VAL)
    local = tl.histogram(v.to(tl.int32), BINS)
    b = tl.arange(0, BINS)
    tl.atomic_add(counts + b, local, mask=b < MAX_VAL)


def histogram(x: torch.Tensor, max_val: int) -> torch.Tensor:
    """Even-width histogram over ``[0, max_val)``, previously CUB.

    Returns int32 like the extension did: consumers feed the counts through
    ``inclusive_cumsum`` into ``nanomoe_ops.build_topology``, which requires
    int32 bins. Rows of a batched input are independent histograms, matching
    the extension's (rows, max_val) output.
    """
    rows = x.shape[0] if x.dim() == 2 else 1
    flat = x.reshape(-1)
    if x.dim() == 2:
        # Offset each row into its own bin range so one launch covers all
        # rows. Only tests use this form; the training path is 1-D.
        offsets = torch.arange(rows, device=x.device, dtype=torch.int32) * max_val
        flat = (flat.view(rows, -1).to(torch.int32) + offsets[:, None]).reshape(-1)
    counts = torch.zeros(rows * max_val, dtype=torch.int32, device=x.device)
    _histogram[(triton.cdiv(flat.numel(), 1024),)](
        flat,
        counts,
        flat.numel(),
        MAX_VAL=rows * max_val,
        BINS=_next_pow2(rows * max_val + 1),
        BLOCK=1024,
        num_warps=4,
    )
    return counts.view(rows, max_val) if x.dim() == 2 else counts
