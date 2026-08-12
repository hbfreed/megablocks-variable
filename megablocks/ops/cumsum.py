# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch


def inclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Cumulative sum that keeps ``x``'s dtype, previously CUB.

    torch promotes integer cumsums to int64 by default; the explicit dtype
    matches the extension's ``empty_like(x)`` output so int32 bins stay int32
    for ``nanomoe_ops.build_topology``.
    """
    return torch.cumsum(x, dim, dtype=x.dtype)


def exclusive_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cumsum(x, dim, dtype=x.dtype) - x
