# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch

from megablocks.backend import kernels


def padded_route_mask(
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
):
    """Return which score-sorted routes fit in each expert's block capacity."""
    return kernels.padded_route_mask(indices, bin_ids, bins, padded_bins)
