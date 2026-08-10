# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import torch

from megablocks.backend import kernels


def pack_route_keys(
    experts: torch.Tensor,
    scores: torch.Tensor,
    score_bits: int,
):
    """Pack expert-major, descending-score radix keys in one GPU launch."""
    return kernels.pack_route_keys(experts, scores, score_bits)
