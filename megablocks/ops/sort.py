# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch


def sort(x: torch.Tensor, end_bit: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Key/value sort, previously CUB's DeviceRadixSort.

    ``end_bit`` is accepted for API compatibility but unused: torch's CUDA
    integer sort is already a radix sort, and the bit-range hint was only a
    CUB-specific optimization. ``stable=True`` matches CUB's stability so
    routes that share an expert keep their token order, and the returned
    indices use ``x``'s dtype exactly as the extension's iota output did.
    """
    del end_bit
    sorted_x, indices = torch.sort(x, stable=True)
    return sorted_x, indices.to(x.dtype)
