# Copyright 2026 MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility fixes for STK kernels on modern Triton releases."""

import stk.ops
import torch
from stk.backend import sputnik
from stk.matrix import Matrix


def _sdd_with_wide_indices(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    topology: Matrix,
) -> Matrix:
    """Run STK SDD with index arithmetic wide enough for large matrices.

    STK stores row and column indices as int16. Triton 3.7 preserves that
    width when multiplying a loaded index by the 128-element block size. Row
    indices of 256 or greater therefore overflow before pointer arithmetic.
    Passing int32 views into the autograd operation avoids the overflow in
    SDD and in its DSD/DDS backward kernels without changing Matrix metadata.
    """
    row_indices = topology.row_indices.to(torch.int32)
    column_indices = topology.column_indices.to(torch.int32)
    column_indices_t = topology.column_indices_t.to(torch.int32)

    out = sputnik.sdd(
        lhs,
        rhs,
        topology.size(),
        topology.data,
        topology.offsets,
        row_indices,
        column_indices,
        topology.offsets_t,
        column_indices_t,
        topology.block_offsets_t,
    )
    return Matrix(
        topology.size(),
        out,
        topology.row_indices,
        topology.column_indices,
        topology.offsets,
        topology.column_indices_t,
        topology.offsets_t,
        topology.block_offsets_t,
    )


def _dsd_with_wide_indices(lhs: Matrix, rhs: torch.Tensor) -> torch.Tensor:
    """Run STK DSD with int32 sparse metadata."""
    return sputnik.dsd(
        lhs.size(),
        lhs.data,
        lhs.offsets,
        lhs.row_indices.to(torch.int32),
        lhs.column_indices.to(torch.int32),
        lhs.offsets_t,
        lhs.column_indices_t.to(torch.int32),
        lhs.block_offsets_t,
        not lhs.is_contiguous(),
        rhs,
    )


def _dds_with_wide_indices(lhs: torch.Tensor, rhs: Matrix) -> torch.Tensor:
    """Run STK DDS with int32 sparse metadata."""
    return sputnik.dds(
        lhs,
        rhs.size(),
        rhs.data,
        rhs.offsets,
        rhs.row_indices.to(torch.int32),
        rhs.column_indices.to(torch.int32),
        rhs.offsets_t,
        rhs.column_indices_t.to(torch.int32),
        rhs.block_offsets_t,
        not rhs.is_contiguous(),
    )


def apply_stk_compatibility_fixes() -> None:
    """Install MegaBlocks' STK compatibility wrappers once per process."""
    if stk.ops.sdd is not _sdd_with_wide_indices:
        stk.ops.sdd = _sdd_with_wide_indices
        stk.ops.dsd = _dsd_with_wide_indices
        stk.ops.dds = _dds_with_wide_indices
