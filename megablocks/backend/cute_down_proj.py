# Copyright 2026 MegaBlocks authors
# SPDX-License-Identifier: Apache-2.0
#
# The Ampere MMA pipeline and layout helpers below are adapted from NVIDIA
# CUTLASS's CuTe DSL tensorop_gemm example (v4.6.0), licensed under BSD-3-Clause:
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Experimental CuTe DSL GEMMs for the ragged MoE training backend.

This is deliberately a narrow backend spike, not part of the default import
path.  It consumes exactly the buffers produced by ``FusedMoEPlan`` and the
Triton route planner. The shared Ampere pipeline implements three operations:

* the forward down projection;
* the fused gate+up input-gradient streams; and
* the down-projection weight gradient.

One CuTe launch spans every expert tile, retains device-side expert selection,
and never reads routing metadata back to the host.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "The CuTe grouped-GEMM experiment needs nvidia-cutlass-dsl>=4.6 "
        "and apache-tvm-ffi. The normal Triton backend does not need them.",
    ) from exc


class _AmpereRaggedDown:
    """SM80/SM86 BF16 tensor-core GEMMs with runtime expert dimensions."""

    def __init__(
        self,
        cta_tiler=(128, 256, 32),
        num_stages=3,
        atom_layout_mnk=(2, 4, 1),
        weight_mode="down",
    ):
        if weight_mode not in ("down", "dx", "dx_fused", "wgrad"):
            raise ValueError(f"unsupported ragged GEMM mode: {weight_mode!r}")
        self.cta_tiler = cta_tiler
        self.num_stages = num_stages
        self.atom_layout_mnk = atom_layout_mnk
        self.weight_mode = weight_mode
        self.num_threads = math.prod(atom_layout_mnk) * 32
        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 16)

    @cute.jit
    def __call__(
        self,
        h: cute.Tensor,
        w_down: cute.Tensor,
        h2: cute.Tensor,
        w2: cute.Tensor,
        y: cute.Tensor,
        tile_expert: cute.Tensor,
        expert_col_off: cute.Tensor,
        expert_nblocks: cute.Tensor,
        expert_row_start: cute.Tensor,
        width_block: cutlass.Constexpr,
    ):
        # Down weights are physically [total_width, hidden], while gate/up
        # weights are [hidden, total_width]. The down projection needs a
        # transposed [hidden, total_width] view; dx consumes its
        # physical row-major layouts directly.
        if cutlass.const_expr(self.weight_mode == "wgrad"):
            input_matrix = cute.make_tensor(
                h.iterator,
                cute.select(h.layout, mode=[1, 0]),
            )
            weight_matrix = cute.make_tensor(
                w_down.iterator,
                cute.select(w_down.layout, mode=[1, 0]),
            )
        elif cutlass.const_expr(self.weight_mode == "down"):
            input_matrix = h
            weight_matrix = cute.make_tensor(
                w_down.iterator,
                cute.select(w_down.layout, mode=[1, 0]),
            )
        else:
            input_matrix = h
            weight_matrix = w_down
        input_matrix2 = h2
        weight_matrix2 = w2

        a_major = (
            utils.LayoutEnum.COL_MAJOR
            if self.weight_mode == "wgrad"
            else utils.LayoutEnum.ROW_MAJOR
        )
        b_major = (
            utils.LayoutEnum.COL_MAJOR
            if self.weight_mode in ("down", "wgrad")
            else utils.LayoutEnum.ROW_MAJOR
        )
        c_major = utils.LayoutEnum.ROW_MAJOR
        copy_bits = 128

        sA_layout = self._make_smem_layout_ab(
            input_matrix.element_type,
            a_major,
            copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_ab(
            w_down.element_type,
            b_major,
            copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = self._make_smem_layout_c(
            y.element_type,
            c_major,
            copy_bits,
            (self.bM, self.bN),
        )

        async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            ),
            input_matrix.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_ab(
            async_copy, h.element_type, a_major, copy_bits, self.bM,
        )
        tiled_copy_B = self._make_gmem_tiled_copy_ab(
            async_copy, w_down.element_type, b_major, copy_bits, self.bN,
        )

        sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            y.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_c(
            sync_copy, y.element_type, c_major, copy_bits,
        )

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            h.element_type,
            cutlass.Float32,
            self.mma_inst_shape,
        )
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=permutation_mnk,
        )

        grid = (
            (cute.ceil_div(y.shape[0], self.bM)
             if self.weight_mode == "wgrad"
             else cute.ceil_div(h.shape[0], self.bM)),
            cute.ceil_div(y.shape[1], self.bN),
            1,
        )
        self.kernel(
            input_matrix,
            weight_matrix,
            input_matrix2,
            weight_matrix2,
            y,
            tile_expert,
            expert_col_off,
            expert_nblocks,
            expert_row_start,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
            width_block,
        ).launch(grid=grid, block=[self.num_threads, 1, 1])

    @cute.kernel
    def kernel(
        self,
        h: cute.Tensor,
        weight_matrix: cute.Tensor,
        h2: cute.Tensor,
        weight_matrix2: cute.Tensor,
        y: cute.Tensor,
        tile_expert: cute.Tensor,
        expert_col_off: cute.Tensor,
        expert_nblocks: cute.Tensor,
        expert_row_start: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        width_block: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        mt, nt, _ = cute.arch.block_idx()
        expert = tile_expert[mt]

        if expert < 0:
            pass
        else:
            col_off = expert_col_off[expert]
            if cutlass.const_expr(self.weight_mode == "wgrad"):
                count = expert_nblocks[expert]
                width = cute.ceil_div(count, width_block) * width_block
                row_start = expert_row_start[expert]
                local_col = mt * self.bM - col_off
                expert_h = cute.domain_offset((local_col, row_start), h)
                expert_w = cute.domain_offset((0, row_start), weight_matrix)
                a_tile = 0
            else:
                width = expert_nblocks[expert] * width_block
                # A and C use the planner's global 128-row tile numbering. B
                # shifts its K mode to select this expert without host-side
                # pointer construction.
                expert_h = h
                expert_w = cute.domain_offset((0, col_off), weight_matrix)
                a_tile = mt
                if cutlass.const_expr(self.weight_mode == "dx_fused"):
                    expert_h2 = h2
                    expert_w2 = cute.domain_offset(
                        (0, col_off), weight_matrix2,
                    )
            gA = cute.local_tile(
                expert_h,
                tiler=(self.bM, self.bK),
                coord=(a_tile, None),
            )
            gB = cute.local_tile(
                expert_w,
                tiler=(self.bN, self.bK),
                coord=(nt, None),
            )
            if cutlass.const_expr(self.weight_mode == "dx_fused"):
                gA2 = cute.local_tile(
                    expert_h2,
                    tiler=(self.bM, self.bK),
                    coord=(a_tile, None),
                )
                gB2 = cute.local_tile(
                    expert_w2,
                    tiler=(self.bN, self.bK),
                    coord=(nt, None),
                )
            gC = cute.local_tile(
                y,
                tiler=(self.bM, self.bN),
                coord=(mt, nt),
            )
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
            if cutlass.const_expr(self.weight_mode == "dx_fused"):
                gA2 = cute.make_tensor(gA2.iterator.align(16), gA2.layout)
                gB2 = cute.make_tensor(gB2.iterator.align(16), gB2.layout)

            @cute.struct
            class SharedStorageAB:
                a: cute.struct.Align[
                    cute.struct.MemRange[h.element_type, cute.cosize(sA_layout)],
                    16,
                ]
                b: cute.struct.Align[
                    cute.struct.MemRange[
                        weight_matrix.element_type, cute.cosize(sB_layout),
                    ],
                    16,
                ]

            @cute.struct
            class SharedStorageC:
                c: cute.struct.Align[
                    cute.struct.MemRange[y.element_type, cute.cosize(sC_layout)],
                    16,
                ]

            smem = utils.SmemAllocator()
            storage = smem.allocate(
                max(
                    SharedStorageAB.size_in_bytes(),
                    SharedStorageC.size_in_bytes(),
                ),
                byte_alignment=16,
            )
            sA = SharedStorageAB(storage).a.get_tensor(sA_layout)
            sB = SharedStorageAB(storage).b.get_tensor(sB_layout)
            sC = SharedStorageC(storage).c.get_tensor(sC_layout)

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            if cutlass.const_expr(self.weight_mode == "dx_fused"):
                tAgA2 = thr_copy_A.partition_S(gA2)
                tBgB2 = thr_copy_B.partition_S(gB2)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCsC = thr_mma.partition_C(sC)
            tCgC = thr_mma.partition_C(gC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.weight_mode == "wgrad", 4,
                ),
                h.element_type,
            )
            copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.weight_mode in ("down", "wgrad"), 4,
                ),
                weight_matrix.element_type,
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(copy_s2r_B, tiled_mma)
            thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy = thr_s2r_A.partition_S(sA)
            tCrA_copy = thr_s2r_A.retile(tCrA)
            tCsB_copy = thr_s2r_B.partition_S(sB)
            tCrB_copy = thr_s2r_B.retile(tCrB)

            self._accumulate_stream(
                tAgA,
                tBgB,
                tAsA,
                tBsB,
                tCsA_copy,
                tCsB_copy,
                tCrA_copy,
                tCrB_copy,
                tCrA,
                tCrB,
                tCrC,
                tiled_copy_A,
                tiled_copy_B,
                tiled_copy_s2r_A,
                tiled_copy_s2r_B,
                tiled_mma,
                width,
            )
            if cutlass.const_expr(self.weight_mode == "dx_fused"):
                self._accumulate_stream(
                    tAgA2,
                    tBgB2,
                    tAsA,
                    tBsB,
                    tCsA_copy,
                    tCsB_copy,
                    tCrA_copy,
                    tCrB_copy,
                    tCrA,
                    tCrB,
                    tCrC,
                    tiled_copy_A,
                    tiled_copy_B,
                    tiled_copy_s2r_A,
                    tiled_copy_s2r_B,
                    tiled_mma,
                    width,
                )

            tCrD = cute.make_fragment_like(tCrC, y.element_type)
            tCrD[None] = tCrC.load().to(y.element_type)
            cute.autovec_copy(tCrD, tCsC)
            cute.arch.sync_threads()

            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)
            cute.copy(tiled_copy_C, tCrC_epilogue, tCgC_epilogue)
        return

    @cute.jit
    def _accumulate_stream(
        self,
        tAgA,
        tBgB,
        tAsA,
        tBsB,
        tCsA_copy,
        tCsB_copy,
        tCrA_copy,
        tCrB_copy,
        tCrA,
        tCrB,
        tCrC,
        tiled_copy_A,
        tiled_copy_B,
        tiled_copy_s2r_A,
        tiled_copy_s2r_B,
        tiled_mma,
        width,
    ):
        # Reinitialize the shared pipeline for each operand pair while leaving
        # the fp32 accumulator live in registers. This gives dx = dg@Wg.T +
        # du@Wu.T without a second accumulator or an intermediate bf16 sum.
        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()

        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = width // self.bK
        k_tile_index = cutlass.Int32(0)
        for stage in range(num_smem_stages - 1):
            if stage < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, stage],
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, k_tile_index],
                    tBsB[None, None, None, stage],
                )
                cute.arch.cp_async_commit_group()
            k_tile_index = k_tile_index + 1

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1
        tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
        tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
        num_k_block = cute.size(tCrA, mode=[2])

        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(
                tiled_copy_s2r_A,
                tCsA_p[None, None, 0],
                tCrA_copy[None, None, 0],
            )
            cute.copy(
                tiled_copy_s2r_B,
                tCsB_p[None, None, 0],
                tCrB_copy[None, None, 0],
            )

        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                k_block_next = (k_block + 1) % num_k_block
                cute.copy(
                    tiled_copy_s2r_A,
                    tCsA_p[None, None, k_block_next],
                    tCrA_copy[None, None, k_block_next],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsB_p[None, None, k_block_next],
                    tCrB_copy[None, None, k_block_next],
                )

                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_A,
                            tAgA[None, None, None, k_tile_index],
                            tAsA[None, None, None, smem_pipe_write],
                        )
                        cute.copy(
                            tiled_copy_B,
                            tBgB[None, None, None, k_tile_index],
                            tBsB[None, None, None, smem_pipe_write],
                        )
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0

                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    def _make_smem_layout_ab(self, dtype, major_mode, copy_bits, shape):
        major_size = shape[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else shape[0]
        major_size = min(64, major_size)
        swizzle_bits = min(
            int(math.log2(major_size * dtype.width // copy_bits)),
            3,
        )
        outer = (
            cute.make_layout((8, major_size), stride=(major_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_size, 8), stride=(1, major_size))
        )
        atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            outer,
        )
        return cute.tile_to_shape(atom, shape, (0, 1, 2))

    def _make_smem_layout_c(self, dtype, major_mode, copy_bits, shape):
        major_size = shape[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else shape[0]
        swizzle_bits = min(
            int(math.log2(major_size * dtype.width // copy_bits)),
            3,
        )
        outer = (
            cute.make_layout((8, major_size), stride=(major_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_size, 8), stride=(1, major_size))
        )
        atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4),
            0,
            outer,
        )
        return cute.tile_to_shape(atom, shape, (0, 1))

    def _make_gmem_tiled_copy_ab(
        self,
        atom,
        dtype,
        major_mode,
        copy_bits,
        mn_extent,
    ):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = self.bK // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1),
            stride=(shape_dim_1, 1),
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = mn_extent // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0),
                stride=(1, shape_dim_0),
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom, thread_layout, value_layout)

    def _make_gmem_tiled_copy_c(self, atom, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = self.bN // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1),
            stride=(shape_dim_1, 1),
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(atom, thread_layout, value_layout)


@dataclass
class _CompiledKernel:
    fn: object
    signature: tuple


_CACHE: dict[tuple, _CompiledKernel] = {}
_DX_CACHE: dict[tuple, _CompiledKernel] = {}
_DX_FUSED_CACHE: dict[tuple, _CompiledKernel] = {}
_WGRAD_CACHE: dict[tuple, _CompiledKernel] = {}


def _as_cute(
    tensor: torch.Tensor,
    dynamic_dim0: bool = False,
    dim0_divisibility: int = 1,
):
    result = cute.runtime.from_dlpack(
        tensor,
        assumed_align=16,
        use_32bit_stride=True,
        enable_tvm_ffi=True,
    )
    if dynamic_dim0:
        result = result.mark_compact_shape_dynamic(
            0,
            stride_order=tensor.dim_order(),
            divisibility=dim0_divisibility,
        )
    return result


def _compile(
    h: torch.Tensor,
    w_down: torch.Tensor,
    y: torch.Tensor,
    tile_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    expert_nblocks: torch.Tensor,
    width_block: int,
) -> _CompiledKernel:
    signature = (
        h.device.index,
        h.dtype,
        h.shape[1],
        tuple(w_down.shape),
        y.shape[1],
        width_block,
    )
    cached = _CACHE.get(signature)
    if cached is not None:
        return cached

    args = (
        _as_cute(h, dynamic_dim0=True, dim0_divisibility=128),
        _as_cute(w_down),
        _as_cute(h, dynamic_dim0=True, dim0_divisibility=128),
        _as_cute(w_down),
        _as_cute(y, dynamic_dim0=True, dim0_divisibility=128),
        _as_cute(tile_expert, dynamic_dim0=True),
        _as_cute(expert_col_off, dynamic_dim0=True),
        _as_cute(expert_nblocks, dynamic_dim0=True),
        _as_cute(expert_col_off, dynamic_dim0=True),
    )
    op = _AmpereRaggedDown(atom_layout_mnk=(2, 4, 1))
    compiled = cute.compile(op, *args, width_block, options="--enable-tvm-ffi")
    result = _CompiledKernel(compiled, signature)
    _CACHE[signature] = result
    return result


def cute_down_proj(
    h: torch.Tensor,
    w_down: torch.Tensor,
    y: torch.Tensor,
    tile_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    expert_nblocks: torch.Tensor,
    width_block: int,
) -> None:
    """Launch the experimental CuTe down projection on the current stream."""
    tensors = (h, w_down, y, tile_expert, expert_col_off, expert_nblocks)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all CuTe down-projection tensors must be on CUDA")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("all CuTe down-projection tensors must share a device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("CuTe down-projection tensors must be contiguous")
    if torch.cuda.get_device_capability(h.device)[0] != 8:
        raise ValueError("CuTe Ampere down projection requires compute capability 8.x")
    if h.dtype != torch.bfloat16 or w_down.dtype != h.dtype or y.dtype != h.dtype:
        raise TypeError("CuTe Ampere down projection currently supports BF16 only")
    if h.ndim != 2 or w_down.ndim != 2 or y.ndim != 2:
        raise ValueError("h, w_down, and y must be matrices")
    if h.shape[0] != y.shape[0] or tile_expert.numel() * 128 != h.shape[0]:
        raise ValueError("tile_expert must contain one entry per 128 rows")
    if y.shape[1] % 256 or w_down.shape[1] != y.shape[1]:
        raise ValueError("hidden size must be a multiple of 256 and match w_down")
    if expert_col_off.shape != expert_nblocks.shape:
        raise ValueError("expert metadata arrays must have the same shape")
    if any(t.dtype != torch.int32 for t in tensors[3:]):
        raise TypeError("expert metadata tensors must be int32")
    if width_block % 32:
        raise ValueError("width_block must be a multiple of the 32-column K tile")

    compiled = _compile(
        h,
        w_down,
        y,
        tile_expert,
        expert_col_off,
        expert_nblocks,
        width_block,
    )
    compiled.fn(
        h,
        w_down,
        h,
        w_down,
        y,
        tile_expert,
        expert_col_off,
        expert_nblocks,
        expert_col_off,
    )


def cute_dx_proj(
    d_rows: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    tile_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    expert_nblocks: torch.Tensor,
    width_block: int,
) -> None:
    """One gate/up input-gradient stream using packed [hidden, width] weights."""
    tensors = (
        d_rows,
        weight,
        out,
        tile_expert,
        expert_col_off,
        expert_nblocks,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("CuTe input-gradient tensors must be contiguous CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("CuTe input-gradient tensors must share a device")
    if torch.cuda.get_device_capability(d_rows.device)[0] != 8:
        raise ValueError("CuTe Ampere input gradient requires compute capability 8.x")
    if any(t.dtype != torch.bfloat16 for t in tensors[:3]):
        raise TypeError("CuTe Ampere input gradient currently supports BF16 only")
    if any(t.dtype != torch.int32 for t in tensors[3:]):
        raise TypeError("expert metadata tensors must be int32")
    if d_rows.ndim != 2 or weight.ndim != 2 or out.ndim != 2:
        raise ValueError("d_rows, weight, and out must be matrices")
    if d_rows.shape[0] != out.shape[0] or tile_expert.numel() * 128 != out.shape[0]:
        raise ValueError("tile_expert must contain one entry per 128 rows")
    if weight.shape[0] != out.shape[1] or out.shape[1] % 256:
        raise ValueError("weight hidden dimension must match a 256-aligned output")
    if expert_col_off.shape != expert_nblocks.shape:
        raise ValueError("expert metadata arrays must have the same shape")
    if width_block % 32:
        raise ValueError("width_block must be a multiple of the 32-column K tile")

    signature = (
        d_rows.device.index,
        d_rows.dtype,
        d_rows.shape[1],
        tuple(weight.shape),
        out.shape[1],
        width_block,
    )
    compiled = _DX_CACHE.get(signature)
    if compiled is None:
        args = (
            _as_cute(d_rows, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(weight),
            _as_cute(d_rows, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(weight),
            _as_cute(out, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(tile_expert, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
            _as_cute(expert_nblocks, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
        )
        op = _AmpereRaggedDown(atom_layout_mnk=(2, 4, 1), weight_mode="dx")
        fn = cute.compile(op, *args, width_block, options="--enable-tvm-ffi")
        compiled = _CompiledKernel(fn, signature)
        _DX_CACHE[signature] = compiled

    compiled.fn(
        d_rows,
        weight,
        d_rows,
        weight,
        out,
        tile_expert,
        expert_col_off,
        expert_nblocks,
        expert_col_off,
    )


def cute_dx_proj_fused(
    dg: torch.Tensor,
    du: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    out: torch.Tensor,
    tile_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    expert_nblocks: torch.Tensor,
    width_block: int,
) -> None:
    """Compute ``dg @ w_gate.T + du @ w_up.T`` with one accumulator."""
    tensors = (
        dg,
        du,
        w_gate,
        w_up,
        out,
        tile_expert,
        expert_col_off,
        expert_nblocks,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("fused CuTe input-gradient tensors must be contiguous CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("fused CuTe input-gradient tensors must share a device")
    if torch.cuda.get_device_capability(dg.device)[0] != 8:
        raise ValueError("fused CuTe input gradient requires compute capability 8.x")
    if any(t.dtype != torch.bfloat16 for t in tensors[:5]):
        raise TypeError("fused CuTe input gradient currently supports BF16 only")
    if any(t.dtype != torch.int32 for t in tensors[5:]):
        raise TypeError("expert metadata tensors must be int32")
    if any(t.ndim != 2 for t in tensors[:5]):
        raise ValueError("gradient, weight, and output tensors must be matrices")
    if dg.shape != du.shape or w_gate.shape != w_up.shape:
        raise ValueError("gate/up gradient and weight pairs must have matching shapes")
    if dg.shape[0] != out.shape[0] or tile_expert.numel() * 128 != out.shape[0]:
        raise ValueError("tile_expert must contain one entry per 128 rows")
    if w_gate.shape[0] != out.shape[1] or out.shape[1] % 256:
        raise ValueError("weight hidden dimension must match a 256-aligned output")
    if expert_col_off.shape != expert_nblocks.shape:
        raise ValueError("expert metadata arrays must have the same shape")
    if width_block % 32:
        raise ValueError("width_block must be a multiple of the 32-column K tile")

    signature = (
        dg.device.index,
        dg.dtype,
        dg.shape[1],
        tuple(w_gate.shape),
        out.shape[1],
        width_block,
    )
    compiled = _DX_FUSED_CACHE.get(signature)
    if compiled is None:
        args = (
            _as_cute(dg, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(w_gate),
            _as_cute(du, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(w_up),
            _as_cute(out, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(tile_expert, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
            _as_cute(expert_nblocks, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
        )
        op = _AmpereRaggedDown(
            atom_layout_mnk=(2, 4, 1),
            weight_mode="dx_fused",
        )
        fn = cute.compile(op, *args, width_block, options="--enable-tvm-ffi")
        compiled = _CompiledKernel(fn, signature)
        _DX_FUSED_CACHE[signature] = compiled

    compiled.fn(
        dg,
        w_gate,
        du,
        w_up,
        out,
        tile_expert,
        expert_col_off,
        expert_nblocks,
        expert_col_off,
    )


def cute_wgrad_rows(
    a_rows: torch.Tensor,
    b_rows: torch.Tensor,
    out: torch.Tensor,
    weight_block_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    counts: torch.Tensor,
    expert_row_start: torch.Tensor,
    row_block: int,
) -> None:
    """Compute packed ``a_rows.T @ b_rows`` expert weight gradients."""
    tensors = (
        a_rows,
        b_rows,
        out,
        weight_block_expert,
        expert_col_off,
        counts,
        expert_row_start,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("CuTe weight-gradient tensors must be contiguous CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("CuTe weight-gradient tensors must share a device")
    if torch.cuda.get_device_capability(a_rows.device)[0] != 8:
        raise ValueError("CuTe Ampere weight gradient requires compute capability 8.x")
    if any(t.dtype != torch.bfloat16 for t in tensors[:3]):
        raise TypeError("CuTe Ampere weight gradient currently supports BF16 only")
    if any(t.dtype != torch.int32 for t in tensors[3:]):
        raise TypeError("expert metadata tensors must be int32")
    if any(t.ndim != 2 for t in tensors[:3]):
        raise ValueError("a_rows, b_rows, and out must be matrices")
    if a_rows.shape[0] != b_rows.shape[0]:
        raise ValueError("weight-gradient row buffers must have equal row counts")
    if out.shape[1] != b_rows.shape[1] or out.shape[1] % 256:
        raise ValueError("output hidden dimension must match and be 256-aligned")
    if out.shape[0] % 128 or weight_block_expert.numel() * 128 != out.shape[0]:
        raise ValueError("weight_block_expert must contain one entry per 128 rows")
    if expert_col_off.shape != counts.shape or counts.shape != expert_row_start.shape:
        raise ValueError("expert metadata arrays must have the same shape")
    if row_block % 32:
        raise ValueError("row_block must be a multiple of the 32-row K tile")

    signature = (
        a_rows.device.index,
        a_rows.dtype,
        a_rows.shape[1],
        b_rows.shape[1],
        tuple(out.shape),
        row_block,
    )
    compiled = _WGRAD_CACHE.get(signature)
    if compiled is None:
        args = (
            _as_cute(a_rows, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(b_rows, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(a_rows, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(b_rows, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(out),
            _as_cute(weight_block_expert, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
            _as_cute(counts, dynamic_dim0=True),
            _as_cute(expert_row_start, dynamic_dim0=True),
        )
        op = _AmpereRaggedDown(
            cta_tiler=(128, 256, 64),
            num_stages=2,
            atom_layout_mnk=(2, 4, 1),
            weight_mode="wgrad",
        )
        fn = cute.compile(op, *args, row_block, options="--enable-tvm-ffi")
        compiled = _CompiledKernel(fn, signature)
        _WGRAD_CACHE[signature] = compiled

    compiled.fn(
        a_rows,
        b_rows,
        a_rows,
        b_rows,
        out,
        weight_block_expert,
        expert_col_off,
        counts,
        expert_row_start,
    )
