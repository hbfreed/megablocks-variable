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

``_AmpereGateUpRows`` adds the forward gate and up projections with SwiGLU,
``_AmpereDgu`` the backward ``dout @ w_down.T`` with the SwiGLU backward, and
``_AmpereWgradPair`` the gate and up weight gradients.

One CuTe launch spans every expert tile, retains device-side expert selection,
and never reads routing metadata back to the host.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import torch

try:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import tvm_ffi
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "The CuTe grouped-GEMM experiment needs nvidia-cutlass-dsl>=4.6 "
        "and apache-tvm-ffi. The normal Triton backend does not need them.",
    ) from exc


@functools.cache
def _is_ampere(device):
    return torch.cuda.get_device_capability(device)[0] == 8


def supports(x, w_gate, w_up, w_down):
    """Whether these kernels take grouped_moe's operands: BF16 x and weights on
    a compute capability 8.x GPU, with hidden a multiple of 256."""
    return (x.is_cuda and x.shape[1] % 256 == 0
            and all(t.dtype == torch.bfloat16 for t in (x, w_gate, w_up, w_down))
            and _is_ampere(x.device))


class _AmpereRaggedDown:
    """SM80/SM86 BF16 tensor-core GEMMs with runtime expert dimensions."""

    def __init__(
        self,
        cta_tiler=(128, 256, 32),
        num_stages=3,
        atom_layout_mnk=(2, 4, 1),
        weight_mode="down",
        accumulate=False,
    ):
        if weight_mode not in ("down", "dx_fused", "wgrad"):
            raise ValueError(f"unsupported ragged GEMM mode: {weight_mode!r}")
        # Add the fp32 tile to the output already in memory and round once
        # (gradient accumulation into an existing .grad). See _add_existing.
        self.accumulate = accumulate
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
        stream: cuda_driver.CUstream,
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
        if cutlass.const_expr(self.weight_mode == "wgrad"):
            # Each copy thread owns one row of the K tile, so it reads one
            # route token per stage. See _gather_b.
            per_row = self.num_threads // self.bK
            tiled_copy_B = cute.make_tiled_copy_tv(
                async_copy,
                cute.make_layout((per_row, self.bK), stride=(1, per_row)),
                cute.make_layout((copy_bits // w_down.element_type.width, 1)),
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

        # The N tile advances fastest, so a tile's CTAs run side by side and
        # share its operand through L2. See grouped_moe's note on grid order.
        grid = (
            cute.ceil_div(y.shape[1], self.bN),
            (cute.ceil_div(y.shape[0], self.bM)
             if self.weight_mode == "wgrad"
             else cute.ceil_div(h.shape[0], self.bM)),
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
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream)

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
        nt, mt, _ = cute.arch.block_idx()
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
                expert_w = weight_matrix
                a_tile = 0
            else:
                row_start = 0
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
                h2,
                weight_matrix,
                row_start,
                nt,
                tidx,
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
                    h2,
                    weight_matrix,
                    row_start,
                    nt,
                    tidx,
                )

            if cutlass.const_expr(self.accumulate):
                self._add_existing(
                    tiled_copy_C, tCgC_epilogue, tCsC_epilogue, tCsC, tCrC,
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
    def _add_existing(self, tiled_copy_C, tCgC_epilogue, tCsC_epilogue, tCsC,
                      tCrC):
        # tCrC += the output tile already in memory, in fp32, so the epilogue
        # rounds the sum once. The tile comes in through the store's own
        # coalesced copy and shared-memory staging buffer, then each thread
        # reads back its accumulator's elements. A thread then overwrites only
        # the shared elements it read, so the epilogue needs no extra barrier.
        tCrE_epilogue = cute.make_fragment_like(tCsC_epilogue)
        cute.copy(tiled_copy_C, tCgC_epilogue, tCrE_epilogue)
        cute.autovec_copy(tCrE_epilogue, tCsC_epilogue)
        cute.arch.sync_threads()
        tCrE = cute.make_fragment_like(tCrC, tCsC.element_type)
        cute.autovec_copy(tCsC, tCrE)
        tCrC.store(tCrC.load() + tCrE.load().to(cutlass.Float32))

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
        route_tokens,
        dout,
        row_start,
        nt,
        tidx,
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
                if cutlass.const_expr(self.weight_mode == "wgrad"):
                    self._gather_b(
                        route_tokens,
                        dout,
                        tBsB[None, None, None, stage],
                        row_start + k_tile_index * self.bK,
                        nt,
                        tidx,
                    )
                else:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, k_tile_index],
                        tBsB[None, None, None, stage],
                    )
            # Commit even an empty group: the wait below counts groups, so with
            # too few K tiles it would pass with this one still in flight.
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
                        if cutlass.const_expr(self.weight_mode == "wgrad"):
                            self._gather_b(
                                route_tokens,
                                dout,
                                tBsB[None, None, None, smem_pipe_write],
                                row_start + k_tile_index * self.bK,
                                nt,
                                tidx,
                            )
                        else:
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

    @cute.jit
    def _gather_b(self, route_tokens, dout, slot, row0, nt, tidx):
        # This thread's row of one K tile, gathered from dout[token]. slot is
        # its (V, chunks, 1) view of one smem stage. A padding row skips the
        # load: the A operand's row is exact zero there, so whatever this stage
        # holds, which is zero or an earlier row of this expert, multiplies to
        # zero and adds nothing as long as dout is finite. Against reading a
        # prebuilt row buffer this costs 0.015 ms in the kernel at the primary
        # shape.
        per_row = self.num_threads // self.bK
        chunk_elems = 128 // dout.element_type.width
        tok = route_tokens[row0 + tidx // per_row]
        if tok >= 0:
            src = (dout.iterator + tok * dout.shape[0] + nt * self.bN
                   + (tidx % per_row) * chunk_elems)
            atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(
                    cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
                ),
                dout.element_type,
                num_bits_per_copy=128,
            )
            layout = cute.make_layout((chunk_elems, 1))
            for i in cutlass.range_constexpr(self.bN // (chunk_elems * per_row)):
                ptr = src + i * per_row * chunk_elems
                cute.copy(
                    atom,
                    cute.make_tensor(ptr.align(16), layout),
                    slot[None, i, None],
                )

    # The helpers below stay undecorated: their Python loops run at trace time,
    # so each thread's rows unroll into plain registers rather than a device loop
    # over an rmem array.
    def _gather_rows(self, x, route_tokens, tAsA, row0, tidx):
        """One (is_token, x pointer) per row this thread copies; zeroes padding."""
        chunks = self.bK // 8
        thread_rows = self.num_threads // chunks
        row = row0 + tidx // chunks
        chunk = tidx % chunks
        hidden = cute.size(x, mode=[1])
        rows = []
        for i in range(self.bM // thread_rows):
            tok = route_tokens[row + i * thread_rows]
            is_token = tok >= 0
            slots = tAsA[None, i, None, None]
            cutlass.if_generate(tok < 0, lambda s=slots: s.fill(0))
            rows.append((is_token, x.iterator + (tok * hidden + chunk * 8)))
        return rows

    def _copy_x_tile(self, async_copy, rows, tAsA, k_tile, stage):
        offset = k_tile * self.bK
        for i, (is_token, ptr) in enumerate(rows):
            src = cute.make_tensor((ptr + offset).align(16), (8,))
            dst = tAsA[None, i, 0, stage]
            cutlass.if_generate(
                is_token,
                lambda s=src, d=dst: cute.copy(async_copy, s, d),
            )

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


class _AmpereGateUpRows(_AmpereRaggedDown):
    """Gate and up projections over padded rows, reading x once.

    g = x[tok(row)] @ Wg_e, u = x[tok(row)] @ Wu_e, h = w_r * silu(g) * u. The
    same fusion in Triton, measured before grouped_moe's alignment note, ran
    0.33 ms slower than its two launches at the primary shape; this one runs
    0.79 ms faster.

    The DSL's tiled copies need a regular source tensor, so x bypasses them: each
    thread computes one pointer per row it owns and issues cp.async itself, in
    the same 16 byte chunks the tiled copy would. A padding row skips its copies,
    and its SMEM slots stay at the zeros written once per tile.
    """

    def __init__(
        self,
        cta_tiler=(128, 128, 32),
        num_stages=2,
        atom_layout_mnk=(2, 4, 1),
    ):
        super().__init__(cta_tiler, num_stages, atom_layout_mnk, "down")

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w_gate: cute.Tensor,
        w_up: cute.Tensor,
        g: cute.Tensor,
        u: cute.Tensor,
        h: cute.Tensor,
        row_w: cute.Tensor,
        route_tokens: cute.Tensor,
        tile_expert: cute.Tensor,
        expert_col_off: cute.Tensor,
        expert_nblocks: cute.Tensor,
        nblock_n: cutlass.Constexpr,
        stream: cuda_driver.CUstream,
    ):
        # Gate and up weights are physically [hidden, total_width], the same
        # K-major shape as w_down in the down projection.
        gate_matrix = cute.make_tensor(
            w_gate.iterator,
            cute.select(w_gate.layout, mode=[1, 0]),
        )
        up_matrix = cute.make_tensor(
            w_up.iterator,
            cute.select(w_up.layout, mode=[1, 0]),
        )
        a_major = utils.LayoutEnum.ROW_MAJOR
        b_major = utils.LayoutEnum.COL_MAJOR
        c_major = utils.LayoutEnum.ROW_MAJOR
        copy_bits = 128

        sA_layout = self._make_smem_layout_ab(
            x.element_type, a_major, copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_ab(
            w_gate.element_type, b_major, copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = self._make_smem_layout_c(
            g.element_type, c_major, copy_bits, (self.bM, self.bN),
        )

        async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            ),
            x.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_ab(
            async_copy, x.element_type, a_major, copy_bits, self.bM,
        )
        tiled_copy_B = self._make_gmem_tiled_copy_ab(
            async_copy, w_gate.element_type, b_major, copy_bits, self.bN,
        )
        sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            g.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_c(
            sync_copy, g.element_type, c_major, copy_bits,
        )

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            x.element_type, cutlass.Float32, self.mma_inst_shape,
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=(
                self.atom_layout_mnk[0] * self.mma_inst_shape[0],
                self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
                self.atom_layout_mnk[2] * self.mma_inst_shape[2],
            ),
        )

        self.kernel(
            x,
            gate_matrix,
            up_matrix,
            g,
            u,
            h,
            row_w,
            route_tokens,
            tile_expert,
            expert_col_off,
            expert_nblocks,
            sA_layout,
            sB_layout,
            sC_layout,
            async_copy,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
            nblock_n,
        ).launch(
            # The width tile advances fastest, so a row tile's CTAs run side by
            # side and share its x rows through L2.
            grid=(
                cute.ceil_div(g.shape[1], self.bN),
                cute.ceil_div(g.shape[0], self.bM),
                1,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        gate_matrix: cute.Tensor,
        up_matrix: cute.Tensor,
        g: cute.Tensor,
        u: cute.Tensor,
        h: cute.Tensor,
        row_w: cute.Tensor,
        route_tokens: cute.Tensor,
        tile_expert: cute.Tensor,
        expert_col_off: cute.Tensor,
        expert_nblocks: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        async_copy: cute.CopyAtom,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        nblock_n: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        nt, mt, _ = cute.arch.block_idx()
        expert = tile_expert[mt]

        if expert < 0:
            pass
        elif nt * nblock_n >= expert_nblocks[expert]:
            pass
        else:
            col_off = expert_col_off[expert]
            gBg = cute.local_tile(
                cute.domain_offset((col_off, 0), gate_matrix),
                tiler=(self.bN, self.bK),
                coord=(nt, None),
            )
            gBu = cute.local_tile(
                cute.domain_offset((col_off, 0), up_matrix),
                tiler=(self.bN, self.bK),
                coord=(nt, None),
            )
            gBg = cute.make_tensor(gBg.iterator.align(16), gBg.layout)
            gBu = cute.make_tensor(gBu.iterator.align(16), gBu.layout)
            gG = cute.local_tile(g, tiler=(self.bM, self.bN), coord=(mt, nt))
            gU = cute.local_tile(u, tiler=(self.bM, self.bN), coord=(mt, nt))
            gH = cute.local_tile(h, tiler=(self.bM, self.bN), coord=(mt, nt))

            @cute.struct
            class SharedStorageAB:
                a: cute.struct.Align[
                    cute.struct.MemRange[x.element_type, cute.cosize(sA_layout)],
                    16,
                ]
                bg: cute.struct.Align[
                    cute.struct.MemRange[
                        gate_matrix.element_type, cute.cosize(sB_layout),
                    ],
                    16,
                ]
                bu: cute.struct.Align[
                    cute.struct.MemRange[
                        up_matrix.element_type, cute.cosize(sB_layout),
                    ],
                    16,
                ]

            @cute.struct
            class SharedStorageC:
                cg: cute.struct.Align[
                    cute.struct.MemRange[g.element_type, cute.cosize(sC_layout)],
                    16,
                ]
                cu: cute.struct.Align[
                    cute.struct.MemRange[u.element_type, cute.cosize(sC_layout)],
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
            sBg = SharedStorageAB(storage).bg.get_tensor(sB_layout)
            sBu = SharedStorageAB(storage).bu.get_tensor(sB_layout)
            sCg = SharedStorageC(storage).cg.get_tensor(sC_layout)
            sCu = SharedStorageC(storage).cu.get_tensor(sC_layout)

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tAsA = thr_copy_A.partition_D(sA)
            tBgBg = thr_copy_B.partition_S(gBg)
            tBsBg = thr_copy_B.partition_D(sBg)
            tBgBu = thr_copy_B.partition_S(gBu)
            tBsBu = thr_copy_B.partition_D(sBu)

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsBg = thr_mma.partition_B(sBg)
            tCsBu = thr_mma.partition_B(sBu)
            tCsCg = thr_mma.partition_C(sCg)
            tCsCu = thr_mma.partition_C(sCu)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrBg = tiled_mma.make_fragment_B(tCsBg[None, None, None, 0])
            tCrBu = tiled_mma.make_fragment_B(tCsBu[None, None, None, 0])
            tCrCg = tiled_mma.make_fragment_C(thr_mma.partition_C(gG))
            tCrCu = tiled_mma.make_fragment_C(thr_mma.partition_C(gU))
            tCrCg.fill(0.0)
            tCrCu.fill(0.0)

            copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), x.element_type,
            )
            copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4),
                gate_matrix.element_type,
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(copy_s2r_B, tiled_mma)
            thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy = thr_s2r_A.partition_S(sA)
            tCrA_copy = thr_s2r_A.retile(tCrA)
            tCsBg_copy = thr_s2r_B.partition_S(sBg)
            tCrBg_copy = thr_s2r_B.retile(tCrBg)
            tCsBu_copy = thr_s2r_B.partition_S(sBu)
            tCrBu_copy = thr_s2r_B.retile(tCrBu)

            # One chunk of each row's K tile per thread column, as the row-major
            # thread layout of _make_gmem_tiled_copy_ab assigns them.
            rows = self._gather_rows(x, route_tokens, tAsA, mt * self.bM, tidx)

            k_tile_count = cute.size(gate_matrix, mode=[1]) // self.bK
            num_smem_stages = self.num_stages
            k_tile_index = cutlass.Int32(0)
            for stage in range(num_smem_stages - 1):
                self._copy_x_tile(async_copy, rows, tAsA, k_tile_index, stage)
                cute.copy(
                    tiled_copy_B,
                    tBgBg[None, None, None, k_tile_index],
                    tBsBg[None, None, None, stage],
                )
                cute.copy(
                    tiled_copy_B,
                    tBgBu[None, None, None, k_tile_index],
                    tBsBu[None, None, None, stage],
                )
                cute.arch.cp_async_commit_group()
                k_tile_index = k_tile_index + 1

            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1
            tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
            tCsBg_p = tCsBg_copy[None, None, None, smem_pipe_read]
            tCsBu_p = tCsBu_copy[None, None, None, smem_pipe_read]
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
                    tCsBg_p[None, None, 0],
                    tCrBg_copy[None, None, 0],
                )
                cute.copy(
                    tiled_copy_s2r_B,
                    tCsBu_p[None, None, 0],
                    tCrBu_copy[None, None, 0],
                )

            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
                        tCsBg_p = tCsBg_copy[None, None, None, smem_pipe_read]
                        tCsBu_p = tCsBu_copy[None, None, None, smem_pipe_read]
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
                        tCsBg_p[None, None, k_block_next],
                        tCrBg_copy[None, None, k_block_next],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsBu_p[None, None, k_block_next],
                        tCrBu_copy[None, None, k_block_next],
                    )

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            self._copy_x_tile(
                                async_copy, rows, tAsA, k_tile_index,
                                smem_pipe_write,
                            )
                            cute.copy(
                                tiled_copy_B,
                                tBgBg[None, None, None, k_tile_index],
                                tBsBg[None, None, None, smem_pipe_write],
                            )
                            cute.copy(
                                tiled_copy_B,
                                tBgBu[None, None, None, k_tile_index],
                                tBsBu[None, None, None, smem_pipe_write],
                            )
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

                    cute.gemm(
                        tiled_mma,
                        tCrCg,
                        tCrA[None, None, k_block],
                        tCrBg[None, None, k_block],
                        tCrCg,
                    )
                    cute.gemm(
                        tiled_mma,
                        tCrCu,
                        tCrA[None, None, k_block],
                        tCrBu[None, None, k_block],
                        tCrCu,
                    )

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            tCrDg = cute.make_fragment_like(tCrCg, g.element_type)
            tCrDu = cute.make_fragment_like(tCrCu, u.element_type)
            tCrDg[None] = tCrCg.load().to(g.element_type)
            tCrDu[None] = tCrCu.load().to(u.element_type)
            cute.autovec_copy(tCrDg, tCsCg)
            cute.autovec_copy(tCrDu, tCsCu)
            cute.arch.sync_threads()

            tCsG = thr_copy_C.partition_S(sCg)
            tCsU = thr_copy_C.partition_S(sCu)
            tCrG = cute.make_fragment_like(tCsG)
            tCrU = cute.make_fragment_like(tCsU)
            cute.autovec_copy(tCsG, tCrG)
            cute.autovec_copy(tCsU, tCrU)
            cute.copy(tiled_copy_C, tCrG, thr_copy_C.partition_D(gG))
            cute.copy(tiled_copy_C, tCrU, thr_copy_C.partition_D(gU))
            self._silu_epilogue(
                tiled_copy_C,
                tCrG,
                tCrU,
                thr_copy_C.partition_D(gH),
                row_w,
                mt * self.bM,
                tidx,
            )
        return

    def _silu_epilogue(self, tiled_copy_C, tCrG, tCrU, tCgH, row_w, row0, tidx):
        # h comes from the bf16 values just stored, not the fp32 accumulators:
        # backward reads the stored g and u. These are the Triton kernels'
        # operations, so _dgu reproduces h bit for bit: exp is mul +
        # ex2.approx and 1/x is div.full, both scalar-only intrinsics. The DSL's
        # defaults (six FMAs + ex2, IEEE rcp) differ in the last bf16 bit.
        chunks = self.bN // 8
        thread_rows = self.num_threads // chunks
        row = row0 + tidx // chunks
        one = cutlass.Float32(1.0)
        log2e = cutlass.Float32(1.4426950408889634)
        tCrH = cute.make_fragment_like(tCrU)
        tCrF = cute.make_fragment_like(tCrU, cutlass.Float32)
        for i in range(self.bM // thread_rows):
            gg = tCrG[None, i, 0].load().to(cutlass.Float32)
            uu = tCrU[None, i, 0].load().to(cutlass.Float32)
            ww = row_w[row + i * thread_rows]
            for j in range(8):
                val = gg[j]
                e = cute.math.exp2(-val * log2e, approx=True)
                sig = cute.math.div(one, one + e, full=True)
                tCrF[j, i, 0] = val * sig * uu[j] * ww
            tCrH[None, i, 0] = tCrF[None, i, 0].load().to(tCrU.element_type)
        cute.copy(tiled_copy_C, tCrH, tCgH)


class _AmpereDgu(_AmpereRaggedDown):
    """``da = dout[tok(row)] @ w_down.T`` with the SwiGLU backward fused.

    grouped_moe's Triton ``_dgu`` describes the math. dout rows are gathered as
    ``_AmpereGateUpRows`` gathers x; padding rows keep the zeros written once per
    tile, so their da is exact zero.

    g and u are read through smem. 128-bit ``cp.async`` copies put them in the
    pipeline stages that the last two K tiles no longer write, so they arrive
    while the MMAs run. A g or u tile is as large as one stage of A and B
    together: its first half of rows sits in stage s of A and the rest in stage
    s of B. dg and du overwrite g and u in place, h goes to a third stage, and
    all three leave through 128-bit stores.

    A row's columns span the atom's N warps. Each warp sums its share of the
    routing-weight dot and writes its own ``dw_part`` column, in place of a
    cross-warp reduction.
    """

    def __init__(self, cfg, fuse_h):
        bm, bn, bk, stages, atom = cfg
        super().__init__((bm, bn, bk), stages, tuple(atom), "down")
        if bm != bn or bn != 2 * bk or stages < 3:
            raise ValueError("the g/u staging needs BLOCK_M = BLOCK_N = 2 * BLOCK_K "
                             "and 3+ stages")
        self.fuse_h = fuse_h
        self.parts_per_tile = atom[1]

    @cute.jit
    def __call__(
        self,
        dout: cute.Tensor,
        w_down: cute.Tensor,
        g: cute.Tensor,
        u: cute.Tensor,
        row_w: cute.Tensor,
        dg: cute.Tensor,
        du: cute.Tensor,
        dw_part: cute.Tensor,
        h: cute.Tensor,
        route_tokens: cute.Tensor,
        tile_expert: cute.Tensor,
        expert_col_off: cute.Tensor,
        expert_nblocks: cute.Tensor,
        width_block: cutlass.Constexpr,
        stream: cuda_driver.CUstream,
    ):
        major = utils.LayoutEnum.ROW_MAJOR
        copy_bits = 128
        sA_layout = self._make_smem_layout_ab(
            dout.element_type, major, copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_ab(
            w_down.element_type, major, copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            ),
            dout.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_ab(
            async_copy, dout.element_type, major, copy_bits, self.bM,
        )
        tiled_copy_B = self._make_gmem_tiled_copy_ab(
            async_copy, w_down.element_type, major, copy_bits, self.bN,
        )
        tiled_copy_gu = self._make_gmem_tiled_copy_c(
            async_copy, g.element_type, major, copy_bits,
        )
        sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dg.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_c(
            sync_copy, dg.element_type, major, copy_bits,
        )
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            dout.element_type, cutlass.Float32, self.mma_inst_shape,
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=(
                self.atom_layout_mnk[0] * self.mma_inst_shape[0],
                self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
                self.atom_layout_mnk[2] * self.mma_inst_shape[2],
            ),
        )
        self.kernel(
            dout,
            w_down,
            g,
            u,
            row_w,
            dg,
            du,
            dw_part,
            h,
            route_tokens,
            tile_expert,
            expert_col_off,
            expert_nblocks,
            sA_layout,
            sB_layout,
            async_copy,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_gu,
            tiled_copy_C,
            tiled_mma,
            width_block,
        ).launch(
            # The width tile advances fastest, so a row tile's CTAs run side by
            # side and share its dout rows through L2.
            grid=(
                cute.ceil_div(g.shape[1], self.bN),
                cute.ceil_div(g.shape[0], self.bM),
                1,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dout: cute.Tensor,
        w_down: cute.Tensor,
        g: cute.Tensor,
        u: cute.Tensor,
        row_w: cute.Tensor,
        dg: cute.Tensor,
        du: cute.Tensor,
        dw_part: cute.Tensor,
        h: cute.Tensor,
        route_tokens: cute.Tensor,
        tile_expert: cute.Tensor,
        expert_col_off: cute.Tensor,
        expert_nblocks: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        async_copy: cute.CopyAtom,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_gu: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        width_block: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        nt, mt, _ = cute.arch.block_idx()
        expert = tile_expert[mt]

        if expert < 0:
            pass
        elif nt * self.bN >= expert_nblocks[expert] * width_block:
            pass
        else:
            gB = cute.local_tile(
                cute.domain_offset((expert_col_off[expert], 0), w_down),
                tiler=(self.bN, self.bK),
                coord=(nt, None),
            )
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
            gG = cute.local_tile(g, tiler=(self.bM, self.bN), coord=(mt, nt))
            gU = cute.local_tile(u, tiler=(self.bM, self.bN), coord=(mt, nt))
            gG = cute.make_tensor(gG.iterator.align(16), gG.layout)
            gU = cute.make_tensor(gU.iterator.align(16), gU.layout)
            gDg = cute.local_tile(dg, tiler=(self.bM, self.bN), coord=(mt, nt))
            gDu = cute.local_tile(du, tiler=(self.bM, self.bN), coord=(mt, nt))

            @cute.struct
            class SharedStorageAB:
                a: cute.struct.Align[
                    cute.struct.MemRange[dout.element_type, cute.cosize(sA_layout)],
                    16,
                ]
                b: cute.struct.Align[
                    cute.struct.MemRange[
                        w_down.element_type, cute.cosize(sB_layout),
                    ],
                    16,
                ]

            smem = utils.SmemAllocator()
            storage = smem.allocate(
                SharedStorageAB.size_in_bytes(), byte_alignment=16,
            )
            sA = SharedStorageAB(storage).a.get_tensor(sA_layout)
            sB = SharedStorageAB(storage).b.get_tensor(sB_layout)
            # (row, col, stage): a tile's rows split between stage s of A and
            # stage s of B, which sit one whole A pipeline apart.
            sGU = cute.make_tensor(
                sA.iterator,
                cute.make_composed_layout(
                    cute.make_swizzle(3, 3, 4),
                    0,
                    cute.make_layout(
                        ((self.bM // 2, 2), self.bN, self.num_stages),
                        stride=(
                            (self.bN, cute.cosize(sA_layout)),
                            1,
                            self.bM * self.bK,
                        ),
                    ),
                ),
            )

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_gu = tiled_copy_gu.get_slice(tidx)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tGgG = thr_copy_gu.partition_S(gG)
            tGgU = thr_copy_gu.partition_S(gU)
            tGsGU = thr_copy_gu.partition_D(sGU)

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(gDg))
            tCrC.fill(0.0)

            copy_s2r = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), dout.element_type,
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(copy_s2r, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(copy_s2r, tiled_mma)
            thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy = thr_s2r_A.partition_S(sA)
            tCrA_copy = thr_s2r_A.retile(tCrA)
            tCsB_copy = thr_s2r_B.partition_S(sB)
            tCrB_copy = thr_s2r_B.retile(tCrB)

            rows = self._gather_rows(dout, route_tokens, tAsA, mt * self.bM, tidx)

            k_tile_count = cute.size(w_down, mode=[1]) // self.bK
            num_smem_stages = self.num_stages
            k_tile_index = cutlass.Int32(0)
            for stage in range(num_smem_stages - 1):
                self._copy_x_tile(async_copy, rows, tAsA, k_tile_index, stage)
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
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0],
                          tCrA_copy[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0],
                          tCrB_copy[None, None, 0])

            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next],
                              tCrA_copy[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next],
                              tCrB_copy[None, None, k_block_next])

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            self._copy_x_tile(
                                async_copy, rows, tAsA, k_tile_index,
                                smem_pipe_write,
                            )
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, None, k_tile_index],
                                tBsB[None, None, None, smem_pipe_write],
                            )
                        elif k_tile + num_smem_stages - 1 == k_tile_count:
                            # The last K tiles write no operands; their stages
                            # take g and then u.
                            cute.copy(
                                tiled_copy_gu, tGgG,
                                tGsGU[None, None, None, smem_pipe_write],
                            )
                        elif k_tile + num_smem_stages - 2 == k_tile_count:
                            cute.copy(
                                tiled_copy_gu, tGgU,
                                tGsGU[None, None, None, smem_pipe_write],
                            )
                        # Commit even an empty group: the wait above counts
                        # groups.
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block],
                              tCrB[None, None, k_block], tCrC)

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # K tile t lands in stage t % stages, so g, copied where tile
            # k_tile_count would go, is in stage k_tile_count % stages.
            sG = sGU[None, None, k_tile_count % self.num_stages]
            sU = sGU[None, None, (k_tile_count + 1) % self.num_stages]
            sH = sGU[None, None, (k_tile_count + 2) % self.num_stages]

            row_base = mt * self.bM
            # Rows are the only thing these index, so their columns have stride 0.
            tCgW = thr_mma.partition_C(
                cute.make_tensor(
                    row_w.iterator + row_base,
                    cute.make_layout((self.bM, self.bN), stride=(1, 0)),
                ),
            )
            num_part = cute.size(dw_part, mode=[1])
            warps_per_part = self.atom_layout_mnk[0]
            tCgP = thr_mma.partition_C(
                cute.make_tensor(
                    dw_part.iterator
                    + row_base * num_part
                    + nt * self.parts_per_tile
                    + tidx // (32 * warps_per_part),
                    cute.make_layout((self.bM, self.bN), stride=(num_part, 0)),
                ),
            )
            self._swiglu_epilogue(
                tCrC,
                thr_mma.partition_C(sG),
                thr_mma.partition_C(sU),
                thr_mma.partition_C(sH),
                tCgW,
                tCgP,
                tidx,
            )
            cute.arch.sync_threads()

            thr_copy_C = tiled_copy_C.get_slice(tidx)
            self._store_tile(tiled_copy_C, thr_copy_C, sG, gDg)
            self._store_tile(tiled_copy_C, thr_copy_C, sU, gDu)
            if cutlass.const_expr(self.fuse_h):
                self._store_tile(
                    tiled_copy_C, thr_copy_C, sH,
                    cute.local_tile(h, tiler=(self.bM, self.bN), coord=(mt, nt)),
                )
        return

    def _store_tile(self, tiled_copy_C, thr_copy_C, s, gD):
        tEsD = thr_copy_C.partition_S(s)
        tErD = cute.make_fragment_like(tEsD)
        cute.autovec_copy(tEsD, tErD)
        cute.copy(tiled_copy_C, tErD, thr_copy_C.partition_D(gD))

    @cute.jit
    def _swiglu_epilogue(self, tCrC, tCsG, tCsU, tCsH, tCgW, tCgP, tidx):
        # One thread holds ((2, 2), M, N): pairs of columns on two rows 8 apart,
        # over M row and N column blocks. Rows are the lower modes of the linear
        # index, so a row's elements in this thread are the ones sharing (r, m).
        n_m = cute.size(tCrC, mode=[1])
        n_n = cute.size(tCrC, mode=[2])
        n_rows = 2 * n_m
        weights = [None] * n_rows
        parts = [None] * n_rows
        for j in cutlass.range_constexpr(n_rows):
            weights[j] = tCgW[2 * (j % 2) + 4 * (j // 2)]
        one = cutlass.Float32(1.0)
        log2e = cutlass.Float32(1.4426950408889634)

        def swiglu(acc, gg, uu, w):
            # Triton's sigmoid: mul + ex2.approx, then div.full, as in
            # _AmpereGateUpRows._silu_epilogue. The operation order is _dgu's.
            sig = cute.math.div(
                one, one + cute.math.exp2(-gg * log2e, approx=True), full=True,
            )
            silu = gg * sig
            dh = acc * w
            return (
                ((dh * uu) * (sig * (one + gg * (one - sig)))).to(cutlass.BFloat16),
                (dh * silu).to(cutlass.BFloat16),
                acc * silu * uu,
                (silu * uu * w).to(cutlass.BFloat16),
            )

        for n in cutlass.range_constexpr(n_n):
            for m in cutlass.range_constexpr(n_m):
                for r in cutlass.range_constexpr(2):
                    j = r + 2 * m
                    i = 2 * r + 4 * m + 4 * n_m * n
                    # A pair is read and written together so that LLVM merges
                    # each into one 32-bit smem access.
                    g0 = tCsG[i].to(cutlass.Float32)
                    g1 = tCsG[i + 1].to(cutlass.Float32)
                    u0 = tCsU[i].to(cutlass.Float32)
                    u1 = tCsU[i + 1].to(cutlass.Float32)
                    dg0, du0, p0, h0 = swiglu(tCrC[i], g0, u0, weights[j])
                    dg1, du1, p1, h1 = swiglu(tCrC[i + 1], g1, u1, weights[j])
                    tCsG[i] = dg0
                    tCsG[i + 1] = dg1
                    tCsU[i] = du0
                    tCsU[i + 1] = du1
                    if cutlass.const_expr(self.fuse_h):
                        tCsH[i] = h0
                        tCsH[i + 1] = h1
                    parts[j] = (
                        p0 + p1
                        if cutlass.const_expr(parts[j] is None)
                        else parts[j] + p0 + p1
                    )
        # All four lanes of a quad hold the same rows, and butterfly shuffles
        # give each the same bits regardless of timing.
        for j in cutlass.range_constexpr(n_rows):
            parts[j] = cute.arch.warp_reduction_sum(parts[j], threads_in_group=4)
        if tidx % 4 == 0:
            for j in cutlass.range_constexpr(n_rows):
                tCgP[2 * (j % 2) + 4 * (j // 2)] = parts[j]


class _AmpereWgradPair(_AmpereRaggedDown):
    """dWg = x[tok]^T @ dg and dWu = x[tok]^T @ du in one launch, reading x once.

    One CTA owns a (hidden, width) tile of both weights and keeps two fp32
    accumulators. The K loop runs over the expert's rows, in order, and stops at
    the route count rounded up to the K tile, as grouped_moe's ``_wgrad_gather``
    does. Each copy thread gathers its K row of x from ``x[route_tokens[row]]``;
    a padding row skips the copy, and dg and du are exact zeros there, so the
    stale slot adds nothing as long as x is finite.

    With ``accumulate`` the results are added to the outputs' contents in fp32
    and rounded once.
    """

    def __init__(self, cfg, accumulate=False):
        bm, bn, bk, stages, atom = cfg
        super().__init__((bm, bn, bk), stages, tuple(atom), "down")
        self.accumulate = accumulate

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        dg: cute.Tensor,
        du: cute.Tensor,
        dwg: cute.Tensor,
        dwu: cute.Tensor,
        route_tokens: cute.Tensor,
        weight_block_expert: cute.Tensor,
        expert_col_off: cute.Tensor,
        counts: cute.Tensor,
        expert_row_start: cute.Tensor,
        stream: cuda_driver.CUstream,
    ):
        # dg and du are [rows, max_width]; as (width, rows) they are the B
        # operand with N contiguous. x^T is (hidden, tokens) with M contiguous.
        dg_t = cute.make_tensor(dg.iterator, cute.select(dg.layout, mode=[1, 0]))
        du_t = cute.make_tensor(du.iterator, cute.select(du.layout, mode=[1, 0]))
        mn_major = utils.LayoutEnum.COL_MAJOR
        copy_bits = 128
        sA_layout = self._make_smem_layout_ab(
            x.element_type, mn_major, copy_bits,
            (self.bM, self.bK, self.num_stages),
        )
        sB_layout = self._make_smem_layout_ab(
            dg.element_type, mn_major, copy_bits,
            (self.bN, self.bK, self.num_stages),
        )
        sC_layout = self._make_smem_layout_c(
            dwg.element_type, utils.LayoutEnum.ROW_MAJOR, copy_bits,
            (self.bM, self.bN),
        )
        async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL,
            ),
            x.element_type,
            num_bits_per_copy=copy_bits,
        )
        # Each copy thread owns one K row of the x tile: one route token per
        # stage. See _gather_k_row.
        per_row = self.num_threads // self.bK
        tiled_copy_A = cute.make_tiled_copy_tv(
            async_copy,
            cute.make_layout((per_row, self.bK), stride=(1, per_row)),
            cute.make_layout((copy_bits // x.element_type.width, 1)),
        )
        tiled_copy_B = self._make_gmem_tiled_copy_ab(
            async_copy, dg.element_type, mn_major, copy_bits, self.bN,
        )
        sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dwg.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_c(
            sync_copy, dwg.element_type, utils.LayoutEnum.ROW_MAJOR, copy_bits,
        )
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            x.element_type, cutlass.Float32, self.mma_inst_shape,
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=(
                self.atom_layout_mnk[0] * self.mma_inst_shape[0],
                self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
                self.atom_layout_mnk[2] * self.mma_inst_shape[2],
            ),
        )
        self.kernel(
            x,
            dg_t,
            du_t,
            dwg,
            dwu,
            route_tokens,
            weight_block_expert,
            expert_col_off,
            counts,
            expert_row_start,
            sA_layout,
            sB_layout,
            sC_layout,
            async_copy,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
        ).launch(
            # The hidden tile advances fastest, so a width block's CTAs run side
            # by side and share its dg and du rows through L2.
            grid=(
                cute.ceil_div(dwg.shape[0], self.bM),
                cute.ceil_div(dwg.shape[1], self.bN),
                1,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        dg_t: cute.Tensor,
        du_t: cute.Tensor,
        dwg: cute.Tensor,
        dwu: cute.Tensor,
        route_tokens: cute.Tensor,
        weight_block_expert: cute.Tensor,
        expert_col_off: cute.Tensor,
        counts: cute.Tensor,
        expert_row_start: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        async_copy: cute.CopyAtom,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        ht, wb, _ = cute.arch.block_idx()
        expert = weight_block_expert[wb]
        count = counts[expert]
        row_start = expert_row_start[expert]
        k_tile_count = cute.ceil_div(count, self.bK)
        local_col = wb * self.bN - expert_col_off[expert]

        gBg = cute.local_tile(
            cute.domain_offset((local_col, row_start), dg_t),
            tiler=(self.bN, self.bK),
            coord=(0, None),
        )
        gBu = cute.local_tile(
            cute.domain_offset((local_col, row_start), du_t),
            tiler=(self.bN, self.bK),
            coord=(0, None),
        )
        gBg = cute.make_tensor(gBg.iterator.align(16), gBg.layout)
        gBu = cute.make_tensor(gBu.iterator.align(16), gBu.layout)
        gCg = cute.local_tile(dwg, tiler=(self.bM, self.bN), coord=(ht, wb))
        gCu = cute.local_tile(dwu, tiler=(self.bM, self.bN), coord=(ht, wb))

        @cute.struct
        class SharedStorageAB:
            a: cute.struct.Align[
                cute.struct.MemRange[x.element_type, cute.cosize(sA_layout)],
                16,
            ]
            bg: cute.struct.Align[
                cute.struct.MemRange[dg_t.element_type, cute.cosize(sB_layout)],
                16,
            ]
            bu: cute.struct.Align[
                cute.struct.MemRange[du_t.element_type, cute.cosize(sB_layout)],
                16,
            ]

        @cute.struct
        class SharedStorageC:
            cg: cute.struct.Align[
                cute.struct.MemRange[dwg.element_type, cute.cosize(sC_layout)],
                16,
            ]
            cu: cute.struct.Align[
                cute.struct.MemRange[dwu.element_type, cute.cosize(sC_layout)],
                16,
            ]

        smem = utils.SmemAllocator()
        storage = smem.allocate(
            max(SharedStorageAB.size_in_bytes(), SharedStorageC.size_in_bytes()),
            byte_alignment=16,
        )
        sA = SharedStorageAB(storage).a.get_tensor(sA_layout)
        sBg = SharedStorageAB(storage).bg.get_tensor(sB_layout)
        sBu = SharedStorageAB(storage).bu.get_tensor(sB_layout)
        sCg = SharedStorageC(storage).cg.get_tensor(sC_layout)
        sCu = SharedStorageC(storage).cu.get_tensor(sC_layout)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAsA = thr_copy_A.partition_D(sA)
        tBgBg = thr_copy_B.partition_S(gBg)
        tBsBg = thr_copy_B.partition_D(sBg)
        tBgBu = thr_copy_B.partition_S(gBu)
        tBsBu = thr_copy_B.partition_D(sBu)

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsBg = thr_mma.partition_B(sBg)
        tCsBu = thr_mma.partition_B(sBu)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrBg = tiled_mma.make_fragment_B(tCsBg[None, None, None, 0])
        tCrBu = tiled_mma.make_fragment_B(tCsBu[None, None, None, 0])
        tCrCg = tiled_mma.make_fragment_C(thr_mma.partition_C(gCg))
        tCrCu = tiled_mma.make_fragment_C(thr_mma.partition_C(gCu))
        tCrCg.fill(0.0)
        tCrCu.fill(0.0)

        copy_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), x.element_type,
        )
        tiled_copy_s2r_A = cute.make_tiled_copy_A(copy_s2r, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(copy_s2r, tiled_mma)
        thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsA_copy = thr_s2r_A.partition_S(sA)
        tCrA_copy = thr_s2r_A.retile(tCrA)
        tCsBg_copy = thr_s2r_B.partition_S(sBg)
        tCrBg_copy = thr_s2r_B.retile(tCrBg)
        tCsBu_copy = thr_s2r_B.partition_S(sBu)
        tCrBu_copy = thr_s2r_B.retile(tCrBu)

        # A padding row never copies its x slot. Zero all of them once, so
        # uninitialized smem cannot hold a NaN that multiplies a zero dg.
        tAsA.fill(0)
        cute.arch.sync_threads()

        num_smem_stages = self.num_stages
        k_tile_index = cutlass.Int32(0)
        for stage in range(num_smem_stages - 1):
            if k_tile_index < k_tile_count:
                self._load_stage(
                    async_copy, x, route_tokens, tAsA, tiled_copy_B, tBgBg,
                    tBsBg, tBgBu, tBsBu, row_start, ht, k_tile_index, stage,
                    tidx,
                )
            # Commit even an empty group: the waits below count groups, so with
            # too few K tiles one would pass with this one still in flight.
            cute.arch.cp_async_commit_group()
            k_tile_index = k_tile_index + 1

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1
        tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
        tCsBg_p = tCsBg_copy[None, None, None, smem_pipe_read]
        tCsBu_p = tCsBu_copy[None, None, None, smem_pipe_read]
        num_k_block = cute.size(tCrA, mode=[2])

        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0],
                      tCrA_copy[None, None, 0])
            cute.copy(tiled_copy_s2r_B, tCsBg_p[None, None, 0],
                      tCrBg_copy[None, None, 0])
            cute.copy(tiled_copy_s2r_B, tCsBu_p[None, None, 0],
                      tCrBu_copy[None, None, 0])

        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_copy[None, None, None, smem_pipe_read]
                    tCsBg_p = tCsBg_copy[None, None, None, smem_pipe_read]
                    tCsBu_p = tCsBu_copy[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                k_block_next = (k_block + 1) % num_k_block
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next],
                          tCrA_copy[None, None, k_block_next])
                cute.copy(tiled_copy_s2r_B, tCsBg_p[None, None, k_block_next],
                          tCrBg_copy[None, None, k_block_next])
                cute.copy(tiled_copy_s2r_B, tCsBu_p[None, None, k_block_next],
                          tCrBu_copy[None, None, k_block_next])

                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        self._load_stage(
                            async_copy, x, route_tokens, tAsA, tiled_copy_B,
                            tBgBg, tBsBg, tBgBu, tBsBu, row_start, ht,
                            k_tile_index, smem_pipe_write, tidx,
                        )
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0

                cute.gemm(tiled_mma, tCrCg, tCrA[None, None, k_block],
                          tCrBg[None, None, k_block], tCrCg)
                cute.gemm(tiled_mma, tCrCu, tCrA[None, None, k_block],
                          tCrBu[None, None, k_block], tCrCu)

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        thr_copy_C = tiled_copy_C.get_slice(tidx)
        for acc, sC, gC in ((tCrCg, sCg, gCg), (tCrCu, sCu, gCu)):
            tCsC = thr_mma.partition_C(sC)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)
            tCrD = cute.make_fragment_like(acc, dwg.element_type)
            if cutlass.const_expr(self.accumulate):
                # Stage the existing tile through smem with coalesced loads,
                # then read it in the accumulator's layout.
                tCrE = cute.make_fragment_like(tCsC_epilogue)
                cute.copy(tiled_copy_C, tCgC_epilogue, tCrE)
                cute.autovec_copy(tCrE, tCsC_epilogue)
                cute.arch.sync_threads()
                cute.autovec_copy(tCsC, tCrD)
                # Each thread overwrites only the elements it read.
                tCrD[None] = (
                    acc.load() + tCrD.load().to(cutlass.Float32)
                ).to(dwg.element_type)
            else:
                tCrD[None] = acc.load().to(dwg.element_type)
            cute.autovec_copy(tCrD, tCsC)
        cute.arch.sync_threads()
        for sC, gC in ((sCg, gCg), (sCu, gCu)):
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)
            cute.copy(tiled_copy_C, tCrC_epilogue, thr_copy_C.partition_D(gC))
        return

    def _load_stage(self, async_copy, x, route_tokens, tAsA, tiled_copy_B,
                    tBgBg, tBsBg, tBgBu, tBsBu, row_start, ht, k_tile, stage,
                    tidx):
        self._gather_k_row(async_copy, x, route_tokens, tAsA[None, None, None, stage],
                           row_start + k_tile * self.bK, ht, tidx)
        cute.copy(tiled_copy_B, tBgBg[None, None, None, k_tile],
                  tBsBg[None, None, None, stage])
        cute.copy(tiled_copy_B, tBgBu[None, None, None, k_tile],
                  tBsBu[None, None, None, stage])

    def _gather_k_row(self, async_copy, x, route_tokens, slot, row0, ht, tidx):
        # This thread's K row of one stage, x[token][ht * bM + ...]. slot is its
        # (V, chunks, 1) view of the stage.
        per_row = self.num_threads // self.bK
        chunk_elems = 8
        tok = route_tokens[row0 + tidx // per_row]
        hidden = cute.size(x, mode=[1])
        src = (x.iterator + tok * hidden + ht * self.bM
               + (tidx % per_row) * chunk_elems)
        layout = cute.make_layout((chunk_elems, 1))
        for i in range(self.bM // (chunk_elems * per_row)):
            ptr = src + i * per_row * chunk_elems
            s_src = cute.make_tensor(ptr.align(16), layout)
            dst = slot[None, i, None]
            cutlass.if_generate(
                tok >= 0,
                lambda s=s_src, d=dst: cute.copy(async_copy, s, d),
            )


@dataclass
class _CompiledKernel:
    fn: object
    signature: tuple


_CACHE: dict[tuple, _CompiledKernel] = {}
_DX_FUSED_CACHE: dict[tuple, _CompiledKernel] = {}
_WGRAD_CACHE: dict[tuple, _CompiledKernel] = {}
_GATE_UP_CACHE: dict[tuple, _CompiledKernel] = {}
_DGU_CACHE: dict[tuple, _CompiledKernel] = {}
_WGRAD_PAIR_CACHE: dict[tuple, _CompiledKernel] = {}
# Fused dWg + dWu: (BLOCK_M on hidden, BLOCK_N on the width, BLOCK_K on rows,
# stages, MNK atom layout). At hidden 512 (128 experts x 512, 8192 tokens, top-k
# 8) the launch took 0.78 ms against 0.83 ms for the two Triton launches; K64 with
# two stages took 0.82, a 64-row hidden tile 1.00, and K16 1.01. One CuTe launch
# per weight, through the wgrad mode with a transposed store, was 2-24% slower
# than Triton.
_WGRAD_PAIR_CFG = (128, 128, 32, 3, (2, 4, 1))
# _dgu: (BLOCK_M, BLOCK_N on the width, BLOCK_K on hidden, stages, MNK atom
# layout). The g/u staging needs BLOCK_M = BLOCK_N = 2 * BLOCK_K. At hidden 512
# this took 0.76 ms against 1.18 ms for the Triton _dgu, and 1.16 against 1.34 at
# hidden 2048. Atoms (2, 4, 1) and (8, 1, 1) took 0.77 and 0.78 ms. It uses 255
# registers, one CTA per SM. K32 tiles with the epilogue in two column chunks fit
# two CTAs per SM in 128 registers, but spilled and took 0.82-0.88 ms.
_DGU_CFG = (128, 128, 64, 3, (4, 2, 1))
# (BLOCK_M, BLOCK_N, BLOCK_K, stages, MNK atom layout). In sustained blocks at the
# primary shape this took 1.65 ms; 2 stages 1.78, K16 2.15, and 16 warps 2.45.
_GATE_UP_CFG = (128, 128, 32, 3, (2, 4, 1))


# Compiled kernels take their stream from TVM-FFI's environment, not an argument.
_ENV_STREAM = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


def _launch(compiled: _CompiledKernel, *args) -> None:
    """Launch on torch's current stream, as the Triton kernels do."""
    # Without a context the environment's stream is the default one. Entering
    # the context on every launch cost 0.05 ms a step at small shapes, and
    # torch.cuda.current_stream() 0.02 ms, so read the raw handle as Triton does.
    if torch._C._cuda_getCurrentRawStream(args[0].device.index) == 0:
        compiled.fn(*args)
    else:
        with tvm_ffi.use_torch_stream():
            compiled.fn(*args)


def _as_cute(
    tensor: torch.Tensor,
    dynamic_dim0: bool = False,
    dim0_divisibility: int = 1,
):
    # Compiling reads only the layout. A one-row view is contiguous but keeps its
    # parent's stride, which the compact dynamic shape below rejects.
    if tensor.is_contiguous() and tensor.ndim == 2 and tensor.stride(0) != tensor.shape[1]:
        tensor = tensor.new_empty(tensor.shape)
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
    compiled = cute.compile(op, *args, width_block, _ENV_STREAM, options="--enable-tvm-ffi")
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
    _launch(
        compiled,
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
        fn = cute.compile(op, *args, width_block, _ENV_STREAM, options="--enable-tvm-ffi")
        compiled = _CompiledKernel(fn, signature)
        _DX_FUSED_CACHE[signature] = compiled

    _launch(
        compiled,
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
    dout: torch.Tensor,
    route_tokens: torch.Tensor,
    out: torch.Tensor,
    weight_block_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    counts: torch.Tensor,
    expert_row_start: torch.Tensor,
    accumulate: bool = False,
) -> None:
    """Compute packed ``a_rows.T @ dout[route_tokens]`` expert weight gradients.

    ``accumulate=True`` adds them to ``out`` in fp32 and rounds once, instead
    of overwriting it.
    """
    tensors = (
        a_rows,
        dout,
        out,
        route_tokens,
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
        raise TypeError("route and expert metadata tensors must be int32")
    if any(t.ndim != 2 for t in tensors[:3]):
        raise ValueError("a_rows, dout, and out must be matrices")
    if a_rows.shape[0] != route_tokens.numel():
        raise ValueError("route_tokens must contain one entry per a_rows row")
    if out.shape[1] != dout.shape[1] or out.shape[1] % 256:
        raise ValueError("output hidden dimension must match and be 256-aligned")
    if out.shape[0] % 128 or weight_block_expert.numel() * 128 != out.shape[0]:
        raise ValueError("weight_block_expert must contain one entry per 128 rows")
    if expert_col_off.shape != counts.shape or counts.shape != expert_row_start.shape:
        raise ValueError("expert metadata arrays must have the same shape")
    # dout is the caller's tensor, and the gather's 16-byte async copies reject
    # a contiguous view at an unaligned offset. A fresh allocation is aligned.
    if dout.data_ptr() % 16:
        dout = dout.clone()

    signature = (
        a_rows.device.index,
        a_rows.dtype,
        a_rows.shape[1],
        dout.shape[1],
        tuple(out.shape),
        accumulate,
    )
    compiled = _WGRAD_CACHE.get(signature)
    if compiled is None:
        args = (
            _as_cute(a_rows, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(dout, dynamic_dim0=True),
            _as_cute(route_tokens, dynamic_dim0=True),
            _as_cute(dout, dynamic_dim0=True),
            _as_cute(out),
            _as_cute(weight_block_expert, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
            _as_cute(counts, dynamic_dim0=True),
            _as_cute(expert_row_start, dynamic_dim0=True),
        )
        # Rows past the route count are exact zeros, so the K loop stops at the
        # count rounded up to the K tile. 3 stages at this tile need 147 KB of
        # shared memory; sm_86 allows 101 KB.
        k_tile = 64
        op = _AmpereRaggedDown(
            cta_tiler=(128, 256, k_tile),
            num_stages=2,
            atom_layout_mnk=(2, 4, 1),
            weight_mode="wgrad",
            accumulate=accumulate,
        )
        fn = cute.compile(op, *args, k_tile, _ENV_STREAM, options="--enable-tvm-ffi")
        compiled = _CompiledKernel(fn, signature)
        _WGRAD_CACHE[signature] = compiled

    _launch(
        compiled,
        a_rows,
        dout,
        route_tokens,
        dout,
        out,
        weight_block_expert,
        expert_col_off,
        counts,
        expert_row_start,
    )


def cute_gate_up_rows(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    row_w: torch.Tensor,
    route_tokens: torch.Tensor,
    tile_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    expert_nblocks: torch.Tensor,
    width_block: int,
) -> None:
    """Compute g, u and h = row_w * silu(g) * u, reading x once."""
    tensors = (
        x,
        w_gate,
        w_up,
        g,
        u,
        h,
        row_w,
        route_tokens,
        tile_expert,
        expert_col_off,
        expert_nblocks,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("CuTe projection tensors must be contiguous CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("CuTe projection tensors must share a device")
    if torch.cuda.get_device_capability(x.device)[0] != 8:
        raise ValueError("CuTe Ampere projection requires compute capability 8.x")
    if any(t.dtype != torch.bfloat16 for t in tensors[:6]):
        raise TypeError("CuTe Ampere projection currently supports BF16 only")
    if row_w.dtype != torch.float32:
        raise TypeError("row_w must be float32")
    if any(t.dtype != torch.int32 for t in tensors[7:]):
        raise TypeError("routing metadata tensors must be int32")
    if g.shape[0] != tile_expert.numel() * 128:
        raise ValueError("tile_expert must contain one entry per 128 rows")
    if g.shape != u.shape or g.shape != h.shape or w_gate.shape != w_up.shape:
        raise ValueError("g, u, h and the two weights must have matching shapes")
    if x.shape[1] != w_gate.shape[0] or x.shape[1] % 256:
        raise ValueError("hidden size must be a multiple of 256 and match the weights")
    if x.data_ptr() % 16:
        raise ValueError("x must be 16 byte aligned")

    bm, bn, bk, stages, atom = _GATE_UP_CFG
    if width_block != bn:
        raise ValueError("the width tile must equal the width block")
    signature = (
        x.device.index,
        x.dtype,
        x.shape[1],
        tuple(w_gate.shape),
        g.shape[1],
        _GATE_UP_CFG,
    )
    compiled = _GATE_UP_CACHE.get(signature)
    if compiled is None:
        args = (
            _as_cute(x, dynamic_dim0=True),
            _as_cute(w_gate),
            _as_cute(w_up),
            _as_cute(g, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(u, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(h, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(row_w, dynamic_dim0=True),
            _as_cute(route_tokens, dynamic_dim0=True),
            _as_cute(tile_expert, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
            _as_cute(expert_nblocks, dynamic_dim0=True),
        )
        op = _AmpereGateUpRows((bm, bn, bk), stages, atom)
        fn = cute.compile(op, *args, 1, _ENV_STREAM, options="--enable-tvm-ffi")
        compiled = _CompiledKernel(fn, signature)
        _GATE_UP_CACHE[signature] = compiled

    _launch(
        compiled,
        x,
        w_gate,
        w_up,
        g,
        u,
        h,
        row_w,
        route_tokens,
        tile_expert,
        expert_col_off,
        expert_nblocks,
    )


def cute_dgu_parts(max_width: int, cfg: tuple | None = None) -> int:
    """Columns of ``dw_part`` that ``cute_dgu`` writes: per width tile, one
    partial dot per N warp of the MMA atom."""
    bm, bn, bk, stages, atom = cfg or _DGU_CFG
    return (max_width + bn - 1) // bn * atom[1]


def cute_dgu(
    dout: torch.Tensor,
    w_down: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    row_w: torch.Tensor,
    dg: torch.Tensor,
    du: torch.Tensor,
    dw_part: torch.Tensor,
    h: torch.Tensor | None,
    route_tokens: torch.Tensor,
    tile_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    expert_nblocks: torch.Tensor,
    width_block: int,
    cfg: tuple | None = None,
) -> None:
    """dg, du, the routing-weight partials and optionally h, as grouped_moe's
    Triton ``_dgu`` computes them, on the current stream.

    ``dw_part`` needs ``cute_dgu_parts(g.shape[1])`` columns, zero-initialized:
    tiles past an expert's width do not run. Sum each row's columns in order.
    """
    tensors = (
        dout,
        w_down,
        g,
        u,
        row_w,
        dg,
        du,
        dw_part,
        route_tokens,
        tile_expert,
        expert_col_off,
        expert_nblocks,
    ) + (() if h is None else (h,))
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("CuTe dgu tensors must be contiguous CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("CuTe dgu tensors must share a device")
    if torch.cuda.get_device_capability(g.device)[0] != 8:
        raise ValueError("CuTe Ampere dgu requires compute capability 8.x")
    if any(t.dtype != torch.bfloat16 for t in (dout, w_down, g, u, dg, du)):
        raise TypeError("CuTe Ampere dgu currently supports BF16 only")
    if h is not None and (h.dtype != g.dtype or h.shape != g.shape):
        raise ValueError("h must match g")
    if row_w.dtype != torch.float32 or dw_part.dtype != torch.float32:
        raise TypeError("row_w and dw_part must be float32")
    if any(t.dtype != torch.int32 for t in tensors[8:12]):
        raise TypeError("routing metadata tensors must be int32")
    if g.ndim != 2 or g.shape != u.shape or g.shape != dg.shape or g.shape != du.shape:
        raise ValueError("g, u, dg, and du must be matrices of one shape")
    if route_tokens.numel() != g.shape[0] or row_w.numel() != g.shape[0]:
        raise ValueError("route_tokens and row_w need one entry per row")
    if tile_expert.numel() * 128 != g.shape[0]:
        raise ValueError("tile_expert must contain one entry per 128 rows")
    hidden = dout.shape[1]
    if w_down.shape[1] != hidden or hidden % 256:
        raise ValueError("hidden size must be a multiple of 256 and match w_down")
    if dw_part.shape != (g.shape[0], cute_dgu_parts(g.shape[1], cfg)):
        raise ValueError("dw_part must have cute_dgu_parts(max_width) columns")
    if expert_col_off.shape != expert_nblocks.shape:
        raise ValueError("expert metadata arrays must have the same shape")
    bm, bn, bk, stages, atom = cfg or _DGU_CFG
    # Expert widths are whole width blocks; a tile must not straddle two experts.
    if width_block % bn or g.shape[1] % bn:
        raise ValueError(f"the width block must be a multiple of {bn}")
    # The gather's 16-byte async copies reject a contiguous view at an unaligned
    # offset. A fresh allocation is aligned.
    if dout.data_ptr() % 16:
        dout = dout.clone()

    fuse_h = h is not None
    signature = (
        g.device.index,
        hidden,
        tuple(w_down.shape),
        g.shape[1],
        width_block,
        fuse_h,
        (bm, bn, bk, stages, tuple(atom)),
    )
    compiled = _DGU_CACHE.get(signature)
    if h is None:
        h = g  # never written
    if compiled is None:
        args = (
            _as_cute(dout, dynamic_dim0=True),
            _as_cute(w_down),
            _as_cute(g, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(u, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(row_w, dynamic_dim0=True),
            _as_cute(dg, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(du, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(dw_part, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(h, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(route_tokens, dynamic_dim0=True),
            _as_cute(tile_expert, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
            _as_cute(expert_nblocks, dynamic_dim0=True),
        )
        op = _AmpereDgu((bm, bn, bk, stages, tuple(atom)), fuse_h)
        fn = cute.compile(op, *args, width_block, _ENV_STREAM,
                          options="--enable-tvm-ffi")
        compiled = _CompiledKernel(fn, signature)
        _DGU_CACHE[signature] = compiled

    _launch(
        compiled,
        dout,
        w_down,
        g,
        u,
        row_w,
        dg,
        du,
        dw_part,
        h,
        route_tokens,
        tile_expert,
        expert_col_off,
        expert_nblocks,
    )


def cute_wgrad_gather_pair(
    dg: torch.Tensor,
    du: torch.Tensor,
    x: torch.Tensor,
    route_tokens: torch.Tensor,
    dwg: torch.Tensor,
    dwu: torch.Tensor,
    weight_block_expert: torch.Tensor,
    expert_col_off: torch.Tensor,
    counts: torch.Tensor,
    expert_row_start: torch.Tensor,
    accumulate: bool = False,
    cfg: tuple | None = None,
) -> None:
    """``dwg = x[route_tokens].T @ dg`` and ``dwu = x[route_tokens].T @ du``,
    packed by expert, in one launch.

    The outputs are ``[hidden, total_width]``, like the gate and up weights. With
    ``accumulate`` each result is added to the output's contents in fp32 and
    rounded once, for gradient accumulation; otherwise the outputs are
    overwritten.
    """
    tensors = (
        dg,
        du,
        x,
        dwg,
        dwu,
        route_tokens,
        weight_block_expert,
        expert_col_off,
        counts,
        expert_row_start,
    )
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError("CuTe weight-gradient tensors must be contiguous CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("CuTe weight-gradient tensors must share a device")
    if torch.cuda.get_device_capability(dg.device)[0] != 8:
        raise ValueError("CuTe Ampere weight gradient requires compute capability 8.x")
    if any(t.dtype != torch.bfloat16 for t in tensors[:5]):
        raise TypeError("CuTe Ampere weight gradient currently supports BF16 only")
    if any(t.dtype != torch.int32 for t in tensors[5:]):
        raise TypeError("route and expert metadata tensors must be int32")
    if any(t.ndim != 2 for t in tensors[:5]):
        raise ValueError("dg, du, x, dwg and dwu must be matrices")
    if dg.shape != du.shape or dwg.shape != dwu.shape:
        raise ValueError("dg and du, and dwg and dwu, must have matching shapes")
    if dg.shape[0] != route_tokens.numel():
        raise ValueError("route_tokens must contain one entry per dg row")
    if dwg.shape[0] != x.shape[1] or dwg.shape[0] % 256:
        raise ValueError("output hidden dimension must match x and be 256-aligned")
    if dwg.shape[1] % 128 or weight_block_expert.numel() * 128 != dwg.shape[1]:
        raise ValueError("weight_block_expert must contain one entry per 128 columns")
    if expert_col_off.shape != counts.shape or counts.shape != expert_row_start.shape:
        raise ValueError("expert metadata arrays must have the same shape")
    # The gather's 16-byte async copies reject a contiguous view at an unaligned
    # offset. A fresh allocation is aligned.
    if x.data_ptr() % 16:
        x = x.clone()

    bm, bn, bk, stages, atom = cfg or _WGRAD_PAIR_CFG
    if bn != 128 or dwg.shape[0] % bm or 128 % bk:
        raise ValueError("the width tile must be 128 and the tiles must divide")
    signature = (
        dg.device.index,
        dg.shape[1],
        x.shape[1],
        tuple(dwg.shape),
        bool(accumulate),
        (bm, bn, bk, stages, tuple(atom)),
    )
    compiled = _WGRAD_PAIR_CACHE.get(signature)
    if compiled is None:
        args = (
            _as_cute(x, dynamic_dim0=True),
            _as_cute(dg, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(du, dynamic_dim0=True, dim0_divisibility=128),
            _as_cute(dwg),
            _as_cute(dwu),
            _as_cute(route_tokens, dynamic_dim0=True),
            _as_cute(weight_block_expert, dynamic_dim0=True),
            _as_cute(expert_col_off, dynamic_dim0=True),
            _as_cute(counts, dynamic_dim0=True),
            _as_cute(expert_row_start, dynamic_dim0=True),
        )
        op = _AmpereWgradPair((bm, bn, bk, stages, tuple(atom)), bool(accumulate))
        fn = cute.compile(op, *args, _ENV_STREAM, options="--enable-tvm-ffi")
        compiled = _CompiledKernel(fn, signature)
        _WGRAD_PAIR_CACHE[signature] = compiled

    _launch(
        compiled,
        x,
        dg,
        du,
        dwg,
        dwu,
        route_tokens,
        weight_block_expert,
        expert_col_off,
        counts,
        expert_row_start,
    )
