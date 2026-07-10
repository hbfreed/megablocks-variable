#include <cstdint>
#include <c10/util/Half.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define CUDA_CALL(code)					    \
  do {                                                      \
    cudaError_t status = code;                              \
    std::string err = cudaGetErrorString(status);           \
    TORCH_CHECK(status == cudaSuccess, err);		    \
  } while (0)

namespace megablocks {
namespace construct_indices {

// We expect the number of outputs per block to be small. For
// example, with ffn_hidden_size=4096, we only need to write
// 32 elements per block per iteration.
const int kThreadsPerBlock = 32;

__global__ void __launch_bounds__(kThreadsPerBlock)
  ConstructIndicesKernel(short * __restrict__ indices,
			 const int * __restrict__ expert_block_counts,  // Number of blocks per expert
			 const int * __restrict__ expert_block_offsets, // Cumulative offsets into weight matrix
			 const int * __restrict__ output_offsets,        // Cumulative offsets into output array
			 int block_size,
			 const int * __restrict__ padded_bins) {
  // Load the offset for this bins indices.
  int start = 0;
  if (blockIdx.x > 0) start = __ldg(padded_bins + blockIdx.x - 1);
  int end = __ldg(padded_bins + blockIdx.x);

  // Divide the start and end into blocks.
  start /= block_size;
  end /= block_size;

  // Load expert-specific block count and offset
  int expert_id = blockIdx.x;
  int blocks_for_this_expert = __ldg(expert_block_counts + expert_id);
  int expert_base_offset = __ldg(expert_block_offsets + expert_id);

  // NEW: Use pre-computed output offset instead of calculating it
  int output_start = __ldg(output_offsets + expert_id);

  // Offset the output buffer to the start of this expert's output
  indices += output_start + blockIdx.y * blocks_for_this_expert + threadIdx.x;

  // Write the indices to the output.
  int bin_offset = blockIdx.y;
  int num_rows = end - start;
  for (; bin_offset < num_rows; num_rows -= gridDim.y) {
    short *out = indices;
    for (int bid = threadIdx.x; bid < blocks_for_this_expert; bid += kThreadsPerBlock) {
      *out = expert_base_offset + bid;
      out += kThreadsPerBlock;
    }
    indices += gridDim.y * blocks_for_this_expert;
  }
}

cudaError_t ConstructIndices(short * __restrict__ indices,
			     int output_block_rows,
			     const int * __restrict__ expert_block_counts,
			     const int * __restrict__ expert_block_offsets,
			     const int * __restrict__ output_offsets,
			     int block_size,
			     const int * __restrict__ padded_bins,
			     int num_bins,
			     cudaStream_t stream) {
  dim3 block_dim(kThreadsPerBlock);
  dim3 grid_dim(num_bins, (int)std::ceil((float)output_block_rows / num_bins));
  ConstructIndicesKernel<<<grid_dim, block_dim, 0, stream>>>(indices,
							     expert_block_counts,
							     expert_block_offsets,
							     output_offsets,
							     block_size,
							     padded_bins);
  return cudaGetLastError();
}

}  // namespace construct_indices

namespace construct_topology {

// One block per expert. Each expert owns a contiguous slice of every output
// array, so there are no cross-block races and no sort is needed: the variable
// MoE topology is block-diagonal-dense (every one of an expert's token-blocks
// connects to all of that expert's weight column-blocks).
//
// Per expert e (0-indexed), with:
//   tb[e] = token-blocks, sb[e] = weight column-blocks,
//   cbo[e] = first column-block, R[e] = first row-block,
//   oo[e] = first non-zero-block (== output offset),
// the forward CSR is laid out (t outer, k inner) and the transpose CSC is laid
// out (k outer, t inner) -- see the closed-form derivation in gpt_model.py.
const int kTopoThreads = 256;

__global__ void __launch_bounds__(kTopoThreads)
  BuildTopologyKernel(const int * __restrict__ tb,
                      const int * __restrict__ sb,
                      const int * __restrict__ cbo,
                      const int * __restrict__ R,
                      const int * __restrict__ oo,
                      int block_rows,
                      int total_col,
                      int total_nnz,
                      int * __restrict__ column_indices,
                      int * __restrict__ row_indices,
                      int * __restrict__ offsets,
                      int * __restrict__ column_indices_t,
                      int * __restrict__ offsets_t,
                      int * __restrict__ block_offsets_t) {
  int e = blockIdx.x;
  int tbe = __ldg(tb + e);
  int sbe = __ldg(sb + e);
  int cboe = __ldg(cbo + e);
  int Re = __ldg(R + e);
  int ooe = __ldg(oo + e);
  int eo = tbe * sbe;

  // Forward CSR block indices: i = t * sbe + k.
  for (int i = threadIdx.x; i < eo; i += blockDim.x) {
    int t = i / sbe;
    int k = i - t * sbe;
    column_indices[ooe + i] = cboe + k;
    row_indices[ooe + i] = Re + t;
  }

  // Transpose CSC block indices: j = k * tbe + t.
  for (int j = threadIdx.x; j < eo; j += blockDim.x) {
    int k = j / tbe;
    int t = j - k * tbe;
    column_indices_t[ooe + j] = Re + t;
    block_offsets_t[ooe + j] = ooe + t * sbe + k;
  }

  // CSR row pointers for this expert's row-blocks.
  for (int t = threadIdx.x; t < tbe; t += blockDim.x) {
    offsets[Re + t] = ooe + t * sbe;
  }

  // CSC column pointers for this expert's column-blocks.
  for (int k = threadIdx.x; k < sbe; k += blockDim.x) {
    offsets_t[cboe + k] = ooe + k * tbe;
  }

  // The last expert closes both pointer arrays with the sentinel value.
  if (e == gridDim.x - 1 && threadIdx.x == 0) {
    offsets[block_rows] = total_nnz;
    offsets_t[total_col] = total_nnz;
  }
}

cudaError_t BuildTopology(const int * tb,
                          const int * sb,
                          const int * cbo,
                          const int * R,
                          const int * oo,
                          int num_experts,
                          int block_rows,
                          int total_col,
                          int total_nnz,
                          int * column_indices,
                          int * row_indices,
                          int * offsets,
                          int * column_indices_t,
                          int * offsets_t,
                          int * block_offsets_t,
                          cudaStream_t stream) {
  BuildTopologyKernel<<<num_experts, kTopoThreads, 0, stream>>>(
      tb, sb, cbo, R, oo, block_rows, total_col, total_nnz,
      column_indices, row_indices, offsets,
      column_indices_t, offsets_t, block_offsets_t);
  return cudaGetLastError();
}

}  // namespace construct_topology

// Builds all six block-sparse topology arrays in a single kernel launch,
// replacing the CUDA indices + stk.row_indices + sort-based transpose path.
void build_topology(torch::Tensor tb,
                    torch::Tensor sb,
                    torch::Tensor cbo,
                    torch::Tensor row_offsets,
                    torch::Tensor nnz_offsets,
                    int block_rows,
                    int total_col,
                    int total_nnz,
                    torch::Tensor column_indices,
                    torch::Tensor row_indices,
                    torch::Tensor offsets,
                    torch::Tensor column_indices_t,
                    torch::Tensor offsets_t,
                    torch::Tensor block_offsets_t) {
  for (auto t : {tb, sb, cbo, row_offsets, nnz_offsets, column_indices,
                 row_indices, offsets, column_indices_t, offsets_t,
                 block_offsets_t}) {
    TORCH_CHECK(t.is_cuda());
    TORCH_CHECK(t.scalar_type() == torch::kInt);
    TORCH_CHECK(t.is_contiguous());
  }
  int num_experts = tb.numel();
  if (num_experts == 0 || total_nnz == 0) return;

  CUDA_CALL(construct_topology::BuildTopology(tb.data_ptr<int>(),
                                          sb.data_ptr<int>(),
                                          cbo.data_ptr<int>(),
                                          row_offsets.data_ptr<int>(),
                                          nnz_offsets.data_ptr<int>(),
                                          num_experts,
                                          block_rows,
                                          total_col,
                                          total_nnz,
                                          column_indices.data_ptr<int>(),
                                          row_indices.data_ptr<int>(),
                                          offsets.data_ptr<int>(),
                                          column_indices_t.data_ptr<int>(),
                                          offsets_t.data_ptr<int>(),
                                          block_offsets_t.data_ptr<int>(),
                                          c10::cuda::getCurrentCUDAStream()));
}

void indices(torch::Tensor padded_bins,
	     torch::Tensor expert_block_counts,
	     torch::Tensor expert_block_offsets,
	     torch::Tensor output_offsets,
	     int block_size,
	     int output_block_rows,
	     torch::Tensor out) {
  TORCH_CHECK(padded_bins.is_cuda());
  TORCH_CHECK(padded_bins.ndimension() == 1);
  TORCH_CHECK(padded_bins.scalar_type() == torch::kInt);

  TORCH_CHECK(expert_block_counts.is_cuda());
  TORCH_CHECK(expert_block_counts.ndimension() == 1);
  TORCH_CHECK(expert_block_counts.scalar_type() == torch::kInt);

  TORCH_CHECK(expert_block_offsets.is_cuda());
  TORCH_CHECK(expert_block_offsets.ndimension() == 1);
  TORCH_CHECK(expert_block_offsets.scalar_type() == torch::kInt);

  TORCH_CHECK(output_offsets.is_cuda());
  TORCH_CHECK(output_offsets.ndimension() == 1);
  TORCH_CHECK(output_offsets.scalar_type() == torch::kInt);

  TORCH_CHECK(out.is_cuda());
  TORCH_CHECK(out.ndimension() == 1);
  TORCH_CHECK(out.scalar_type() == torch::kInt16);

  // Exit early if there is no work to do.
  if (out.numel() == 0) return;

  CUDA_CALL(construct_indices::ConstructIndices(out.data_ptr<short>(),
						output_block_rows,
						expert_block_counts.data_ptr<int>(),
						expert_block_offsets.data_ptr<int>(),
						output_offsets.data_ptr<int>(),
						block_size,
						padded_bins.data_ptr<int>(),
						padded_bins.numel(),
						c10::cuda::getCurrentCUDAStream()));
}

}  // namespace megablocks
