# Variable-MoE kernel optimization

Goal: make the MoE kernels in this fork faster. Target is the variable-size
`MoEMLP` (`gpt_model.py`) exercised by `profile_moe.py --variable-only`.

**Result: 10.4 ms → 7.35 ms per fwd+bwd step (1.41×, −29%); ~790k → ~1.12M tokens/s.**
Output, all three aux losses, and every gradient are **bit-identical** to the
original (0.0 error) across four configs: variable-size experts, uniform experts,
`top_k=1`, and 512-wide experts.

Benchmark config: RTX 3090 (Ampere), bf16, batch 8 × seq 1024, hidden 768,
64 experts, ffn 256/expert, top_k 8.

## How the time was actually spent

Profiling showed the step was **launch/sync-bound**, not compute-bound. GPU-busy
time was ~5.75 ms but wall time ~10.4 ms. The tell (from the pipelining probe):

| measurement | original | optimized |
|---|---|---|
| forward-only, isolated | 3.24 ms | 2.92 ms |
| full step, serialized (CUDA events) | 8.59 ms | 7.31 ms |
| **full step, pipelined (real training loop)** | **10.44 ms** | **6.93 ms** |

In the original, the *pipelined* loop is **slower** than the *serialized* one —
a signature of a CPU/sync bottleneck: hidden host↔device syncs stall the CPU so
it can't queue the next step's work while the GPU runs. After the changes this
inverts to healthy pipelining (wall < serialized). Note the isolated forward
barely changed (3.24→2.92) while the pipelined step dropped 3.5 ms — the win is
removing serialization, not raw compute.

A per-stage forward breakdown found the culprit. **Topology construction was the
single biggest forward stage — 1.36 ms — despite producing only integer index
tensors.** Inside it, two ops that do no real math dominated:

- `torch.repeat_interleave(sizes, counts)` — **0.44 ms**, entirely a hidden device
  sync (it must learn the output length before it can allocate).
- `TopologyVarOp` `.item()` on `total_nnz` plus a 0-d-tensor→`int` conversion for
  the grid size — **~0.3 ms** of pipeline drain; the CUDA kernel itself is tiny.

## Changes

### 1. `padded_gather`: zero only the padding, not the whole buffer (`megablocks/backend/kernels.py`)
The gather output is padded per expert (rounded up to the 128 block size), so it
was allocated with `torch.zeros`. But only ~6% of the rows are padding — the copy
overwrites the other ~94% immediately. Zeroing all of it wrote ~113 MB/call for
nothing. Replaced with `torch.empty` + a tiny `_zero_padding` kernel that zeros
only the per-expert padding gaps.

Padding *must* stay zero (the backward weight-grad `dds` sums over all padded
rows and relies on `padding × grad == 0`; garbage would give `Inf×0 = NaN`), so
`torch.empty` alone would be wrong — hence the targeted zeroing.

*Isolated:* gather 0.561→0.456 ms. *Profiler:* `PaddedGatherOp` CUDA 17.5→11.9 ms.

### 2. `repeat_interleave` made sync-free (`gpt_model.py::_create_topology`)
Passed `output_size=` (the row-block count, already known) so it no longer syncs
to discover the length. *Isolated:* 0.437→0.012 ms.

### 3. `total_nnz` computed host-side (`gpt_model.py::TopologyVarOp`, `_create_topology`)
The topology build needs `padded_bins[-1]` on the host anyway (to size buffers).
We now pull `padded_bins` to the host **once** (`.tolist()`) and derive every
python-int size — last bin, row-block count, and `total_nnz` — from it, then hand
`total_nnz` to `TopologyVarOp` so it skips its own `.item()` sync. The host-side
sum is integer-exact with the device reduction it replaces. Also removes the
implicit 0-d-tensor→`int` sync by passing `block_rows` as a python int.

### 4. Deduplicate the `bins` cumsum (`gpt_model.py::forward`)
`_gather_tokens` and `_scatter_tokens` each recomputed
`inclusive_cumsum(tokens_per_expert)`. Compute it once and share it.

### 5. Copy-kernel cleanup (`megablocks/backend/kernels.py`)
Shared the autotune config list across the copy kernels and skip the
`bf16→fp32→bf16` round-trip on the non-scaling (pure-copy) path — bit-identical,
fewer instructions. **Net-neutral on measured time** here: these kernels are
memory-bandwidth-bound at hidden=768, so the copy body isn't the bottleneck (the
profiler shows `_padded_copy` unchanged at ~47 ms). Kept because it's clean and
the wider autotune space (keyed on `NUM_COLUMNS`) can help other hidden sizes.

## What was deliberately *not* done

- **Fusing the scatter's top-k reduction** (`padded_scatter` writes a
  `(tokens, top_k, hidden)` buffer then `.sum(dim=1)`; the sum is ~36% of scatter
  time). An atomic-accumulate fusion would save ~0.28 ms/step but makes training
  **non-deterministic** (bf16 atomics reorder). A deterministic fusion
  (grid-over-tokens + inverse permutation) is a sizeable new indexing kernel that
  also changes the numerics (no longer bit-identical) for ~4%. Not worth the risk
  here; noted as future work.
- **Removing the remaining `padded_gather` output-size sync.** It's genuinely
  redundant (the value is already on the host), but eliminating it means threading
  an `output_rows` argument through the shared `megablocks/ops` autograd
  functions (also used by the non-variable dMoE path). Poor risk/reward for ~0.1 ms.
- **The block-sparse GEMMs** (`stk` `sdd`/`dsd`/`dds`) dominate remaining CUDA
  time but are an external library.

## Biggest remaining opportunity

Topology construction is still the largest forward stage (~0.83 ms) and is mostly
Python glue + ~20 tiny launches (`row_indices`, a sparse transpose that re-sorts /
re-histograms, several `int32` casts). Folding the whole topology build —
column indices, row indices, offsets, and the transpose — into the existing
`nanomoe_ops` CUDA kernel would remove most of it. Larger effort; left as future
work.

## Verifying

`profile_moe.py --variable-only` for timing. Correctness was checked by capturing
the full fwd+bwd (output, aux losses, w1/w2/x/router grads) from the pristine code
and diffing against the optimized version for the four configs above — all 0.0.
(`tests/` can't run out of the box: `conftest.py` imports `composer`, which isn't
installed.)
