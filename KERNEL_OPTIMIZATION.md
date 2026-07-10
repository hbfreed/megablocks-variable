# Variable-MoE: correctness fix + kernel optimization

Work on the variable-size `MoEMLP` (`gpt_model.py`) exercised by
`profile_moe.py --variable-only`. Two independent outcomes:

1. **Correctness:** fixed a latent bug that produced **wrong expert weight
   gradients** for any expert wider than 128 (i.e. every real config).
2. **Speed:** **10.4 ms → 6.85 ms per fwd+bwd step (1.52×, −34%)**; ~790k →
   ~1.20M tokens/s.

Benchmark config: RTX 3090 (Ampere), bf16, batch 8 × seq 1024, hidden 768,
64 experts, ffn 256/expert, top_k 8.

---

## 1. Correctness bug: corrupted expert weight gradients

`transpose_sort_end_bit` was sized to `ceil(log2(num_experts))`. But the sparse
**transpose** sorts *column-block* indices, whose range is the number of
column-blocks = `total_expert_width / block_size`, not `num_experts`. Once an
expert spans more than one 128-block, that range exceeds `num_experts`, so the
radix sort ran with too few bits and **silently dropped the high bits**,
mis-grouping the transpose. The transpose feeds the `dds` op that computes the
expert weight gradients, so `w1.grad` / `w2.grad` came out wrong.

It was **silent**: the forward output and the input gradient (`x.grad`) don't use
the transpose, so they stayed correct — loss still decreased and the router / rest
of the network trained fine. Only the experts themselves got bad gradients.

Detected by comparing against a dense fp32 ground-truth MoE:

| tensor | before | after |
|---|---|---|
| output | 2e-3 ✓ | 2e-3 ✓ |
| x.grad | 2e-3 ✓ | 2e-3 ✓ |
| **w1.grad** | **1.13 (garbage)** | **2e-3 ✓** |
| **w2.grad** | **1.26 (garbage)** | **2e-3 ✓** |

(The ~2e-3 floor is Ampere TF32 matmul, not error.) Confirmed it triggers only
when experts are wider than one block: `(8,128)` experts are correct, `(4,256)`
are not. Upstream `megablocks/layers/dmoe.py` sizes this correctly
(`ffn_hidden_size * num_experts // blocking`); the fork regressed it when adapting
to variable sizes. The fused topology kernel below removes the sort entirely, so
the final code is correct by construction.

> The same bug exists in **variable-flex-olmo** (`megablocks_core.py`). **variable-reap**
> uses HF OLMoE looped experts (not this block-sparse path) and is unaffected.

## 2. The step was sync/launch-bound, not compute-bound

GPU-busy time was ~5.75 ms but wall ~10.4 ms. In the original, the *pipelined*
training loop (10.4 ms) was **slower** than a *serialized* isolated step (8.6 ms) —
the signature of hidden host↔device syncs starving the CPU. Topology construction
was the biggest forward stage (1.36 ms) despite only building integer index
tensors, dominated by two ops that do no math:

- `torch.repeat_interleave(sizes, counts)` — a hidden device sync (learns the
  output length before allocating).
- `TopologyVarOp`'s `.item()` on `total_nnz` — pipeline drain; the kernel is tiny.

## 3. Changes

### a. Fixed the weight-grad bug (§1)
Immediate fix sized `transpose_sort_end_bit` to the column-block count; superseded
by the fused kernel (c) which does no sort.

### b. Removed host↔device syncs
- `_create_topology` now pulls `padded_bins` to the host **once** (`.tolist()`) and
  derives every python-int size (last bin, row-block count, `total_nnz`) from it.
- Deduplicated the `bins` cumsum shared by gather and scatter.

### c. Fused topology kernel (`csrc/indices.h::build_topology`)
The variable-MoE topology is **block-diagonal-dense per expert** — each of an
expert's token-blocks connects to *all* of that expert's weight column-blocks — so
all six block-sparse arrays (`column_indices`, `row_indices`, `offsets`,
`column_indices_t`, `offsets_t`, `block_offsets_t`) have a closed form. One CUDA
kernel (one block per expert, no cross-block races) emits all six in a single
launch, replacing `indices_variable` + `stk.ops.row_indices` + the
`repeat_interleave`/`cumsum` glue + the sort-based transpose.

Isolated topology build **0.95 ms → 0.33 ms (2.9×)**; verified bit-identical to
the corrected sort-based path across four configs (variable / uniform / top_k=1 /
512-wide). Rebuild with `python setup.py build_ext --inplace` (or `pip install -e .`).

### d. `padded_gather`: zero only the padding
The padded gather output was `torch.zeros` (~113 MB/call), but the copy overwrites
~94% of it. Now `torch.empty` + a `_zero_padding` kernel that zeros only the
per-expert padding gaps. Padding *must* be zero (the backward weight-grad sums over
padded rows, relying on `padding × grad == 0`). Gather 0.56 → 0.46 ms.

### e. Copy-kernel cleanup
Shared autotune list + skip the fp32 round-trip on pure copies. Net-neutral
(these kernels are bandwidth-bound) but bit-identical; kept for clarity.

## Results

| stage (fwd, isolated) | original | final |
|---|---|---|
| topology | 1.36 ms | 0.48 ms |
| gather | 0.47 ms | 0.42 ms |
| everything else | ~same | ~same |
| **full step (pipelined)** | **10.4 ms** | **6.85 ms** |

The pipelined-vs-serialized gap inverted from pathological (wall > serial) to
healthy (wall < serial): the model is no longer launch-bound.

Remaining CUDA time is dominated by the block-sparse GEMMs (`stk` `sdd`/`dsd`/
`dds`), which are an external library.

## Verifying

Timing: `profile_moe.py --variable-only`. Correctness: a dense fp32 expert-loop
reference (see the bug table) — **not** the old pristine baseline, which is buggy.
Full fwd+bwd (output, aux losses, all grads) is bit-identical to the corrected
sort-based path. (`tests/` can't run out of the box: `conftest.py` imports
`composer`, which isn't installed.)
