# Variable-MoE: correctness fix + kernel optimization

Work on the variable-size `MoEMLP` (`gpt_model.py`) exercised by
`profile_moe.py --variable-only`. Two independent outcomes:

1. **Correctness:** fixed a latent bug that produced **wrong expert weight
   gradients** for any expert wider than 128 (i.e. every real config).
2. **Speed:** **10.4 ms → 6.0 ms per fwd+bwd step (1.73×, −42%)**; ~790k →
   ~1.37M tokens/s.
3. **PyTorch 2.12 follow-up:** **5.58 ms → 4.95 ms pipelined** (another
   1.13×, -11.3%). The variable path is now about 6% faster than the sparse
   default on the same RTX 3090 benchmark instead of being slower.

Benchmark config: RTX 3090 (Ampere), bf16, batch 8 × seq 1024, hidden 768,
64 experts, ffn 256/expert, top_k 8.

### PyTorch 2.12 follow-up

After the PyTorch/STK upgrade, profiling exposed two redundant synchronizations
and a launch-heavy topology-preparation path that were hidden by the original
bottlenecks:

- `_create_topology` already copied `padded_bins` to the host, but forward gather
  read its last element back again. Scatter backward repeated that read despite
  already saving the required output shape. Passing those known Python sizes
  through reduced device-to-host copies from three per step to one.
- `BuildTopologyKernel` now derives token-block, row, and nonzero offsets directly
  from `padded_bins`. This removes the device `diff`, multiply, two scans, two
  concatenations, and two zero-fills that previously prepared its inputs.
- Router expert IDs are converted to int32 before the CUB radix sort, matching the
  default dMoE path and halving routing key/value metadata traffic.
- Scatter builds the original-route-to-padded-row map directly in Triton. The
  reducer performs one mapping load instead of reconstructing an inverse with
  `arange`, an int64 conversion, indexed assignment, and four metadata reads.
- The expert-frequency denominator uses the statically known route count instead
  of launching a reduction.

Clean measurements (100 synchronized iterations; a separate 100-iteration batch
for the pipelined number):

| implementation | synchronized | pipelined |
|---|---:|---:|
| variable, before follow-up | 5.98 ms | 5.58 ms |
| variable, final | **5.37 ms** | **4.95 ms** |
| default dMoE, sparse STK | 5.65 ms | 5.25 ms |

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

### e. Fused scatter + top-k reduction (`_scatter_reduce`)
`padded_scatter` wrote a `(tokens, top_k, hidden)` buffer (**96 MiB**) and then
`.sum(dim=1)` over it — a whole extra write + read (~33% of scatter). Replaced with
a kernel that, per output token, accumulates that token's top_k scattered rows
directly in fp32 registers (using `inv = inverse(indices)` to locate each source
row) and writes `(tokens, hidden)`. Only `top_k > 1` uses it. Scatter
**0.44 → 0.245 ms (1.8×)**. The no-weights path is bit-identical; the weighted path
differs only at bf16 epsilon and is slightly *more* accurate (fp32 accumulation).

### f. Fused `relu_squared` activation
`F.relu(x).square()` ran as separate eager kernels (+ their autograd) over the
sparse block data. Wrapped in a custom autograd Function with single fused (jit)
forward/backward kernels. Bit-identical; isolated fwd+bwd **0.67 → 0.41 ms (1.6×)**.

### g. Copy-kernel cleanup + no-op removal
Shared autotune list, skip the fp32 round-trip on pure copies (net-neutral,
bandwidth-bound), and dropped an identity `rearrange` in the compute-loss term.

## Results

| stage (fwd, isolated) | original | final |
|---|---|---|
| topology | 1.36 ms | 0.49 ms |
| scatter | 0.47 ms | 0.27 ms |
| activation | 0.19 ms | 0.13 ms |
| gather | 0.47 ms | 0.43 ms |
| sdd / dsd (external GEMMs) | ~1.1 ms | ~1.1 ms |
| **full step (pipelined)** | **10.4 ms** | **6.0 ms** |

The pipelined-vs-serialized gap inverted from pathological (wall > serial) to
healthy (wall < serial): the model is no longer launch-bound. Remaining time is
dominated by the block-sparse GEMMs (`stk` `sdd`/`dsd`/`dds`), which are external.
The gather's residual cost is scattered reads (fundamental); further wins would
need `torch.compile` of the routing/aux-loss glue, which `stk`'s custom autograd
currently blocks.

---

# Serving: fused grouped path (`megablocks/backend/fused_moe.py`)

The STK path is efficient for training. Its GEMMs use 75% to 80% of the RTX
3090 BF16 peak. Its copy kernels use approximately 930 GB/s.

The STK path is not efficient for decode. It pads each active expert to 128
rows. A decode step usually sends only one to three tokens to an expert. This
causes unnecessary work.

The measurements used one pruned Qwen3.5-MoE layer. The model has 220 experts,
a hidden size of 2048, a top-k value of 8, and expert widths from 128 to 512.
The STK path had the following padding ratios:

- 128 times at one token.
- 49 times at 64 tokens.
- 13.8 times at 256 tokens.

CPU dispatch was also a large part of the decode time. At one token, the layer
used 1075 microseconds. Host enqueue work used approximately 1045 microseconds.

`fused_moe.py` uses a grouped GEMM for this workload. It uses a 16-row tile
when the route count is small. It does not build an STK topology. It uses four
kernel launches and does not synchronize with the host. It calculates buffer
bounds from the token count and the top-k value.

| tokens | block-sparse | fused | speedup | CPU dispatch |
|---:|---:|---:|---:|---:|
| 1 | 1.33 ms | 0.67 ms | **1.97×** | 1292 → 646 µs |
| 64 | 2.60 ms | 1.44 ms | **1.80×** | 2163 → 669 µs |
| 256 | 2.88 ms | 1.64 ms | **1.75×** | 2460 → 795 µs |
| 4096 | 5.15 ms | 4.58 ms | 1.13× | 5103 → 3074 µs |

These measurements use `VariableQwenMoE.forward` on an RTX 3090. At 64 tokens,
the estimated 40-layer GPU time decreased from 104 ms to 58 ms. The CPU
dispatch time decreased from 86 ms to 27 ms.

The fused path had a relative error of 2.87e-3 against the dense FP32
reference. The block-sparse path had a relative error of 4.27e-3.

`tests/ops/fused_moe_test.py` verifies single-token, prefill, top-k=1, and
top-k equal to the expert count. Tests also covered 120 adversarial routing
distributions. Compute Sanitizer reported zero memory errors.

The fused path is for inference only. It does not implement backward
operations. The static buffer bounds make CUDA graph integration possible,
but the application must provide and test the graph wrapper.

## Verifying

Timing: `profile_moe.py --variable-only`. Correctness: a dense fp32 expert-loop
reference (see the bug table) — **not** the old pristine baseline, which is buggy.
Full fwd+bwd (output, aux losses, all grads) is bit-identical to the corrected
sort-based path. (`tests/` can't run out of the box: `conftest.py` imports
`composer`, which isn't installed.)

## Persistent-kernel experiment (2026-08-12, negative result)

Hypothesis: `_gate_up_silu` launches a rectangular `(m_tiles, max_nblocks)`
grid and relies on early-exit for dead tiles, because the grid must be
host-sized while the real work count is device-resident. On the served keep50
checkpoint (16 layers, ~55 experts, widths 128-1024, top_k 8) the over-launch
is 2.0-2.6x for gate/up and 1.3-1.6x for down. A persistent kernel (fixed
grid, CTAs stride over the work) should recover that waste.

Two variants, measured on one RTX 3090 at the served shape:

1. Stride over row tiles, inner loop over the expert's width blocks.
   0.38x at 1 token, 0.68x at 2048. The inner loop serializes work that the
   2D grid ran as parallel CTAs; at decode the SMs are idle, so the lost
   parallelism costs far more than the dead CTAs did.
2. Stride over the flat `(row tile, width block)` index space, dead pairs
   skipped in-loop. 0.97-1.04x for 1-256 tokens, 0.67x at 2048 tokens
   (the outer loop defeats Triton's software pipelining of the K loop, and
   the capped grid gives up wave oversubscription).

Conclusion: dead CTAs cost almost nothing here. They exit after two scalar
loads, and the kernel is weight-bandwidth-bound, so the 2D grid with
early-exit is the right design on Ampere. Do not retry persistence for its
own sake; it becomes interesting again only with TMA/warp-specialization
(Hopper+) or if a megakernel fuses gate/up -> down through shared memory.

Both variants passed the full `fused_moe_test.py` matrix (25 cases) before
being reverted; correctness was not the problem.

## CuTe DSL Ampere training GEMMs (2026-08-28)

The grouped training backend is already pure Triton, so CuTe DSL was first
tested on one self-contained hot GEMM before considering a wider rewrite. The
initial candidate replaced `_down_proj` and consumed the same device-side
ragged plan. One launch covers every expert row tile; expert widths remain
runtime values and no routing metadata is copied to the host.

The stock CuTe Ampere tile (128x128x32, four stages) was slower than the tuned
Triton kernel: about 1.65 ms versus 1.53 ms. A 128x256x32 CTA with an
`(1, 8, 1)` MMA atom layout and three pipeline stages reversed the result. A
later interleaved cross-shape sweep refined the atom layout to `(2, 4, 1)` for
a further small improvement. On the keep50 width tables at 4096 tokens, hidden
size 2048, and top-k 8:

| routing pattern | Triton | CuTe | CuTe / Triton |
|---|---:|---:|---:|
| uniform, four sampled layers | 1.52-1.79 ms | 1.50-1.76 ms | 0.982-0.986 |
| concentrated on eight widest experts | 2.20-2.30 ms | 2.14-2.24 ms | 0.972-0.977 |
| concentrated on eight narrowest experts | 0.50-0.80 ms | 0.48-0.77 ms | 0.954-0.972 |

The BF16 outputs were bit-identical to Triton. The complete grouped forward was
also bit-identical and improved by roughly 0.5-2% (5.04 ms versus 5.07 ms at
the median of six paired runs). Dynamic row and expert dimensions let a small
compile cache serve all 16 layer layouts; distinct maximum-width variants
compile separately. TVM-FFI accepts PyTorch tensors directly, reducing
measured host launch overhead from the initial DLPack prototype's roughly 84
microseconds to roughly 9 microseconds.

The same pipeline was then extended to the backward input projection and the
down-projection weight gradient. The retained training changes are:

- Fuse the up projection's SwiGLU epilogue in Triton: 1.781 ms for projection
  plus activation became 1.541 ms (13.5% faster).
- Combine the two Triton input-gradient streams into one accumulator as the
  dependency-free fallback: 4.334 ms became 3.683 ms (15.0% faster).
- For the CuTe backend, stream the gate and up input-gradient operands through
  one 128x256x32 kernel and one fp32 accumulator. This writes one bf16 row
  buffer and uses the normal scatter, rather than launching two kernels and
  adding a second buffer inside the scatter. The fused kernel measured about
  2.95 ms; the former pair measured about 3.05 ms before its wider scatter.
- Compute `dW_down` with a 128x256x64 CuTe tile, two pipeline stages, and a
  `(2, 4, 1)` atom layout. The isolated keep50 kernel measured about 1.49 ms
  versus 1.85-1.94 ms in Triton, and was bit-identical. The final K64 tile was
  another roughly 0.01 ms faster than the original K32 CuTe version.
- Retune the gate projection independently from the fused up+SwiGLU projection.
  Gate uses a 128x128x16, four-warp, two-stage tile; up retains
  128x128x32. The gate kernel dropped by about 0.05 ms.

The final native tuning pass swept the important kernels over the real width
tables rather than optimizing only layer 0. In particular, 120 candidate
configurations for each gate/up weight-gradient GEMM were checked, then the
leaders were interleaved across layers 0, 3, 7, 10, and 15 under uniform,
widest-expert, and narrowest-expert routing. The retained 128x64x32,
four-warp, two-stage tile reduced a complete Triton step by a paired median
0.157 ms and a complete CuTe step by 0.235 ms. An equivalent `_dgu` search
confirmed its existing 128x128x64 tile. Sweeps of the small route, scatter,
padding-zero, and count reductions found only microsecond-scale noise, so their
launches were left unchanged.

On the layer-0 math keep50 table (54 live experts, total width 32768), 4096
tokens, hidden size 2048, and top-k 8, the final locked Torch 2.12.1 / CUDA
12.6 / CUTLASS DSL 4.6.3 stack measured:

| training path | step | forward | backward |
|---|---:|---:|---:|
| optimized Triton | 16.491 ms | 5.197 ms | 11.265 ms |
| CuTe, recompute activation | 15.626 ms | 5.133 ms | 10.491 ms |
| CuTe, retain activation | **15.264 ms** | **5.110 ms** | **10.126 ms** |

These are medians from 150 rotated runs. CuTe with the default activation
recompute won all 150 paired comparisons against Triton, with a median paired
advantage of 0.906 ms (5.5%). Passing `recompute_activation=False` won all 150
against Triton and 143/150 against recompute, improving the paired median by a
further 0.309 ms and by 1.165 ms (7.1%) over Triton. It retains another 77.5
MiB at this one-layer shape; across
16 simultaneously saved layer activations that is roughly 1.2 GiB. The
pre-change Triton step measured about 16.97 ms in the same development session,
so the fastest retained path is roughly 10% faster overall, though that
before/after figure was not paired.

Several plausible fusions were measured and reverted:

- Writing a recomputed forward activation from `_dgu` made the combined
  backward activation stage 4.7% slower (2.146 versus 2.049 ms). Retaining the
  already-computed forward activation is faster, but deliberately remains an
  explicit memory/speed option.
- CuTe's raw `dY @ W_down.T` core beat Triton (1.527 versus 1.72 ms), but its
  fused SwiGLU-derivative epilogue raised the kernel to 2.45 ms.
- Fusing gate/up weight gradients in Triton was 5.4% slower (3.231 versus
  3.065 ms). Pre-gathering `x` for two CuTe weight gradients was also slower in
  the full step by 0.107 ms and lost 97 of 100 paired samples.
- Forward routing itself was only about 25 microseconds, too small to justify a
  routing/projection megakernel with extra synchronization and register state.

Helion 1.4 was also piloted as an autotuning front end for the gate projection.
The first high-level formulation accidentally serialized width tiles and took
2.93 ms after quick tuning. Expressing expert-row and width tiles as the outer
grid and preserving the runtime width early-exit brought it close: a quick
72-second search covered 33 configurations and selected a kernel that measured
1.56 ms versus 1.47 ms for the handwritten Triton projection. A seeded
three-minute full search tried 40 configurations (10 compile failures) and
retained the same shape at 1.55 ms. Helion's autotuning interface is
convenient, but it
tunes kernels generated from Helion source rather than existing Triton or CuTe
kernels; the small remaining loss and extra dependency did not justify a
production rewrite. See the [Helion project](https://github.com/pytorch/helion)
and its [deployment/autotuning guide](https://helionlang.com/deployment_autotuning.html).

The backend remains opt-in because CUTLASS DSL is a large optional dependency
and the implementation currently supports only BF16 Ampere with hidden sizes
divisible by 256. Install `megablocks[cute]` and pass
`down_proj_backend="cute"` to `grouped_moe`; Triton remains the default.
