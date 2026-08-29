# Pull request history

This file preserves the substantive descriptions of the eight pull requests
merged while `hbfreed/megablocks-variable` was attached to the
`databricks/megablocks` fork network. GitHub removes pull request metadata when
a fork leaves its network. The repository was detached on 2026-08-29 after
this archive was committed.

All eight pull requests were authored by `hbfreed`, merged, and had no review
comments or conversation comments. Their commits and merge commits remain in
the Git history.

## PR 1: Add MoE profiling script

- Merged 2026-01-13
- Branch: `claude/optimize-performance-3gc1R` into `main`
- Commit: `266b66a9edb65ea190ea7178a9e4d772e0d6d596`
- Merge commit: `8b970d6db8810d50347f510738437609e9efb569`

Added `profile_moe.py` for individual operation timing, full MoE forward and
backward timing, CPU/GPU synchronization detection, and optional TensorBoard
trace generation.

## PR 2: Upgrade to PyTorch 2.12

- Merged 2026-07-12
- Branch: `agent/upgrade-torch-2-12` into `main`
- Commit: `1d9d402bd0045034f40c58c88fd857040447fc02`
- Merge commit: `2af4aeafc7d15cdee5cf73a315c79395dbf492d9`

Upgraded the project, lockfile, and development container to PyTorch 2.12.1,
CUDA 12.6, and Triton 3.7.1. Removed obsolete Triton 2.1 and MosaicML Composer
constraints, restored modern sparse coverage, fixed the Python/CUDA topology
signature mismatch, and added an STK compatibility shim for int16 sparse
metadata overflow under modern Triton.

Validation included 58 sparse dMoE/GLU tests, 20 topology tests, high-top-k
forward and backward cases, rebuilt CUDA extensions, and compile/diff checks.

## PR 3: Add fused grouped MoE forward for variable-width experts

- Merged 2026-07-25
- Branch: `fused-grouped-moe-forward` into `main`
- Commit: `b1ea9e1bf26d19cd1f661c5d4be91c680db0f730`
- Merge commit: `bcacfd8b1217f84ac4efaa034c7d273426689c3e`

Replaced the STK block-sparse inference pipeline with a grouped GEMM for
variable-width experts. The row tile is selected from 16, 32, and 64 rather
than being fixed at STK's 128, avoiding extreme decode padding. A Triton
counting sort fused the histogram/cumsum intermediates and removed the host
synchronization required by the former CUB routing path.

On the served keep-50 Qwen3.5-MoE checkpoint (40 layers, 220 experts, PP=2,
64 sequences, 256 generated tokens), decode improved from 688 to 1536 tok/s
(2.21x), prefill improved from 0.33 to 0.27 seconds, and greedy continuations
were identical. Accuracy relative to a dense fp32 reference improved from
4.27e-3 to 2.87e-3.

The change also made public package names lazy so the pure-Triton serving path
could load without an ABI-matched CUDA extension.

## PR 4: Restore the `nanomoe_ops.indices_variable` export

- Merged 2026-07-25
- Branch: `restore-indices-variable-export` into `main`
- Commit: `c33c71539049c26d88ae43952023ebd6bc11a45b`
- Merge commit: `d90b4bbf4494df27a92d26c7411c2f89398f9390`

Restored a compatibility export removed in PR 3 after finding three external
consumers: nanoMOE, nanoMoEchat, and variable-flex-olmo. The export is a second
name for the still-live `megablocks::indices` function and therefore has no
meaningful maintenance or binary cost.

## PR 5: Add score-aware nearest-block routing

- Merged 2026-08-10
- Branch: `agent/score-aware-nearest-routing` into `main`
- Commits: `5ecb134952060f7f1d93af081f896491323872f2`,
  `d6c478828b4c7192da42e14c28eef1e7d141db35`
- Merge commit: `20213e5a4f204e8c6edf8791c0d604ec4fe57591`

Sorted routes by expert and descending router score, then allowed training to
round each expert's route count to the nearest 128-row block. Dropped routes
receive zero output and zero routing-weight gradient. The default ceiling mode
and all inference behavior remained unchanged.

Validation included 95 route/gather/scatter tests and 20 full-MoE and fused
serving tests.

## PR 6: Make fused serving installs extension-free

- Merged 2026-08-10
- Branch: `agent/fast-runtime-install` into `main`
- Commits: `863d4d145c664fe85e4bf50d090917b80d53e33d`,
  `22f3aa81a1720469590269928847cb530ae17b01`,
  `c6faf01db806f2f9c790ac37c9a2a5f8b5ee893d`
- Merge commit: `cc7a9854d854fe22ea55717a3e27dff3b5088f71`

Made CUDA training extensions opt-in through
`MEGABLOCKS_BUILD_EXTENSIONS=1`, kept pure-Triton imports independent of STK,
and retained the compatibility patch when legacy training operations load.
This allowed Winnow-vLLM to install the fused serving backend without compiling
or ABI-matching unused extensions.

## PR 7: Remove dead upstream code and the CUB extension

- Merged 2026-08-12
- Branch: `cleanup/remove-dead-upstream` into `main`
- Commits: `6d285505d203864c90f74de24642b41690c1f797`,
  `579ea065cea6cbc2b82eb398848431c284648800`,
  `64706c7c41a42cf051a74df188c68149857d4d33`
- Merge commit: `5e763d067740db065869b927518f05dd5c6eb84e`

Removed roughly 4,500 lines of unused upstream model layers, sparse operations,
benchmarks, and tests. Replaced CUB routing operations with stable
`torch.sort`, a Triton histogram kernel, and dtype-preserving `torch.cumsum`,
removing `megablocks_ops`; only the legacy `nanomoe_ops` topology extension
remained.

The PR also documented why persistent grouped-GEMM variants were reverted:
dead CTAs exit after two scalar loads, while persistent scheduling was
0.38-1.04x as fast despite a theoretical 2.0-2.6x reduction in dead-tile
launches. Validation covered 216 GPU tests plus downstream Glean and vLLM
plugin tests.

## PR 8: Add grouped variable-width MoE backward

- Merged 2026-08-12
- Branch: `grouped-backward` into `cleanup/remove-dead-upstream`
- Commits: `10412c98ed9637cf5ee98cc3c6a9b251ba110b20`,
  `c1720f05f93a41a26f1b522cba86c68d184a6a29`
- Merge commit: `c0ecbec9fc9ce258eb513940af20487c38cb751a`

Added a differentiable grouped-MoE twin of the fused serving path. The forward
stores gate/up pre-activations and the deterministic backward computes routing
weight gradients, the down-input/SwiGLU derivative, all three weight
gradients, and the input gradient with fp32 tile accumulators and fixed
reduction order.

All gradients were within roughly 4e-3 of a dense fp32 autograd reference. A
subsequent tile sweep reduced the real keep-50 training step from 22.5 ms to
roughly 16.9 ms, within 5-8% of the legacy STK path at the measured shapes.
The initial implementation passed eight grouped-MoE cases and the full fork
suite passed 224 tests.

Further Triton/CuTe fusion and tuning performed after PR 8 is documented in
[`KERNEL_OPTIMIZATION.md`](../KERNEL_OPTIMIZATION.md).
