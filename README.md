# MegaBlocks Variable

High-performance GPU kernels for mixture-of-experts models whose experts have
different intermediate widths.

MegaBlocks Variable began as a fork of
[Databricks MegaBlocks](https://github.com/databricks/megablocks) and now
evolves as an independent project. It preserves the upstream Git history,
license, and attribution, while focusing narrowly on variable-width routing,
inference, and training.

## What is here

- A fused Triton inference path with device-side counting-sort routing and
  row tiles sized for decode and prefill.
- A differentiable grouped-MoE training path with no STK dependency whose
  forward and backward are bitwise reproducible from run to run.
- An optional CuTe DSL backend for the dominant Ampere training GEMMs.
- Score-aware nearest-block route dropping for reducing training padding,
  including `token_rounding='nearest'` in the grouped training path.
- Extension-free installation for the default Triton paths.

The implementation is intentionally compact. Legacy upstream model layers and
unused sparse operations were removed; this package is a kernel backend for
projects that own their model and routing layers.

## Performance

On an RTX 3090, the fused variable-width serving path improved measured
keep-50 Qwen3.5-MoE decode throughput from 688 to 1536 tok/s (2.21x) while
preserving greedy continuations.

For one training layer with 54 experts, 4096 tokens, hidden size 2048, and top-k
8, on an RTX 3090 at a 275 W limit with the BF16 unlock, under sustained load:

| backend | forward + backward |
|---|---:|
| Triton | 9.23 ms |
| CuTe | **8.85 ms** |

See [`KERNEL_OPTIMIZATION.md`](KERNEL_OPTIMIZATION.md) for methodology, rejected
fusions, memory tradeoffs, and shape-specific results.

## Installation

MegaBlocks Variable currently keeps the Python import/package name
`megablocks` for downstream compatibility. Install this repository explicitly;
`pip install megablocks` may resolve the upstream Databricks distribution.

```console
uv pip install "megablocks @ git+https://github.com/hbfreed/megablocks-variable.git"
```

For an editable development environment:

```console
git clone https://github.com/hbfreed/megablocks-variable.git
cd megablocks-variable
uv sync --extra dev
```

The lock file pins PyTorch 2.12 with its CUDA 13.0 wheels, the stack the
numbers above were measured on.

The fused and grouped paths need only PyTorch. `megablocks.ops` and
`megablocks.layers.relu_squared`, the older block-sparse path, also need
[STK](https://github.com/stanford-futuredata/stk): install `megablocks[stk]`.

The default Triton paths do not build CUDA extensions. Set
`MEGABLOCKS_BUILD_EXTENSIONS=1` only for legacy consumers that still need
`nanomoe_ops`; that build needs PyTorch in the build environment (for uv,
list `torch` under `[tool.uv.extra-build-dependencies]`).

### Optional CuTe training backend

Install the CuTe extra:

```console
uv sync --extra cute
```

The CuTe kernels target BF16 on Ampere (compute capability 8.x) with hidden
sizes divisible by 256, where they run 3-7% faster than Triton per layer. The
default `down_proj_backend="auto"` uses them whenever the extra is installed and
the call qualifies, and Triton otherwise, so no flag is needed. Pass `"cute"` or
`"triton"` to force one:

```python
from megablocks.backend.grouped_moe import grouped_moe

output = grouped_moe(
    x,
    top_weights,
    expert_ids,
    plan,
    top_k,
    w_gate,
    w_up,
    w_down,
    recompute_activation=False,
)
```

`recompute_activation=False` is the fastest mode but retains about 77.5 MiB
for the measured layer shape. Keep the default `True` when activation memory
is more valuable than the roughly 0.31 ms backward improvement.

## Tests

GPU operator tests are marked explicitly:

```console
pytest -m gpu tests/ops
```

The optional CuTe tests require its dependencies and Ampere hardware, and are
skipped without them. A plain `pytest` runs only the few tests that need no
GPU.

## Project history and attribution

The descriptions of the eight pull requests merged before this repository
left the GitHub fork network are preserved in
[`docs/pull-request-history.md`](docs/pull-request-history.md). All commits and
authorship remain in Git.

MegaBlocks Variable is based on the original MegaBlocks work:

```bibtex
@article{megablocks,
  title={{MegaBlocks: Efficient Sparse Training with Mixture-of-Experts}},
  author={Trevor Gale and Deepak Narayanan and Cliff Young and Matei Zaharia},
  journal={Proceedings of Machine Learning and Systems},
  volume={5},
  year={2023}
}
```

See [`LICENSE`](LICENSE) and file-level notices for licensing and attribution.
