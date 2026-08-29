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
- A differentiable grouped-MoE training path with deterministic forward and
  backward kernels and no STK dependency.
- An optional CuTe DSL backend for the dominant Ampere training GEMMs.
- Score-aware nearest-block route dropping for reducing training padding.
- Extension-free installation for the default Triton paths.

The implementation is intentionally compact. Legacy upstream model layers and
unused sparse operations were removed; this package is a kernel backend for
projects that own their model and routing layers.

## Performance

On an RTX 3090, the fused variable-width serving path improved measured
keep-50 Qwen3.5-MoE decode throughput from 688 to 1536 tok/s (2.21x) while
preserving greedy continuations.

For one keep-50 training layer with 4096 tokens, hidden size 2048, and top-k 8,
150 rotated measurements on the locked Torch 2.12.1 / CUDA 12.6 stack gave:

| backend | forward + backward |
|---|---:|
| tuned Triton | 16.491 ms |
| CuTe, recompute activation | 15.626 ms |
| CuTe, retain activation | **15.264 ms** |

The CuTe path with retained activation was 7.1% faster than the final Triton
path in paired measurements and roughly 10% faster than the pre-optimization
training implementation. See
[`KERNEL_OPTIMIZATION.md`](KERNEL_OPTIMIZATION.md) for methodology, rejected
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
UV_TORCH_BACKEND=cu126 uv sync --extra dev
```

The default Triton paths do not build CUDA extensions. Set
`MEGABLOCKS_BUILD_EXTENSIONS=1` only for legacy consumers that still need
`nanomoe_ops.indices_variable`.

### Optional CuTe training backend

Install the CuTe extra:

```console
UV_TORCH_BACKEND=cu126 uv sync --extra cute
```

The current CuTe implementation targets BF16 on Ampere (compute capability
8.x) with hidden sizes divisible by 256. Select it explicitly:

```python
output = grouped_moe(
    x,
    top_weights,
    expert_ids,
    plan,
    top_k,
    w_gate,
    w_up,
    w_down,
    down_proj_backend="cute",
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

The Triton and CuTe training changes were validated with 226 single-GPU
operator cases. The optional CuTe tests require its dependencies and Ampere
hardware.

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
