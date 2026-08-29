# Contributing to MegaBlocks Variable

Thanks for helping improve the variable-width MoE kernels. Bug reports,
performance measurements, new hardware results, and focused kernel changes are
all welcome.

This project began from
[Databricks MegaBlocks](https://github.com/databricks/megablocks), but issues
and pull requests for MegaBlocks Variable belong in this repository.

## Development environment

Create the locked development environment without building legacy CUDA
extensions:

```bash
git clone https://github.com/hbfreed/megablocks-variable.git
cd megablocks-variable
UV_TORCH_BACKEND=cu126 uv sync --extra dev
```

Add `--extra cute` when working on the optional CuTe backend. Set
`MEGABLOCKS_BUILD_EXTENSIONS=1` only when testing the legacy
`nanomoe_ops.indices_variable` compatibility extension.

Install the formatting hooks if you plan to contribute regularly:

```bash
uv run pre-commit install
```

## Making a change

1. Open an issue when the intended behavior or design needs discussion.
2. Create a focused branch from `main`.
3. Add or update tests for behavior changes.
4. Record the hardware, software stack, tensor shapes, warmup, sample count,
   and paired comparison method for performance claims.
5. Keep optional backends lazily imported so the default Triton path remains
   extension-free.

Before opening a pull request, run the checks relevant to the change:

```bash
uvx --from ruff==0.11.12 ruff check .
uv lock --check
uv run python -m compileall -q megablocks tests
uv run pytest -m gpu tests/ops
```

The full GPU suite requires CUDA. CuTe tests additionally require Ampere
hardware and the `cute` optional dependencies. If you cannot run a relevant
hardware test, say so explicitly in the pull request.

## Pull requests

A useful pull request description includes:

- the problem and why it belongs in this kernel backend;
- the affected inference/training shapes and hardware;
- correctness tolerances and test results;
- before/after timings for performance changes;
- memory, compile-time, dependency, or portability tradeoffs;
- plausible alternatives that were measured and rejected.

See [`KERNEL_OPTIMIZATION.md`](KERNEL_OPTIMIZATION.md) for the measurement style
and prior negative results. Please preserve the upstream license notices and
add attribution when adapting code from another project.
