# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""MegaBlocks (variable-width expert fork).

The public names below are resolved lazily (PEP 562). Importing them eagerly
would pull in ``megablocks.layers.dmoe`` -> ``megablocks.ops`` -> the
``megablocks_ops`` CUDA extension, which means *any* import of this package --
including ``megablocks.backend.fused_moe``, which is pure Triton and uses none
of it -- would require an ABI-matched compiled extension to be present.

Serving depends only on the Triton path, so this keeps that import free of the
extension entirely. ``from megablocks import ops`` and
``from megablocks.layers... import ...`` still work exactly as before: the
submodule import machinery falls through to them when ``__getattr__`` declines.
"""

import importlib
from typing import Any

from megablocks.stk_compat import apply_stk_compatibility_fixes

apply_stk_compatibility_fixes()

_LAZY = {
    'Arguments': 'megablocks.layers.arguments',
    'ParallelDroplessMLP': 'megablocks.layers.dmoe',
    'dMoE': 'megablocks.layers.dmoe',
    'SparseGLU': 'megablocks.layers.glu',
    'MLP': 'megablocks.layers.mlp',
    'SparseMLP': 'megablocks.layers.mlp',
    'MoE': 'megablocks.layers.moe',
    'ParallelMLP': 'megablocks.layers.moe',
    'get_load_balancing_loss': 'megablocks.layers.moe',
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        # Must raise, not return None: the submodule import machinery relies on
        # AttributeError to fall through for `from megablocks import ops`.
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # resolve once
    return value


def __dir__() -> list:
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    'MoE',
    'dMoE',
    'get_load_balancing_loss',
    'ParallelMLP',
    'ParallelDroplessMLP',
    'SparseMLP',
    'MLP',
    'SparseGLU',
    'Arguments',
]
