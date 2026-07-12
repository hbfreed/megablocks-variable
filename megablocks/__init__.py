# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

from megablocks.stk_compat import apply_stk_compatibility_fixes

apply_stk_compatibility_fixes()

from megablocks.layers.arguments import Arguments
from megablocks.layers.dmoe import ParallelDroplessMLP, dMoE
from megablocks.layers.glu import SparseGLU
from megablocks.layers.mlp import MLP, SparseMLP
from megablocks.layers.moe import MoE, ParallelMLP, get_load_balancing_loss

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
