# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""MegaBlocks (variable-width expert fork).

Upstream's model-layer classes (dMoE, MoE, SparseMLP, ...) are removed: every
consumer of this fork builds its own modules and uses only ``megablocks.ops``,
``megablocks.backend`` and ``megablocks.layers.relu_squared``. Keeping this
module import-free also keeps the pure-Triton serving path
(``megablocks.backend.fused_moe``) importable without the compiled
``megablocks_ops`` extension.
"""
