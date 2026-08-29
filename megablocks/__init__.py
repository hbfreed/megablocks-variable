# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0
"""MegaBlocks Variable GPU kernel package.

Upstream's model-layer classes (dMoE, MoE, SparseMLP, ...) are removed: every
consumer builds its own modules and uses only ``megablocks.ops``,
``megablocks.backend`` and ``megablocks.layers.relu_squared``. The CUB-backed
sort/histogram/cumsum extension is gone too — those ops are plain torch now —
so the only compiled piece left is ``nanomoe_ops`` (topology construction),
and only the stk training path needs it.
"""
