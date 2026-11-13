# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

try:
    from megablocks.stk_compat import apply_stk_compatibility_fixes
except ModuleNotFoundError as e:
    if e.name != 'stk':
        raise
    raise ModuleNotFoundError(
        "megablocks.ops needs stanford-stk; install 'megablocks[stk]'. "
        'megablocks.backend.fused_moe and grouped_moe work without it.',
        name='stk',
    ) from e

apply_stk_compatibility_fixes()

from megablocks.ops.cumsum import inclusive_cumsum
from megablocks.ops.histogram import histogram
from megablocks.ops.pack_route_keys import pack_route_keys
from megablocks.ops.padded_gather import padded_gather
from megablocks.ops.padded_route_mask import padded_route_mask
from megablocks.ops.padded_scatter import padded_scatter
from megablocks.ops.round_up import round_up
from megablocks.ops.sort import sort

__all__ = [
    'inclusive_cumsum',
    'histogram',
    'pack_route_keys',
    'padded_gather',
    'padded_route_mask',
    'padded_scatter',
    'round_up',
    'sort',
]
