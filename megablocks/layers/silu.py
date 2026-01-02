import stk
import torch
import torch.nn.functional as F


@torch.jit.script
def _silu_backward_inplace(g, x):
    sig = torch.sigmoid(x)
    g.mul_(sig * (1 + x * (1 - sig)))
    return g


def silu_backward_(grad: stk.Matrix, x: stk.Matrix):
    # NOTE: The two sparse matrices must have the same topology.
    if isinstance(grad, stk.Matrix) and isinstance(x, stk.Matrix):
        return stk.Matrix(
            x.size(),
            _silu_backward_inplace(grad.data, x.data),
            x.row_indices,
            x.column_indices,
            x.offsets,
            x.column_indices_t,
            x.offsets_t,
            x.block_offsets_t,
        )
    return _silu_backward_inplace(grad, x)


def silu(x: stk.Matrix):
    assert isinstance(x, stk.Matrix)
    return stk.Matrix(
        x.size(),
        F.silu(x.data),
        x.row_indices,
        x.column_indices,
        x.offsets,
        x.column_indices_t,
        x.offsets_t,
        x.block_offsets_t,
    )
