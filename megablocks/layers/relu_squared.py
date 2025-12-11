import stk
import torch
import torch.nn.functional as F


@torch.jit.script
def _relu_squared_backward_inplace(g,x):
    #super simple: d/dx[relu(x)^2] = 2*relu(x)
    return g.mul_(2 * F.relu(x))

def relu_squared_backward_(grad: stk.Matrix, x: stk.Matrix):
    # NOTE: The two sparse matrices must have the same topology.
    if isinstance(grad, stk.Matrix) and isinstance(x, stk.Matrix):
        return stk.Matrix(
                x.size(),
                _relu_squared_backward_inplace(grad.data, x.data),
                x.row_indices,
                x.column_indices,
                x.offsets,
                x.column_indices_t,
                x.offsets_t,
                x.block_offsets_t,
            )
    return _relu_squared_backward_inplace(grad, x)


def relu_squared(x: stk.Matrix):
    assert isinstance(x, stk.Matrix)
    return stk.Matrix(
            x.size(),
            F.relu(x.data).square(),
            x.row_indices,
            x.column_indices,
            x.offsets,
            x.column_indices_t,
            x.offsets_t,
            x.block_offsets_t,
    )
