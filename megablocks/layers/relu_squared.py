import stk
import torch
import torch.nn.functional as F


@torch.jit.script
def _relu_squared_backward_inplace(g,x):
    #super simple: d/dx[relu(x)^2] = 2*relu(x)
    return g.mul_(2 * F.relu(x))


@torch.jit.script
def _relu_squared_fwd(x):
    r = F.relu(x)
    return r * r


@torch.jit.script
def _relu_squared_backward(g, x):
    # Non-inplace form for use inside autograd (must not mutate grad_output).
    return g * (2 * F.relu(x))


class _ReluSquared(torch.autograd.Function):
    """Fused relu(x)**2. Both forward and backward run as single fused kernels,
    avoiding the extra relu/square intermediates of F.relu(x).square()."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _relu_squared_fwd(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return _relu_squared_backward(g, x)

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
            _ReluSquared.apply(x.data),
            x.row_indices,
            x.column_indices,
            x.offsets,
            x.column_indices_t,
            x.offsets_t,
            x.block_offsets_t,
    )
