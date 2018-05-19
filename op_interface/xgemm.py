from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from .utils import get_xmodule

xmodule = get_xmodule()
xgemm = xmodule.xgemm


@ops.RegisterGradient("XGEMM")
def _xgemm_grad(op, grad):
    """
    Gradient computation for the XGEMM
    :param op: XGEMM operation that is differentiated
    :param grad: gradient with respect to the output of XGEMM
    :return: gradients with respect to the input matrices of the XGEMM
    """
    a = op.inputs[0]
    b = op.inputs[1]
    grad_a = math_ops.matmul(grad, b, transpose_b=True)
    grad_b = math_ops.matmul(a, grad, transpose_a=True)
    return grad_a, grad_b
