import os
import tensorflow as tf
from configuration import ROOT_DIR


def get_xmodule():
    """
    Uses relative path to load generated library from C++ and CUDA sources
    :return: xnor module
    :raises: RuntimeError if not found
    """
    path = os.path.join(ROOT_DIR, 'operators', 'operators.so')
    if not os.path.exists(path):
        raise RuntimeError('Could not find XNOR net module in `{}`'.format(path))
    return tf.load_op_library(path)
