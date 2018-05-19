import tensorflow as tf

from .utils import get_xmodule

xmodule = get_xmodule()


def binarize_rows(input_matrix, qtype=tf.int32):
    """
    Wrapper for the binarize_rows
    :param input_matrix: input matrix
    :param qtype: type used in binarization
    :return: binarized matrix
    """
    return xmodule.binarize_rows(input_matrix, qtype=qtype)
