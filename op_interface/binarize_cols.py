import tensorflow as tf

from .utils import get_xmodule

xmodule = get_xmodule()


def binarize_cols(input_matrix):
    """
    Wrapper for the binarize_rows
    :param input_matrix: input matrix
    :return: binarized matrix
    """
    return xmodule.binarize_cols(input_matrix)
