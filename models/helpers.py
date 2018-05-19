import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MNIST_FEATURES = 784
MNIST_CLASSES = 10


@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)


def quantize(x):
    with tf.get_default_graph().gradient_override_map({"Sign": "QuantizeGrad"}):
        return tf.sign(x)


def get_mnist():
    return input_data.read_data_sets("MNIST_data", one_hot=True)
