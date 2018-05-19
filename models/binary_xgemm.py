import time
from typing import Dict

import tensorflow as tf

from op_interface import xgemm
from .helpers import get_mnist, MNIST_CLASSES, MNIST_FEATURES


# Create model
def binary_xgemm_net(features: tf.placeholder, weights: Dict[str, tf.Variable]):
    # Hidden fully connected layer with 1024 neurons
    z_1 = tf.matmul(features, weights['h1'])
    # a_1_b = quantize(z_1)

    # Hidden fully connected layer with 1024 neurons
    # w_2_b = quantize(weights['h2'])
    z_2 = xgemm(z_1, weights['h2'], otype=tf.float32)
    # a_2_b = quantize(z_2)

    # Hidden fully connected layer with 1024 neurons
    # w_3_b = quantize(weights['h3'])
    z_3 = xgemm(z_2, weights['h3'], otype=tf.float32)
    # a_3_b = quantize(z_3)

    # Output fully connected layer with a neuron for each class
    # w_out_b = quantize(weights['out'])
    out_layer = tf.matmul(z_3, weights['out'])
    return out_layer


def run_binary_xgemm_mlp(learning_rate: float,
                         num_steps: int,
                         batch_size: int,
                         display_step: int,
                         hidden_size: int):
    # get MNIST dataset
    mnist = get_mnist()

    # region <<<Network build>>>

    # construct stateless graph
    features = tf.placeholder("float", [None, MNIST_FEATURES])
    targets = tf.placeholder("float", [None, MNIST_CLASSES])

    # store layers weights
    weights = {
        'h1': tf.Variable(tf.random_normal([MNIST_FEATURES, hidden_size])),
        'h2': tf.Variable(tf.random_normal([hidden_size, hidden_size])),
        'h3': tf.Variable(tf.random_normal([hidden_size, hidden_size])),
        'out': tf.Variable(tf.random_normal([hidden_size, MNIST_CLASSES]))
    }

    # construct model
    logits = binary_xgemm_net(
        features=features,
        weights=weights
    )

    prediction = tf.nn.softmax(logits)

    # define loss and optimizer
    loss_op = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    )

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(targets, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialize the variables
    init = tf.global_variables_initializer()
    # endregion

    # region <<<Training>>>
    with tf.Session() as sess:

        # run initializer
        sess.run(init)

        start_time = time.time()
        # repeat required number of steps
        for step in range(1, num_steps + 1):

            # get mini-batch
            batch_features, batch_targets = mnist.train.next_batch(batch_size)

            # feed batch into network and optimize
            sess.run(
                train_op,
                feed_dict={
                    features: batch_features,
                    targets: batch_targets
                }
            )

            # logging
            if step % display_step == 0 or step == 1:
                # calculate loss and accuracy for the batch
                loss, acc = sess.run(
                    [loss_op, accuracy],
                    feed_dict={
                        features: batch_features,
                        targets: batch_targets
                    }
                )

                # print the result
                print('[{:5d}] loss: {:10.2f}\taccuracy: {:2.2f}%'.format(
                    step, loss, acc * 100.0
                ))
        # endregion

        print("elapsed time: {:.4f} seconds".format(time.time() - start_time))

        # region <<<Testing>>>

        total_correct = 0
        for i in range(0, 10000, 1000):
            batch_features, batch_targets = mnist.test.next_batch(1000)
            total_correct += sum(sess.run(correct_pred,
                                          feed_dict={
                                              features: batch_features,
                                              targets: batch_targets
                                          }))
        print("Testing Accuracy: {:.4f}%".format(
            (total_correct / 100.0)
        ))
    # endregion
