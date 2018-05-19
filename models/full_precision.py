import time
from typing import Dict

import tensorflow as tf

from .helpers import get_mnist, MNIST_CLASSES, MNIST_FEATURES


def full_precision_net(features: tf.placeholder, weights: Dict[str, tf.Variable]):
    """
    Constructs full precision model
    :param features: model input in form of placeholder
    :param weights: dictionary of weights
    """
    # First hidden layer
    z_1 = tf.matmul(features, weights['h1'])
    a_1 = tf.nn.relu(z_1)

    # Second hidden layer
    z_2 = tf.matmul(a_1, weights['h2'])
    a_2 = tf.nn.relu(z_2)

    # Third hidden layer
    z_3 = tf.matmul(a_2, weights['h3'])
    a_3 = tf.nn.relu(z_3)

    # Network output
    out_layer = tf.matmul(a_3, weights['out'])
    return out_layer


def run_full_precision_mlp(learning_rate: float,
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
    logits = full_precision_net(
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

        elapsed_time = time.time() - start_time
        print("elapsed time: {:.4f} seconds".format(elapsed_time))

        # region <<<Testing>>>
        test_accuracy = sess.run(
            accuracy,
            feed_dict={
                features: mnist.test.images,
                targets: mnist.test.labels
            }
        ) * 100.0
        print("Testing Accuracy: {:.4f}%".format(test_accuracy))
    # endregion
