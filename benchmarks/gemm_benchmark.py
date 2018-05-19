import os
import time

import numpy as np
import tensorflow as tf

from op_interface import xgemm


def gemm_benchmark(run_xgemm: bool = True,
                   run_tf: bool = True,
                   min: int = 100,
                   max: int = 4101,
                   step: int = 100,
                   batch_size: int = 32,
                   iterations: int = 41,
                   out_path: str = ''):
    """
    Compare xgemm kernel to tf.matmul and produce 3 output logs
    :param run_xgemm: run xgemm operation
    :param run_tf: run tf.matmul operation
    :param min: starting matrix dimension
    :param max: maximal matrix dimension
    :param step: increase of matrix dimension for each step
    :param batch_size: first matrix dimension
    :param iterations: number of iterations
    :param out_path: path where logs will be outputed
    """
    assert min < max

    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    def _run(name: str, operator) -> dict:
        elapsed_start = time.time()

        results = dict()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            # first run only xgemm
            for n in range(min, max, step):

                # create placeholders for inputs
                A = tf.placeholder(tf.float32, [batch_size, n])
                B = tf.placeholder(tf.float32, [n, batch_size])

                # run on GPU
                with tf.device('/gpu:0'):
                    if name == 'xgemm':
                        matmul = operator(A, B, otype=tf.float32)
                    else:
                        matmul = operator(A, B)

                timings = np.zeros(iterations)

                # Re-use a for benchmarking on GPU w/only 4GB memory
                a_T = tf.sign(tf.sign(tf.cast(tf.random_normal(shape=[batch_size, n], seed=1), dtype=tf.float32)) - 0.5)
                b_T = tf.sign(tf.sign(tf.cast(tf.random_normal(shape=[n, batch_size], seed=2), dtype=tf.float32)) - 0.5)

                a = sess.run(a_T)
                b = sess.run(b_T)

                for i in range(iterations):
                    start_time = time.time()
                    _ = sess.run(matmul, feed_dict={A: a, B: b})
                    timings[i] = time.time() - start_time

                results[n] = dict(
                    ite=iterations - 1,
                    avg=timings[1:].mean(),
                    std=timings[1:].std(),
                    med=np.median(timings[1:])
                )

            print("Finished BM for `{}` batch size: {} in {:.2f}s".format(name, batch_size, time.time() - elapsed_start))
            return results

    xgemm_result = base_result = dict()
    if run_xgemm:
        xgemm_result = _run('xgemm', xgemm)

    if run_tf:
        base_result = _run('tf.matmul', tf.matmul)

    # log out xgemm results
    speedup_log = os.path.join(out_path, '{}{}').format('su', '_{}.log'.format(batch_size))
    speedup_file = open(speedup_log, 'w')

    print("===================")
    for size in sorted(list(set(xgemm_result.keys()) & set(base_result.keys()))):
        speedup_file.write('{}\t{}\n'.format(size, base_result[size]['avg'] / xgemm_result[size]['avg']))

    speedup_file.close()
