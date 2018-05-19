import tensorflow as tf

from op_interface import xgemm, binarize_cols, binarize_rows
from .test_fixture import TestFixture


class TestXGEMM(TestFixture):

    def testXnorMatmul(self):
        """
        32 x 32 matmul test
        """
        with self.test_session():
            xnor_result = xgemm(self.A, self.B, otype=tf.float32)
            tf_result = tf.matmul(self.A, self.B)
            self.assertAllEqual(xnor_result.eval(), tf_result.eval())

    def testXnorAny(self):
        """
        General matmul test with randomized inputs (random stuff in tests but whatever...)
        Simulate 235 batches of 440-width input for 200 nodes (all irregular)
        """
        with self.test_session():
            BATCH_SIZE = 235
            INPUT_WIDTH = 440
            NEURONS = 200

            features = tf.random_uniform(minval=-10, maxval=10, shape=[BATCH_SIZE, INPUT_WIDTH], dtype=tf.float32).eval()
            weights = tf.random_uniform(minval=-10, maxval=10, shape=[INPUT_WIDTH, NEURONS], dtype=tf.float32).eval()

            # binarize both using sign and cast to integer
            bin_features = tf.sign(tf.sign(features) - 0.5)
            bin_weights = tf.sign(tf.sign(weights) - 0.5)

            # weights squeezed transed
            bqt = binarize_rows(tf.transpose(bin_weights), qtype=tf.int32)
            bqt2 = binarize_cols(bin_weights)

            self.assertAllEqual(bqt.eval(), bqt2.eval())

            xnor_result = xgemm(bin_features, bin_weights, otype=tf.float32)
            tf_result = tf.matmul(bin_features, bin_weights)
            self.assertAllEqual(xnor_result.eval(), tf_result.eval())

    def testGemm(self):
        with self.test_session():
            a = tf.sign(tf.random_normal(shape=[256, 784], seed=1).eval())
            b = tf.sign(tf.random_normal(shape=[784, 1000], seed=2).eval())

            xnor_result = xgemm(a, b, otype=tf.float32)
            tf_result = tf.matmul(a, b)
            self.assertAllEqual(xnor_result.eval(), tf_result.eval())

    def testGemm2(self):
        with self.test_session():
            a = tf.sign(tf.random_normal(shape=[2005, 1024], seed=1).eval())
            b = tf.sign(tf.random_normal(shape=[1024, 2005], seed=2).eval())

            xnor_result = xgemm(a, b, otype=tf.float32)
            tf_result = tf.matmul(a, b)
            self.assertAllEqual(xnor_result.eval(), tf_result.eval())

    def testGemm3(self):
        with self.test_session():
            a = tf.sign(tf.sign(tf.random_normal(shape=[5000, 5000], seed=1)) - 0.5).eval()
            b = tf.sign(tf.sign(tf.random_normal(shape=[5000, 5000], seed=2)) - 0.5).eval()

            xnor_result = xgemm(a, b, otype=tf.float32)
            tf_result = tf.matmul(a, b)
            self.assertAllEqual(xnor_result.eval(), tf_result.eval())
