import tensorflow as tf

from op_interface import binarize_rows, binarize_cols
from .test_fixture import TestFixture


class TestBinarizeCols(TestFixture):

    def testBinarizeColsTranposeB(self):
        with self.test_session():
            b_squeeze = binarize_cols(self.B)
            self.assertAllEqual(b_squeeze.eval(), tf.transpose(self.BC).eval())

    def testBinarizeColsTranposeUnaligned(self):
        with self.test_session():
            M = tf.constant(
                [[-1, -1, -1],
                 [-1, -1, 1],
                 [-1, 1, -1],
                 [-1, 1, 1],
                 [-1, -1, -1],
                 [-1, -1, 1]], dtype=tf.int32
            )

            bst = binarize_rows(tf.transpose(M), qtype=tf.int32)
            b_squeeze = binarize_cols(M)
            self.assertAllEqual(bst.eval(), b_squeeze.eval())

    def testBSTvsBS(self):
        with self.test_session():
            X = tf.random_uniform(minval=-10, maxval=10, shape=[512, 440], dtype=tf.float32).eval()
            X = tf.sign(tf.sign(X) - 0.5)

            bst = binarize_rows(X, qtype=tf.int32)
            Xt = tf.transpose(X)
            bst2 = binarize_cols(Xt)
            self.assertAllEqual(bst.eval(), bst2.eval())

    def testBSTvsBS2(self):
        with self.test_session():
            X = tf.random_uniform(minval=-10, maxval=10, shape=[784, 1000], dtype=tf.float32).eval()
            X = tf.sign(tf.sign(X) - 0.5)

            bst = binarize_rows(X, qtype=tf.int32)
            Xt = tf.transpose(X)
            bst2 = binarize_cols(Xt)
            self.assertAllEqual(bst.eval(), bst2.eval())

    def testBSTvsBS3(self):
        with self.test_session():
            X = tf.random_uniform(minval=-10, maxval=10, shape=[2000, 2000], dtype=tf.float32)
            X = tf.sign(tf.sign(X) - 0.5).eval()

            bst = binarize_rows(X, qtype=tf.int32)
            Xt = tf.transpose(X)
            bst2 = binarize_cols(Xt)
            self.assertAllEqual(bst.eval(), bst2.eval())

    def testBinarizeColsA_GPU_32bit(self):
        with self.test_session():
            with tf.device('/gpu:0'):
                a_squeeze = binarize_cols(tf.transpose(self.A)).eval()
            self.assertAllEqual(a_squeeze, self.AC.eval())

    def testBinarizeColsB_GPU(self):
        with self.test_session():
            with tf.device('/gpu:0'):
                b_squeeze = tf.transpose(binarize_cols(self.B))
            self.assertAllEqual(b_squeeze.eval(), self.BC.eval())

    # def testBinarizeColsTransposeLarge(self):
    #     with self.test_session():
    #         m = generate_matrix(3333)
    #         mc = tf.constant(m, dtype=tf.float32).eval()
    #
    #         mm = tf.transpose(mc).eval()
    #         pm = tf.constant(m.transpose(), dtype=tf.float32).eval()
    #
    #         self.assertAllEqual(mm, pm)
    #
    #         pp = tf.constant(python_binsqueeze(m.transpose()), dtype=tf.int32)
    #         with tf.device('gpu:0'):
    #             mc_squeeze = binarize_cols(mc).eval()
    #             mm_squeeze = binarize_rows(mm, qtype=tf.int32).eval()
    #
    #         self.assertAllEqual(mm_squeeze, mc_squeeze)
    #         self.assertAllEqual(pp.eval(), mc_squeeze)
    #         self.assertAllEqual(pp.eval(), mm_squeeze)
