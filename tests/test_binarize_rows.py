import tensorflow as tf

from op_interface import binarize_rows
from .test_fixture import TestFixture, generate_matrix, python_binarize_rows


class TestBinarizeRows(TestFixture):

    # region <<<GPU Binary Squeeze Tests>>
    def testBinsqueezeA_GPU_32bit(self):
        with self.test_session():
            with tf.device('/gpu:0'):
                a_squeeze = binarize_rows(self.A, qtype=tf.int32).eval()
            self.assertAllEqual(a_squeeze, self.AC.eval())

    def testBinsqueezeA_GPU_16bit(self):
        with self.test_session():
            with tf.device('/gpu:0'):
                a_squeeze = binarize_rows(self.A, qtype=tf.int16).eval()
            self.assertAllEqual(a_squeeze, self.AC16.eval())

    def testBinsqueezeA_GPU_8bit(self):
        with self.test_session():
            with tf.device('/gpu:0'):
                a_squeeze = binarize_rows(self.A, qtype=tf.int8).eval()
            self.assertAllEqual(a_squeeze, self.AC8.eval())

    def testBinsqueezeB_GPU(self):
        with self.test_session():
            with tf.device('/gpu:0'):
                b_squeeze = tf.transpose(binarize_rows(tf.transpose(self.B), qtype=tf.int32))
            self.assertAllEqual(b_squeeze.eval(), self.BC.eval())

    # endregion

    # region <<<CPU Binary Squeeze Tests>>
    def testBinsqueezeA_CPU_32bit(self):
        with self.test_session():
            with tf.device('/cpu:0'):
                a_squeeze = binarize_rows(self.A, qtype=tf.int32).eval()
            self.assertAllEqual(a_squeeze, self.AC.eval())

    def testBinsqueezeA_CPU_16bit(self):
        with self.test_session():
            with tf.device('/cpu:0'):
                a_squeeze = binarize_rows(self.A, qtype=tf.int16).eval()
            self.assertAllEqual(a_squeeze, self.AC16.eval())

    def testBinsqueezeA_CPU_8bit(self):
        with self.test_session():
            with tf.device('/cpu:0'):
                a_squeeze = binarize_rows(self.A, qtype=tf.int8).eval()
            self.assertAllEqual(a_squeeze, self.AC8.eval())

    def testBinsqueezeB_CPU(self):
        with self.test_session():
            with tf.device('/cpu:0'):
                b_squeeze = tf.transpose(binarize_rows(tf.transpose(self.B), qtype=tf.int32))
            self.assertAllEqual(b_squeeze.eval(), self.BC.eval())

    # endregion

    def testPythonSqueeze(self):
        with self.test_session():
            m = generate_matrix(200)
            mm = tf.constant(m, dtype=tf.float32)
            pp = tf.constant(python_binarize_rows(m)).eval()

            with tf.device('gpu:0'):
                mm_squeeze = binarize_rows(mm, qtype=tf.int32).eval()

            self.assertAllEqual(mm_squeeze, pp)

    # def testBinsqueezeLarge(self):
    #     with self.test_session():
    #         m = generate_matrix(3333)
    #         mm = tf.constant(m, dtype=tf.float32)
    #         pp = tf.constant(python_binsqueeze(m)).eval()
    #
    #         with tf.device('gpu:0'):
    #             mm_squeeze = binarize_rows(mm, qtype=tf.int32).eval()
    #
    #         self.assertAllEqual(mm_squeeze, pp)

    # def testBinsqueezeLargeCPU(self):
    #     with self.test_session():
    #         m = generate_matrix(1200)
    #         mm = tf.constant(m, dtype=tf.float32)
    #         pp = tf.constant(python_binsqueeze(m)).eval()
    #
    #         with tf.device('cpu:0'):
    #             mm_squeeze = binarize_rows(mm, qtype=tf.int32).eval()
    #
    #         self.assertAllEqual(mm_squeeze, pp)
