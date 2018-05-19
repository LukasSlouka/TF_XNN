import argparse
import sys

import tensorflow as tf


def parse_arguments():
    """
    Parse command line arguments and return them
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', help='test all operators', action="store_true")
    parser.add_argument('--binarize_cols', help='test binarize cols operator', action="store_true")
    parser.add_argument('--binarize_rows', help='test binarize rows operator', action="store_true")
    parser.add_argument('--xgemm', help='test xgemm operator', action="store_true")
    ns, args = parser.parse_known_args()
    return ns, sys.argv[:1] + args


if __name__ == '__main__':

    args, argv = parse_arguments()
    sys.argv[:] = argv

    # import requested test cases
    if args.all or args.binarize_cols:
        # noinspection PyUnresolvedReferences
        from tests import TestBinarizeCols
    if args.all or args.binarize_rows:
        # noinspection PyUnresolvedReferences
        from tests import TestBinarizeRows
    if args.all or args.xgemm:
        # noinspection PyUnresolvedReferences
        from tests import TestXGEMM

    # run tests
    tf.test.main()
