import argparse

from models import run_full_precision_mlp, run_binary_concept_mlp, run_binary_xgemm_mlp


def parse_arguments():
    """
    Parse command line arguments and return them
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--full_precision', help='trains full precision model', action="store_true"
    )
    parser.add_argument(
        '--binary_concept', help='trains binary proof-of-concept model', action="store_true"
    )
    parser.add_argument(
        '--binary_xgemm', help='trains binary with xnor model', action="store_true"
    )
    parser.add_argument(
        '--batch_size', help='number of input per batch (used in training, validating and testing)', type=int, default=256
    )
    parser.add_argument(
        '--learning_rate', help='starting traning learning rate', type=float, default=1e-2
    )
    parser.add_argument(
        '--steps', help='number of steps', type=int, default=1000
    )
    parser.add_argument(
        '--display_step', help='step used for logging', type=int, default=100
    )
    parser.add_argument(
        '--hidden_size', help='number of neurons in hidden layers', type=int, default=2000
    )
    args = parser.parse_args()

    models = [args.full_precision, args.binary_concept, args.binary_xgemm]

    if not any(models):
        raise RuntimeError('Select some model to train')

    if len([x for x in models if x]) > 1:
        raise RuntimeError('Too many models selected, select one')

    return args


def log_args(args):
    print(
        "[LOG] network model: {}\n"
        "[LOG] batch size: {}\n"
        "[LOG] learning rate: {}\n"
        "[LOG] number of neurons in hidden layers: {}\n"
        "[LOG] training steps {} (report every {} steps)".format(
            "FULL PRECISION" if args.full_precision else "BINARY (proof-of-concept)" if args.binary_concept else "BINARY XGEMM",
            args.batch_size, args.learning_rate, args.hidden_size, args.steps, args.display_step
        )
    )


if __name__ == '__main__':

    # parse and log arguments
    args = parse_arguments()
    log_args(args)

    kwargs = dict(
        learning_rate=args.learning_rate,
        num_steps=args.steps,
        batch_size=args.batch_size,
        display_step=args.display_step,
        hidden_size=args.hidden_size
    )

    if args.full_precision:
        run_full_precision_mlp(**kwargs)

    if args.binary_concept:
        run_binary_concept_mlp(**kwargs)

    if args.binary_xgemm:
        run_binary_xgemm_mlp(**kwargs)
