import argparse



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    

def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='Config')

    parser.add_argument('--cfg',
                        dest='cfg_file',
                        type=str,
                        required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--train',
                        type=str2bool,
                        default=True,
                        help='Whether to train or evaluate.')
    parser.add_argument('--parallelseed',
                        type=str2bool,
                        default=False,
                        help='Whether run random seeds in parallel.')
    parser.add_argument('--parallel',
                        type=str2bool,
                        default=True,
                        help='Whether run different configurations in parallel.')
    parser.add_argument('--mark_done',
                        action='store_true',
                        help='Mark yaml as done after a job has finished.')

    return parser.parse_args()