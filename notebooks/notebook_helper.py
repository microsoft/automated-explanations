import argparse
import sys
import os.path
from os.path import dirname, join
repo_dir = dirname(dirname(os.path.abspath(__file__)))


def get_main_args_list(fname='01_train_model.py'):
    """Returns main arguments from the argparser used by an experiments script
    """
    if fname.endswith('.py'):
        fname = fname[:-3]
    sys.path.append(join(repo_dir, 'experiments'))
    train_script = __import__(fname)
    args = train_script.add_main_args(argparse.ArgumentParser()).parse_args([])
    return list(vars(args).keys())
