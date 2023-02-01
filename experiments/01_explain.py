import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle as pkl
import imodelsx
import torch
import mprompt.modules.fmri
import mprompt.methods.ngrams
import cache_save_utils

# initialize args
def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument('--subsample_frac', type=float,
                        default=1, help='fraction of samples to use')

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default='results/tmp',
                        help='directory for saving')

    # module args
    parser.add_argument('--module_name', type=str,
                        default='fmri', help='name of module', choices=['fmri'])
    parser.add_argument('--module_num', type=int,
                        default=0, help='number of module to select')


    # algo args
    parser.add_argument('--method_name', type=str, choices=['ngrams'],
                        default='ngrams', help='name of algo for explanation')
    return parser

def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldn't use when checking cache)
    """
    parser.add_argument('--use_cache', type=int, default=1, choices=[0, 1],
                        help='whether to check for cache')
    return parser

if __name__ == '__main__':
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir)
    
    if args.use_cache and already_cached:
        logging.info(
            f'cached version exists! Successfully skipping :)\n\n\n')
        exit(0)
    for k in sorted(vars(args)):
        logger.info('\t' + k + ' ' + str(vars(args)[k]))
    logging.info(f'\n\n\tsaving to ' + save_dir_unique + '\n')

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load module to interpret
    if args.module_name == 'fmri':
        mod = mprompt.modules.fmri.fMRIModule(voxel_num_best=args.module_num)    

    # load text data
    text_str_list = mod.get_relevant_data()
    text_str_list = text_str_list[:int(len(text_str_list) * args.subsample_frac)] # note: this isn't shuffling!

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r['save_dir_unique'] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname='params.json', r=r)

    # explain with method
    explanation = mprompt.methods.ngrams.explain_ngrams(text_str_list, mod)
    r['explanation_init'] = explanation

    # r, model = fit_model(model, X_train, y_train, feature_names, r)
    
    # evaluate
    # r = evaluate_model(model, X_train, X_cv, X_test, y_train, y_cv, y_test, r)

    # save results
    pkl.dump(r, open(join(save_dir_unique, 'results.pkl'), 'wb'))
    # pkl.dump(model, open(join(save_dir_unique, 'model.pkl'), 'wb'))
    logging.info('Succesfully completed :)\n\n')
