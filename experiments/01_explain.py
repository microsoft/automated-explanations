import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join, dirname
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle as pkl
import imodelsx
import torch
import mprompt.llm
import mprompt.modules.fmri_module
import mprompt.modules.prompted_module
import mprompt.methods.m1_ngrams
import mprompt.methods.m2_summarize
import mprompt.methods.m3_generate
from mprompt.data.data import TASKS_D3, TASKS_TOY
import cache_save_utils


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument('--subsample_frac', type=float,
                        default=1, help='fraction of samples to use')
    parser.add_argument('--checkpoint', type=str,
                        # default='google/flan-t5-xxl',
                        default='text-davinci-003',
                        help='which llm to use for each step')
    parser.add_argument('--checkpoint_module', type=str,
                        default='gpt2-xl', help='which llm to use for the module (if synthetic)')

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default='results/tmp',
                        help='directory for saving')

    # module args
    parser.add_argument('--module_name', type=str,
                        default='prompted_toy', help='name of module', choices=['fmri', 'prompted_d3', 'prompted_toy'])
    parser.add_argument('--module_num', type=int, # good task is d3_13_water or d3_16_hillary
                        default=0, help='number of module to select')

    # algo args
    parser.add_argument('--method_name', type=str, choices=['ngrams'],
                        default='ngrams', help='name of algo for explanation')
    parser.add_argument('--num_top_ngrams', type=int,
                        default=10, help='number of ngrams to use to start the explanation')
    parser.add_argument('--num_summaries', type=int,
                        default=2, help='number of summaries to start with')
    parser.add_argument('--num_synthetic_strs', type=int,
                        default=2, help='number of synthetic strings to generate')
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldn't use when checking cache)
    """
    parser.add_argument('--use_cache', type=int, default=0, choices=[0, 1],
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
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir)

    if args.use_cache and already_cached:
        logging.info(
            f'cached version exists! Successfully skipping :)\n\n\n')
        exit(0)
    for k in sorted(vars(args)):
        logging.info('\t' + k + ' ' + str(vars(args)[k]))
    logging.info(f'\n\n\tsaving to ' + save_dir_unique + '\n')

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r['save_dir_unique'] = save_dir_unique
    cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname='params.json', r=r)

    # load module to interpret
    if args.module_name == 'fmri':
        mod = mprompt.modules.fmri_module.fMRIModule(
            voxel_num_best=args.module_num)
    elif args.module_name.startswith('prompted'):
        if args.module_name == 'prompted_d3':
            T = TASKS_D3
        else:
            T = TASKS_TOY
        task_str = list(T.keys())[args.module_num]
        print('running', task_str)
        mod = mprompt.modules.prompted_module.PromptedModule(
            task_str=task_str,
            checkpoint=args.checkpoint_module,
        )

    # load text data
    text_str_list = mod.get_relevant_data()

    # subsample data
    if args.subsample_frac < 1 and not args.module_name == 'fmri':
        assert False, 'dont subsample data right now, since explain_ngrams is using caching'
        # n_subsample = int(len(text_str_list) * args.subsample_frac)

        # randomly subsample list
        # text_str_list, size=n_subsample, replace=False).tolist()
            # text_str_list, size=n_subsample, replace=False).tolist()

    # explain with method
    explanation_init_ngrams = mprompt.methods.m1_ngrams.explain_ngrams(
        args,
        text_str_list,
        mod,
    )
    r['explanation_init_ngrams'] = explanation_init_ngrams
    logging.info(
        f'{explanation_init_ngrams[:3]=} {len(explanation_init_ngrams)}')

    # summarize the ngrams into some candidate strings
    llm = mprompt.llm.get_llm(args.checkpoint)
    explanation_strs = mprompt.methods.m2_summarize.summarize_ngrams(
        llm, explanation_init_ngrams,
        num_summaries=args.num_summaries, num_top_ngrams=args.num_top_ngrams)
    r['explanation_init_strs'] = explanation_strs
    logging.info('explanation_init_strs\n\t' + '\n\t'.join(explanation_strs))

    # generate synthetic data
    logging.info('\n\nGenerating synthetic data....')
    for explanation_str in explanation_strs:
        strs_added, strs_removed = mprompt.methods.m3_generate.generate_synthetic_strs(
            llm, explanation_str=explanation_str,
            num_synthetic_strs=args.num_synthetic_strs)
        r['strs_added'].append(strs_added)
        r['strs_removed'].append(strs_removed)

        # evaluate synthetic data (higher score is better)
        r['score_synthetic'].append(
            np.mean(mod(strs_added) - mod(strs_removed)))
    logging.info(f'{explanation_strs[0]}\n+++++++++\n\t' + '\n\t'.join(r['strs_added'][0][:3]) +
                 '\n--------\n\t' + '\n\t'.join(r['strs_removed'][0][:3]))

    # sort everything by score
    sort_inds = np.argsort(r['score_synthetic'])[::-1]
    for k in ['explanation_init_strs', 'strs_added', 'strs_removed', 'score_synthetic']:
        r[k] = [r[k][i] for i in sort_inds]
        r['top_' + k] = r[k][0]

    # evaluate how well explanation matches a "groundtruth"
    if getattr(mod, "get_groundtruth_explanation", None):
        logging.info('\n\Scoring explanation....')

        # get groundtruth explanation
        explanation_groundtruth = mod.get_groundtruth_explanation()
        check_func = mod.get_groundtruth_keywords_check_func()

        for explanation_str in explanation_strs:
            # compute bleu score with groundtruth explanation
            # r['score_bleu'].append(
            # calc_bleu_score(explanation_groundtruth, explanation_str))

            # compute whether explanation contains any of the synthetic keywords
            r['score_contains_keywords'].append(check_func(explanation_str))

    # save results
    pkl.dump(r, open(join(save_dir_unique, 'results.pkl'), 'wb'))
    logging.info('Succesfully completed :)\n\n')
