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
import sasc.methods.llm
import sasc.modules.old_fmri_module
import sasc.modules.fmri_module
import sasc.modules.prompted_module
import sasc.modules.emb_diff_module
import sasc.modules.dictionary_module
import sasc.methods.m1_ngrams
import sasc.methods.m2_summarize
import sasc.methods.m3_generate
import sasc.data.data
from sasc.data.data import TASKS_D3, TASKS_TOY
from sasc.modules.dictionary_learning.norm_std import get_std
from imodelsx import cache_save_utils


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

    # data ablations
    parser.add_argument('--noise_ngram_scores', type=float, default=0,
                        help='''ablation: how much noise to add to ngram scores
                        (noise stddev = noise_ngram_scores * stddev(top-100 ngram responses)''')
    parser.add_argument('--module_num_restrict', type=int, default=-1,
                        help='''ablation: alternative module_num to specify a corpus to restrict the ngram responses''')
    

    # training misc args
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--save_dir', type=str, default='results/tmp',
                        help='directory for saving')

    # module args
    parser.add_argument('--module_name', type=str,
                        default='emb_diff_toy', help='name of module',
                        choices=['fmri', 'old_fmri',
                                 'emb_diff_d3', 'emb_diff_toy',
                                 'prompted_d3', 'prompted_toy',
                                 'dict_learn_factor'])
    parser.add_argument('--module_num', type=int, # good task is d3_13_water or d3_16_hillary
                        default=0, help='number of module to select')
    parser.add_argument('--subject', type=str,
                        default='UTS03', help='for fMRI, which subject to use')
    parser.add_argument('--factor_layer', type=int,
                        default=4, help='for dictionary learning module, which layer of factor to use')
    parser.add_argument('--factor_idx', type=int,
                        default=2, help='for dictionary learning module, which index of factor in the layer to use')
    parser.add_argument('--get_baseline_exp', type=int,
                        default=0, help='whether to get full baseline exp results')

    # algo args
    parser.add_argument('--method_name', type=str, choices=['ngrams'],
                        default='ngrams', help='name of algo for explanation')
    parser.add_argument('--num_top_ngrams_to_use', type=int,
                        default=3, help='number of ngrams to use to start the explanation')
    parser.add_argument('--num_top_ngrams_to_consider', type=int,
                        default=5, help='select top ngrams from this many')
    parser.add_argument('--num_summaries', type=int,
                        default=2, help='number of summaries to start with')
    parser.add_argument('--num_synthetic_strs', type=int,
                        default=3, help='number of synthetic strings to generate')
    parser.add_argument('--generate_template_num', type=int,
                        default=1, help='which template to use for generation')
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
    assert not args.module_num == args.module_num_restrict, 'module_num and module_num_restrict should be different'

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
        mod = sasc.modules.fmri_module.fMRIModule(
            voxel_num_best=args.module_num, subject=args.subject)
        r['fmri_test_corr'] = mod.corr
    elif args.module_name == 'old_fmri':
        mod = sasc.modules.old_fmri_module.OldFMRIModule(
            voxel_num_best=args.module_num)
        r['fmri_test_corr'] = mod.corr
    elif args.module_name == 'dict_learn_factor':
        mod = sasc.modules.dictionary_module.DictionaryModule(
            layer_idx=args.factor_layer, factor_idx=args.factor_idx)
    else:
        task_str = sasc.data.data.get_task_str(args.module_name, args.module_num)
        logging.info('running ' + task_str)
        if args.module_name.startswith('prompted'):
            mod = sasc.modules.prompted_module.PromptedModule(
                task_str=task_str,
                checkpoint=args.checkpoint_module,
            )
        elif args.module_name.startswith('emb_diff'):
            mod = sasc.modules.emb_diff_module.EmbDiffModule(
                task_str=task_str,
                # checkpoint=args.checkpoint_module,
            )
    
    llm = sasc.methods.llm.get_llm(args.checkpoint)

    # load explanation result
    explanation_strs, control_data = sasc.data.data.get_eval_data(
        factor_layer = args.factor_layer, factor_idx = args.factor_idx, get_baseline = args.get_baseline_exp)
    r['explanation_init_strs'] = explanation_strs

    std_lst = np.array(get_std(''))
    std = std_lst[args.factor_layer][args.factor_idx]

    # generate synthetic data
    logging.info('\n\nGenerating synthetic data....')
    for explanation_str in explanation_strs:
        strs_added, strs_removed = sasc.methods.m3_generate.generate_synthetic_strs(
            llm,
            explanation_str=explanation_str,
            num_synthetic_strs=args.num_synthetic_strs,
            template_num=args.generate_template_num,
        )
        r['strs_added'].append(strs_added)
        r['strs_removed'].append(strs_removed)

        # evaluate synthetic data (higher score is better)
        mod_responses = mod(strs_added + strs_removed)
        r['score_synthetic'].append(
            np.mean(mod_responses[:len(strs_added)]) -
            np.mean(mod_responses[len(strs_added):])
        )
        
        r['score_synthetic_std'].append(
            r['score_synthetic'] / std
        )

    logging.info(f'{explanation_strs[0]}\n+++++++++\n\t' + '\n\t'.join(r['strs_added'][0][:3]) +
                 '\n--------\n\t' + '\n\t'.join(r['strs_removed'][0][:3]))
    
    if not args.get_baseline_exp:
        # evaluate control data (higher score is better)
        control_strs_added = control_data['strs_added']
        control_strs_removed = control_data['strs_removed']
        r['control_strs_added'] = control_strs_added
        r['control_strs_removed'] = control_strs_removed
        c_mod_responses = mod(control_strs_added + control_strs_removed)
        r['control_score_synthetic'] = np.mean(c_mod_responses[:len(control_strs_added)]) - np.mean(c_mod_responses[len(control_strs_added):])

    # save results
    pkl.dump(r, open(join(save_dir_unique, f'eval.pkl'), 'wb'))
    logging.info('Succesfully completed :)\n\n')
