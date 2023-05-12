from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))
from imodelsx import submit_utils
from copy import deepcopy

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

low_level = [(4, 2), (4, 16), (4, 33), (0, 30), (2, 30),
             (4, 30), (4, 47)]
mid_level = [(10, 42), (10, 50), (6, 86), (10, 102), (8, 125),
             (10, 184), (10, 195), (4, 13), (6, 13), (10, 13),
             (10, 24), (10, 25), (10, 51), (10, 86), (10, 99),
             (10, 125), (10, 134), (10, 152), (4, 193), (6, 225)]
high_level = [(10, 297), (10, 322), (10, 386), (10, 179)]
levels = low_level + mid_level + high_level
args_list_list = []
for factor_layer, factor_idx in levels:
    # List of values to sweep over (sweeps over all combinations of these)
    params_shared_dict = {
        'seed': [1],
        'save_dir': [join(repo_dir, 'results', 'bert', 'human_eval', f'b_l{factor_layer}_i{factor_idx}')],
        'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
        'get_baseline_exp': [1],
        'subsample_frac': [1],
        'module_name': ['dict_learn_factor'],
        'factor_idx': [factor_idx],
        'factor_layer': [factor_layer],
        'num_top_ngrams_to_use': [30],
        'num_top_ngrams_to_consider': [50],
        'num_summaries': [2],
        'num_synthetic_strs': [10],
    }

    # List of tuples to sweep over (these values are coupled, and swept over together)
    params_coupled_dict = {}

    # Args list is a list of dictionaries
    # If you want to do something special to remove some of these runs, can remove them before calling run_args_list
    args_list = submit_utils.get_args_list(
        params_shared_dict=params_shared_dict,
        params_coupled_dict=params_coupled_dict,
    )
    args_list_list.append(deepcopy(args_list))
# print('args_list', args_list)
submit_utils.run_args_list(
    sum(args_list_list, []),
    script_name=join(repo_dir, 'experiments', '02_dict_learn_eval.py'),
    actually_run=True,
    # gpu_ids=[0, 1],
    # n_cpus=1,
    shuffle=False,
    # reverse=True,
    repeat_failed_jobs=False,
)
