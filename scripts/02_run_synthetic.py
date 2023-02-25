from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))
from imodelsx import submit_utils

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    'save_dir': [join('/home/chansingh/mntv1/mprompt', 'feb18_synthetic_sweep')],
    'use_cache': [1],
    'subsample_frac': [1],
    'module_num': list(range(54)),
    # 'module_num': [3],
    # [3, 7, 9, 12, 13, 16, 23, 32, 50, 53],
    'module_name': ['emb_diff_d3'],
    'num_top_ngrams_to_use': [30],
    'num_top_ngrams_to_consider': [50],
    'generate_template_num': [1],
    'num_summaries': [2],
    'num_synthetic_strs': [10],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
params_coupled_dict = {

    # default params
    ('noise_ngram_scores', 'module_num_restrict'): [
        (0, -1),
    ],

    # ablations
    # ('noise_ngram_scores',): [
        # (i,) for i in [3]
    # ],
    ('module_num', 'module_num_restrict',): [
        (i, (i + 1) % 54) for i in range(54)
    ],
}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
print('args_list', args_list)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '01_explain.py'),
    actually_run=True,
    # gpu_ids=[0],
    n_cpus=1, # 20
    repeat_failed_jobs=True,
)
