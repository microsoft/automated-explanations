from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))
from imodelsx import submit_utils

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'seed': [1],
    # 'save_dir': [join('/home/chansingh/mntv1', 'mprompt', 'jan30')],
    'save_dir': [join(repo_dir, 'results', 'feb12_fmri_sweep')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
    'subsample_frac': [1],
    'module_num': list(range(200)),
    'module_name': ['fmri'],
    'num_top_ngrams': [30],
    'num_summaries': [5],
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
print('args_list', args_list)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '01_explain.py'),
    actually_run=True,
    # gpu_ids=[0],
    n_cpus=4,
)
