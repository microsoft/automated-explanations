import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    # 'seed': [1],
    'save_dir': [join('/home/chansingh/mntv1', 'mprompt', 'jan30')],
    'use_cache': [1], # pass binary values with 0/1 instead of the ambiguous strings True/False
    'subsample_frac': [1],
    # 'module_num': list(range(50)),
}

# List of tuples to sweep over (these values are coupled, and swept over together)
params_coupled_dict = {('module_num', 'seed'): [(i, 1) for i in range(50)]}

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
)
