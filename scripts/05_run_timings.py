from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# Showcasing different ways to sweep over arguments
# Can pass any empty dict for any of these to avoid sweeping

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'dummy_arg': list(range(10)),  # not used
}

# List of tuples to sweep over (these values are coupled, and swept over together)
params_coupled_dict = {}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
# print('args_list', args_list)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'notebooks_stories',
                     '1_generate', '03_timings.py'),
    repeat_failed_jobs=True,
    # reverse=True,
    # n_cpus=15,
    # gpu_ids=[0, 1, 2, 3], #, 2, 3],
    gpu_ids=[2, 3],
)
