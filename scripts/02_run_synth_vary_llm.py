from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))
from imodelsx import submit_utils

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'checkpoint': ['text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-003'],
    'seed': [1],
    'save_dir': [join('/home/chansingh/sasc/results/', 'may10')],
    'use_cache': [1],
    'subsample_frac': [1],
    'module_name': ['emb_diff_d3'],
    'num_top_ngrams_to_use': [30],
    'num_top_ngrams_to_consider': [50],
    'generate_template_num': [1],
    'num_summaries': [5],
    'num_synthetic_strs': [10],
    'module_num': list(range(54)), # should go all the way to 54
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
    n_cpus=5, # 20
    repeat_failed_jobs=True,
    shuffle=True,
)
