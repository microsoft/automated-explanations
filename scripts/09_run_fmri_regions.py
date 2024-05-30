from imodelsx import submit_utils
from os.path import dirname, join
import os.path

repo_dir = dirname(dirname(os.path.abspath(__file__)))

# python /home/chansingh/automated-explanations/experiments/01_explain.py --save_dir /home/chansingh/mntv1/mprompt/aug1_llama --checkpoint_module decapoda-research/llama-30b-hf --use_cache 1 --subsample_frac 1 --module_num 499 --module_name fmri --subject UTS03 --num_top_ngrams_to_use 30 --num_top_ngrams_to_consider 50 --num_summaries 5 --num_synthetic_strs 10 --seed 1

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    "save_dir": [join("/home/chansingh/automated-explanations/results/regions_may29")],
    'checkpoint_module': ["facebook/opt-30b"],
    # 'checkpoint_module': ["decapoda-research/llama-30b-hf"],
    "use_cache": [
        1
    ],  # pass binary values with 0/1 instead of the ambiguous strings True/False
    "subsample_frac": [1],
    "module_name": ["fmri"],
    "subject": ["UTS02"],  # , 'UTS03', 'UTS02',
    # "module_num": list(range(500)),
    "module_num": ["all"],  # list(range(500)),
    # "num_top_ngrams_to_use": [30],
    # "num_top_ngrams_to_consider": [50],
    # "num_summaries": [5],
    # "num_synthetic_strs": [10],
    "seed": [1],
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
    script_name=join(repo_dir, "experiments", "03_explain_regions.py"),
    repeat_failed_jobs=True,
    # reverse=True,
    # actually_run=False,
    # shuffle=False,
    n_cpus=1,
)
