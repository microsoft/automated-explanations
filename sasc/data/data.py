from typing import List
import numpy as np
from datasets import load_dataset
from sasc.data.d3 import TASKS_D3
from sasc.data.toy import TASKS_TOY
from sasc.modules.old_fmri_module import SAVE_DIR_FMRI
from sasc.modules.fmri_module import get_train_story_texts
from sasc.modules.dictionary_learning.utils import (
    get_exp_data,
    get_baseline_data,
    SAVE_DIR_DICT,
)
from os.path import join
import re

TASKS = {**TASKS_D3, **TASKS_TOY}


def get_relevant_data(
    module_name: str, module_num: int, subject: str = "UTS03", dl_task: str = "wiki"
) -> List[str]:
    if module_name == "fmri":
        return get_train_story_texts(subject)
    elif module_name == "old_fmri":
        # read in full text of 26 narrative stories (includes train and test)
        with open(join(SAVE_DIR_FMRI, "narrative_stories.txt"), "r") as f:
            narrative_stories = f.readlines()
        return narrative_stories
    elif module_name == "dict_learn_factor":
        num_instances = 30000
        if dl_task == "wiki":
            sentences = np.load(join(SAVE_DIR_DICT, "train_sentences.npy")).tolist()[
                :num_instances
            ]
        else:  # dl_task = 'sst2'
            dataset = load_dataset("glue", "sst2")
            sentences = dataset["train"]["sentence"][:num_instances]
        return sentences
    else:
        task_str = get_task_str(module_name, module_num)
        return TASKS[task_str]["gen_func"]()


def get_eval_data(
    factor_idx: int = 2, factor_layer: int = 4, get_baseline=False
) -> List[str]:
    # this function is now only used by the dict_learn module
    if get_baseline:
        return get_baseline_data(factor_idx, factor_layer)
    else:
        return get_exp_data(factor_idx, factor_layer)


def get_groundtruth_keyword(task_name):
    return TASKS[task_name]["target_token"].strip()


def get_task_str(module_name, module_num) -> str:
    """Return the task string, which is the key in TASKS"""
    if module_name.endswith("d3"):
        T = TASKS_D3
    elif module_name.endswith("toy"):
        T = TASKS_TOY
    task_str = list(T.keys())[module_num]
    return task_str


def get_groundtruth_explanation(task_str):
    """Return the groundtruth explanation"""
    return TASKS[task_str]["groundtruth_explanation"]


def get_groundtruth_keywords_check_func(task_str):
    """Return the groundtruth keywords"""
    task = TASKS[task_str]
    regex = task["check_func"]
    regex_compiled = re.compile(regex, re.IGNORECASE).search

    def check_answer_func(x):
        return bool(regex_compiled(x))

    return check_answer_func


if __name__ == "__main__":
    # task_str = get_task_str('emb_diff_d3', 0)
    # print(task_str)
    # print(get_groundtruth_explanation(task_str))
    # check_func = get_groundtruth_keywords_check_func(task_str)
    # print(check_func('irony'), check_func('apple'))
    get_relevant_data("dict_learn_factor", 2, dl_task="sst2")
