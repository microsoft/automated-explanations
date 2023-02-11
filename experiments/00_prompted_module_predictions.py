import pickle as pkl
from mprompt.modules.prompted_module import PromptedModule
from mprompt.llm import llm_hf
import random
import torch
import numpy as np
from mprompt.data.data import TASKS, TASKS_TOY, TASKS_D3
from os.path import join, dirname
from tqdm import tqdm
import os
path_to_repo = dirname(dirname(os.path.abspath(__file__)))


def calculate_mean_preds_matrix_over_tasks(mod, task_names, assert_checks=False):
    """Calculate mean predictions using prompts from different modules.
    Matrix is (n_tasks, n_tasks) where each entry is the mean prediction
    Each row uses a different prompt in the task and each column evaluates on examples from different tasks
    """
    n = len(task_names)
    mean_preds_matrix = np.zeros((n, n))

    for r, task_str in enumerate(tqdm(task_names)):
        # decide which task we are going to predict for
        print('\ntask_str', task_str)
        mod._init_task(task_str)
        X = TASKS[task_str]['examples']
        pred = mod(X)
        probs_pos = {
            x: p for x, p in zip(X, pred)
        }

        # calculate probs for other categories
        probs_baseline = {}
        for c, task_str_baseline in enumerate(task_names):
            if task_str_baseline == task_str:
                mean_preds_matrix[r, c] = np.mean(list(probs_pos.values()))
            else:
                X = TASKS[task_str_baseline]['examples']
                pred = mod(X)
                probs_baseline |= {
                    x: p for x, p in zip(X, pred)
                }
                mean_preds_matrix[r, c] = np.mean(list(probs_baseline.values()))

        if assert_checks:
            for k in sorted(probs_pos, key=probs_pos.get, reverse=True):
                print(f'\t{k} {probs_pos[k]:.2e}')
            print('\t-------------------')
            for k in sorted(probs_baseline, key=probs_baseline.get, reverse=True):
                print(f'\t{k} {probs_baseline[k]:.2e}')

            vals = np.array(list(probs_pos.values()))
            vals_baseline = np.array(list(probs_baseline.values()))
            assert np.mean(vals) > np.mean(vals_baseline), \
                f'mean for inputs in {task_str} should be higher for positive examples'
            assert np.min(vals) > np.max(vals_baseline), \
                f'min for pos inputs should be greater than max for neg inputs in {task_str}'

    return mean_preds_matrix


if __name__ == '__main__':
    checkpoint = 'gpt2-xl'
    task_names = list(TASKS_D3.keys())
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    mod = PromptedModule(
        checkpoint=checkpoint,
    )
    mean_preds_matrix = calculate_mean_preds_matrix_over_tasks(
        mod, task_names, assert_checks=False)
    pkl.dump(mean_preds_matrix,
             open(join(path_to_repo, 'results', 'mean_preds_matrix_d3.pkl'), 'wb'))