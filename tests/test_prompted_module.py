import pickle as pkl
from sasc.modules.prompted_module import PromptedModule
from sasc.modules.emb_diff_module import EmbDiffModule
from imodelsx.sasc.llm import llm_hf
import random
import torch
import numpy as np
from sasc.data.data import TASKS, TASKS_TOY, TASKS_D3
from os.path import join, dirname
from tqdm import tqdm
import sys
import os
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
path_to_repo = dirname(dirname(os.path.abspath(__file__)))
sys.path.append(join(path_to_repo, 'experiments'))
prompted_module_exp = __import__('00_prompted_module_predictions')

def plot_mean_preds_matrix(mean_preds_matrix, task_names, save_name):
    plt.imshow(mean_preds_matrix)
    plt.colorbar()
    plt.ylabel('Task used for prompt')
    plt.xlabel('Examples from this task')
    plt.yticks(np.arange(len(task_names)), labels=task_names, rotation='horizontal', fontsize='small')
    plt.xticks(np.arange(len(task_names)), labels=task_names, rotation='vertical', fontsize='small')
    plt.savefig(save_name, dpi=300)
    plt.close()

def test_mean_preds_matrix(plot=False, assert_checks=False):
    mean_preds_matrix_dict = {}
    checkpoints = ['facebook/opt-iml-30b']
    # checkpoints = ['roberta-large']
    for checkpoint in checkpoints:
        # mod = PromptedModule(checkpoint=checkpoint)
        mod = EmbDiffModule(checkpoint=checkpoint)
        task_names = list(TASKS_TOY.keys())
        mean_preds_matrix = prompted_module_exp.calculate_mean_preds_matrix_over_tasks(
            mod, task_names, assert_checks=assert_checks,
        )
        # mean_preds_matrix = normalize(mean_preds_matrix, axis=1, norm='max')

        if plot:
            save_name = f'mean_preds_matrix_{checkpoint.replace("/", "__")}.png'
            plot_mean_preds_matrix(mean_preds_matrix, task_names, save_name)
        mean_preds_matrix_dict[checkpoint] = mean_preds_matrix
    if plot:
        save_name = 'mean_preds_matrix_toy_diff.png'
        plot_mean_preds_matrix(mean_preds_matrix_dict[checkpoints[1]] - mean_preds_matrix_dict[checkpoints[0]], task_names, save_name)
    

if __name__ == '__main__':
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    test_mean_preds_matrix(plot=True)