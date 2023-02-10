import numpy as np
from mprompt.llm import llm_hf
from mprompt.modules.prompted_module import PromptedModule, TASKS
import torch
import random

def test_synthetic(
    checkpoint='gpt2-xl', # 1.5B
    # checkpoint='EleutherAI/gpt-j-6B',
    # checkpoint='facebook/opt-6.7b', # fails with opt models
    # checkpoint='facebook/opt-13b',
):
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    mod = PromptedModule(
        checkpoint=checkpoint
    )
    for task_str in TASKS.keys():
    # for task_str in ['animal']:
    # for task_str in ['numbers']:
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
        for task_str_baseline in TASKS.keys():
            if task_str_baseline == task_str:
                continue
            X = TASKS[task_str_baseline]['examples']
            pred = mod(X)
            probs_baseline |= {
                x: p for x, p in zip(X, pred)
            }

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


if __name__ == '__main__':
    test_synthetic()
