
from typing import List
from mprompt.data.d3 import TASKS_D3
from mprompt.data.toy import TASKS_TOY
from mprompt.modules.fmri_module import SAVE_DIR_FMRI
from os.path import join
TASKS = {**TASKS_D3, **TASKS_TOY}

def get_relevant_data(module_name, module_num) -> List[str]:
    if module_name == 'fmri':
        """read in full text of 26 narrative stories (includes train and test)
        """
        with open(join(SAVE_DIR_FMRI, 'narrative_stories.txt'), 'r') as f:
            narrative_stories = f.readlines()
        return narrative_stories
    else:
        task_str = get_task_str(module_name, module_num)
        return TASKS[task_str]['gen_func']()

def get_task_str(module_name, module_num) -> str:
    if module_name.endswith('d3'):
        T = TASKS_D3
    elif module_name.endswith('toy'):
        T = TASKS_TOY
    task_str = list(T.keys())[module_num]
    return task_str

def get_task_keyword(task_name):
    return TASKS[task_name]['target_token'].strip()

if __name__ == '__main__':
    for i, task in enumerate(TASKS_D3):
        task_keyword = get_task_keyword(task)
        print(task, get_task_keyword(task))