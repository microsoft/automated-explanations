import logging
from typing import List
import datasets
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import sklearn.preprocessing
from spacy.lang.en import English
import imodelsx
import imodelsx.util
import pickle as pkl
from os.path import dirname, join
import os.path
import re
import mprompt.llm
from langchain import PromptTemplate
modules_dir = dirname(os.path.abspath(__file__))


def generate_synthetic_data() -> List[str]:
    s = []
    for k in TASKS.keys():
        s.append(' '.join(TASKS[k]['examples']))
    return s


TASKS = {
    # Observations: yes/no questions are terrible
    # 'template': 'Does the input contain an animal?\nInput: {input}\nAnswer (yes or no):',
    # 'target_token': ' yes',

    'animal': {
        'check_func': r'animal',
        'groundtruth_explanation': 'Return whether the input is an animal.',
        'template': 'A {input} is a type of',
        'target_token': ' animal',
        'get_relevant_data': generate_synthetic_data,
        'examples': ['cat', 'dog', 'giraffe', 'horse', 'zebra', 'raccoon'],
    },
    'food': {
        'check_func': r'fruit|edible',
        'groundtruth_explanation': 'Return whether the input is a food.',
        'template': '{input} is a type of',
        'target_token': ' food',
        'get_relevant_data': generate_synthetic_data,
        'examples': ['apple', 'orange', 'pear', 'pizza', 'lasagna', 'curry', 'salad', 'chopstick'],
    },
    'numbers': {
        'check_func': r'number',
        'groundtruth_explanation': 'Return whether the input is a number.',
        'template': '{input} is related to the concept of',
        'target_token': ' numbers',
        'get_relevant_data': generate_synthetic_data,
        'examples': ['1', '2', '3', 'four', 'five', 'six', 'plus', 'minus', 'divide'],
    }
}


class SyntheticModule():

    def __init__(self, task_str: str = 'animal', checkpoint='EleutherAI/gpt-j-6B'):
        """
        Params
        ------
        """
        self.llm = mprompt.llm.get_llm(checkpoint)
        self._init_task(task_str)

    def _init_task(self, task_str: str):
        self.task_str = task_str
        self.task = TASKS[task_str]
        self.prompt_template = PromptTemplate(
            input_variables=['input'],
            template=self.task['template'],
        )

    def __call__(self, X: List[str]) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        probs = np.zeros(len(X))
        for i, x in enumerate(X):
            prompt = self.prompt_template.format(input=x)
            probs[i] = self.llm._get_logit_for_target_token(
                prompt, target_token_str=self.task['target_token'])
        return probs

    def get_relevant_data(self) -> List[str]:
        """Return text corpus for this task
        """
        return self.task['get_relevant_data']()

    def get_groundtruth_explanation(self) -> str:
        """Return the groundtruth explanation
        """
        return self.task['groundtruth_explanation']

    def get_groundtruth_keywords_check_func(self) -> str:
        """Return the groundtruth keywords
        """
        regex = self.task['check_func']
        regex_compiled = re.compile(regex, re.IGNORECASE).search

        def check_answer_func(x):
            return bool(regex_compiled(x))
        return check_answer_func


if __name__ == '__main__':
    mod = SyntheticModule()
    X = mod.get_relevant_data()
    print(X[0][:50])
    resp = mod(X[:3])
    print(resp)
