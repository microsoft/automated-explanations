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
from mprompt.data.data import TASKS
modules_dir = dirname(os.path.abspath(__file__))


class PromptedModule():

    def __init__(self, task_str: str = 'toy_animal', checkpoint='gpt-xl'):
        """
        Params
        ------
        """
        print(f'loading {checkpoint}...')
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
    mod = PromptedModule()
    X = mod.get_relevant_data()
    print(X[0][:50])
    resp = mod(X[:3])
    print(resp)
