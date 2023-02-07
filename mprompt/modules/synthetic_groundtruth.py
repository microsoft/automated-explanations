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
modules_dir = dirname(os.path.abspath(__file__))


TASKS = {
    'animal': {
        'check_func': r'animal',
        'template': 'True or False: A {input} is an animal.\nAnswer:',
    },
    'food': {
        'check_func': r'fruit|edible',
    }
}


class SyntheticModule():

    def __init__(self, task_str: str = 'animal', checkpoint='google/flan-t5-xl'):
        """
        Params
        ------
        """
        self.task_str = task_str
        self.llm = mprompt.llm.get_llm(checkpoint)
        self.task = TASKS[task_str]


    def __call__(self, X: List[str]) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        probs = np.zeros(len(X))
        for i, x in enumerate(X):
            probs[i] = self.llm.get_next_token_probs(X, 'yes')
        return probs

    def get_relevant_data(self) -> List[str]:
        """read in full text of 26 narrative stories
        """
        return ['apple orange pear three 4 five cat dog zebra']

    def get_groundtruth_explanation(self) -> str:
        """Return the groundtruth explanation
        """
        return 'animals'

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
