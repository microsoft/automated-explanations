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
import scipy.spatial.distance
from langchain import PromptTemplate
from mprompt.data.data import TASKS
modules_dir = dirname(os.path.abspath(__file__))


class EmbDiffModule():

    def __init__(
        self,
        task_str: str = 'toy_animal',
        checkpoint='gpt2-xl',
        use_instructor=True,
    ):
        """
        Params
        ------
        """
        if use_instructor:
            print(f'loading hkunlp/instructor-xl...')
            from InstructorEmbedding import INSTRUCTOR
            self.extract_embs = INSTRUCTOR('hkunlp/instructor-xl')
        else:
            print(f'loading {checkpoint}...')
            self.extract_embs = pipeline(
                "feature-extraction",
                model=checkpoint,
                truncation=True,
                device=0
            )
        self.use_instructor = use_instructor
        self._init_task(task_str)

    def _init_task(self, task_str: str):
        self.task_str = task_str
        self.task = TASKS[task_str]
        if 'target_str' in self.task:
            self.target_str = self.task['target_str']
        else:
            self.target_str = self.task['target_token'].strip().split()[0]
        self.emb = self._get_emb(self.target_str)
        # embs = [
        # self._get_emb(x) for x in ['horse', 'dog', 'cat']
        # ]
        # self.emb = np.mean(embs, axis=0)

        # print('ref', self.emb.shape)

    def _get_emb(self, x: str) -> np.ndarray:
        if self.use_instructor:
            instruction = f"Represent the short phrase for clustering: "
            embs = self.extract_embs.encode([[instruction, x]])
            return embs[0]
        else:
            # emb is (batch_size, 1, (seq_len + 2), embedding_dim)
            # embedding_dim = 768 for bert-base-uncased and 1024 for roberta-large
            emb = np.array(self.extract_embs([x]))
            return emb[0, 0].mean(axis=0)  # mean over seq_len
            # return emb[0, 0, 0] # take cls token (first)

    def __call__(self, X: List[str]) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        neg_dists = np.zeros(len(X))
        for i, x in enumerate(tqdm(X)):
            emb = self._get_emb(x)
            # neg_dists[i] = - np.linalg.norm(emb - self.emb, ord=2)
            neg_dists[i] = - scipy.spatial.distance.euclidean(emb, self.emb)
        return neg_dists

    def get_relevant_data(self) -> List[str]:
        """Return text corpus for this task
        """
        return self.task['gen_func']()

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
    mod = EmbDiffModule(
        task_str='toy_animal',
        # checkpoint='bert-base-uncased',
        checkpoint='roberta-large',
    )
    X = mod.get_relevant_data()
    # X = sum([[a for a in x] for x in X], [])
    resps = mod(X[:3])
    for x, resp in zip(X, resps):
        print(x, resp)
    print('X', X)
    # print(X[0][:50])
    # print(resp)
