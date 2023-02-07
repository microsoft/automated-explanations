from typing import Callable, List
import imodelsx
import numpy as np
from spacy.lang.en import English
from os.path import dirname, join
import os.path
from joblib import Memory
methods_dir = dirname(os.path.abspath(__file__))
location = join(dirname(dirname(methods_dir)), 'results', 'cache_ngrams')
memory = Memory(location, verbose=0)

def explain_ngrams(
        X: List[str],
        mod,
        ngrams: int = 3,
        all_ngrams: bool = True,
        num_top_ngrams: int = 100
) -> List[str]:
    """Note: this caches the call that gets the scores
    """
    # get all ngrams
    tok = English()
    X_str = ' '.join(X)
    ngrams_list = imodelsx.util.generate_ngrams_list(
        X_str, ngrams=ngrams, tokenizer_ngrams=tok, all_ngrams=all_ngrams)
    
    # get unique ngrams
    ngrams_list = sorted(list(set(ngrams_list)))
    print(f'{ngrams_list=}')

    # compute scores
    # call_cached = memory.cache(mod.__call__)
    # ngram_scores = call_cached(mod(ngrams_list))
    ngram_scores = mod(ngrams_list)
    print(f'{ngram_scores=}')
    scores_top_idxs = np.argsort(ngram_scores)[::-1]
    return np.array(ngrams_list)[scores_top_idxs][:num_top_ngrams].flatten().tolist()


if __name__ == '__main__':
    def mod(X):
        return np.arange(len(X))
    explanation = explain_ngrams(
        ['test input', 'input2'],
        mod
    )
    print(explanation)
