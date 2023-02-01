from typing import Callable, List
import imodelsx
import numpy as np
from spacy.lang.en import English


def explain_ngrams(
        X: List[str],
        mod,
        ngrams: int = 3,
        all_ngrams: bool = True,
        num_top_ngrams: int = 100
):
    tok = English()
    X_str = ' '.join(X)
    ngrams_list = imodelsx.util.generate_ngrams_list(
        X_str, ngrams=ngrams, tokenizer_ngrams=tok, all_ngrams=all_ngrams)
    
    # get unique ngrams
    ngrams_list = list(set(ngrams_list))
    scores = mod(ngrams_list)
    scores_top_idxs = np.argsort(scores)[::-1]
    return np.array(ngrams_list)[scores_top_idxs][:num_top_ngrams].tolist()


if __name__ == '__main__':
    def mod(X):
        return np.arange(len(X))
    explanation = explain_ngrams(
        ['test input', 'input2'],
        mod
    )
    print(explanation)
