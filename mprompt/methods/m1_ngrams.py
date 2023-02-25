from typing import Callable, List
import imodelsx
import numpy as np
from spacy.lang.en import English
from os.path import dirname, join
import os.path
import pickle as pkl
import inspect
from mprompt.config import CACHE_DIR

def explain_ngrams(
        args,
        X: List[str],
        mod,
        ngrams: int = 3,
        all_ngrams: bool = True,
        num_top_ngrams: int = 75,
        use_cache: bool = True,
) -> List[str]:
    """Note: this caches the call that gets the scores
    """
    # get all ngrams
    tok = English(max_length=10e10)
    X_str = ' '.join(X)
    ngrams_list = imodelsx.util.generate_ngrams_list(
        X_str, ngrams=ngrams, tokenizer_ngrams=tok, all_ngrams=all_ngrams)
    
    # get unique ngrams
    ngrams_list = sorted(list(set(ngrams_list)))
    # print(f'{ngrams_list=}')
    print('num ngrams', len(ngrams_list), 'examples', ngrams_list[:5])

    # compute scores and cache...
    # fmri should cache all preds together, since they are efficiently computed together
    if args.module_name == 'fmri':
        cache_file  = join(CACHE_DIR, 'cache_ngrams', f'{args.module_name}.pkl')
    else:
        cache_file = join(CACHE_DIR, 'cache_ngrams', f'{args.module_name}_{args.module_num}.pkl')
    if os.path.exists(cache_file):
        ngram_scores = pkl.load(open(cache_file, 'rb'))
    else:
        call_parameters = inspect.signature(mod.__call__).parameters.keys()
        print('predicting all ngrams...')
        if 'return_all' in call_parameters:
            ngram_scores = mod(ngrams_list, return_all=True)
        else:
            ngram_scores = mod(ngrams_list)
        os.makedirs(dirname(cache_file), exist_ok=True)
        pkl.dump(ngram_scores, open(cache_file, 'wb'))

    # multidimensional predictions
    if len(ngram_scores.shape) > 1 and ngram_scores.shape[1] > 1:
        ngram_scores = ngram_scores[:, args.module_num]

    # add noise to ngram scores
    if args.noise_ngram_scores > 0:
        # scores_top_100
        # std = np.std()
        ngram_scores += np.random.normal(
            scale=args.noise_ngram_scores, size=ngram_scores.shape)

    # print(f'{ngram_scores=}')
    scores_top_idxs = np.argsort(ngram_scores)[::-1]
    ngrams_top = np.array(ngrams_list)[scores_top_idxs][:num_top_ngrams]
    return ngrams_top.flatten().tolist()


if __name__ == '__main__':
    def mod(X):
        return np.arange(len(X))
    explanation = explain_ngrams(
        ['test input', 'input2'],
        mod
    )
    print(explanation)
