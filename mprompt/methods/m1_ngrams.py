from typing import Callable, List
import imodelsx
import numpy as np
from spacy.lang.en import English
from os.path import dirname, join
import os.path
import pickle as pkl
import inspect
from mprompt.config import CACHE_DIR
import mprompt.data.data


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
        X_str,
        ngrams=ngrams,
        tokenizer_ngrams=tok,
        all_ngrams=all_ngrams
    )

    # get unique ngrams
    ngrams_list = sorted(list(set(ngrams_list)))
    # print(f'{ngrams_list=}')
    print('num ngrams', len(ngrams_list), 'examples', ngrams_list[:5])

    # compute scores and cache...
    # fmri should cache all preds together, since they are efficiently computed together
    if args.module_name == 'fmri':
        cache_file = join(CACHE_DIR, 'cache_ngrams',
                            f'{args.module_name}_{args.subject}.pkl')
    if args.module_name == 'old_fmri':
        cache_file = join(CACHE_DIR, 'cache_ngrams',
                            f'{args.module_name}.pkl')
    else:
        cache_file = join(CACHE_DIR, 'cache_ngrams',
                            f'{args.module_name}_{args.module_num}.pkl')
    if use_cache and os.path.exists(cache_file):
        ngram_scores = pkl.load(open(cache_file, 'rb'))
    else:
        call_parameters = inspect.signature(mod.__call__).parameters.keys()
        print('predicting all ngrams...')
        if 'return_all' in call_parameters:
            ngram_scores = mod(ngrams_list, return_all=True)
        else:
            ngram_scores = mod(ngrams_list)
            
        if use_cache:
            os.makedirs(dirname(cache_file), exist_ok=True)
            pkl.dump(ngram_scores, open(cache_file, 'wb'))

    # multidimensional predictions
    if len(ngram_scores.shape) > 1 and ngram_scores.shape[1] > 1:
        ngram_scores = ngram_scores[:, args.module_num]

    # add noise to ngram scores
    if args.noise_ngram_scores > 0:
        scores_top_100 = np.sort(ngram_scores)[::-1][:100]
        std_top_100 = np.std(scores_top_100)
        rng = np.random.default_rng(args.seed)
        ngram_scores += rng.normal(
            scale=std_top_100 * args.noise_ngram_scores,
            size=ngram_scores.shape,
        )

    # restrict top ngrams to alternative corpus
    if args.module_num_restrict >= 0:
        print('before', ngrams_list)
        text_str_list_alt = mprompt.data.data.get_relevant_data(
            args.module_name, args.module_num_restrict)
        ngrams_set_alt = set(imodelsx.util.generate_ngrams_list(
            ' '.join(text_str_list_alt),
            ngrams=ngrams,
            tokenizer_ngrams=tok,
            all_ngrams=all_ngrams
        ))
        idxs_to_keep = np.array([i for i, ngram in enumerate(ngrams_list) if ngram in ngrams_set_alt])
        ngrams_list = [ngrams_list[i] for i in idxs_to_keep]
        ngram_scores = ngram_scores[idxs_to_keep]
        print('after', ngrams_list)

    # print(f'{ngram_scores=}')
    scores_top_idxs = np.argsort(ngram_scores)[::-1]
    ngrams_top = np.array(ngrams_list)[scores_top_idxs][:num_top_ngrams]
    return ngrams_top.flatten().tolist()


if __name__ == '__main__':
    def mod(X):
        return np.arange(len(X)).astype(float)

    class a:
        noise_ngram_scores = 3
        seed = 100
        module_name = 'emb_diff_d3'
        module_num = 0
        module_num_restrict = -1

    explanation = explain_ngrams(
        a(),
        ['and', 'i1', 'i2', 'i3', 'i4'],
        mod,
        use_cache=False,

    )
    print(explanation)
