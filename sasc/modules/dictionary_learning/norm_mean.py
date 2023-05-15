import joblib
import os
import pickle
from os.path import dirname, join
from mprompt.config import CACHE_DIR
import numpy as np
from tqdm import tqdm

cur_dir = dirname(os.path.abspath(__file__))
SAVE_DIR_DICT = cur_dir

def get_mean(store_dir):
    mean_file = join(SAVE_DIR_DICT, 'wiki_ngram_mean.pkl')
    
    if not os.path.exists(mean_file):
        ngram_dir = join(CACHE_DIR, store_dir)
        mean_lst = [] #[layer_size, 1500]
        for layer in tqdm(range(13)):
            p = join(ngram_dir, f'cache_ngrams_l{layer}.jbl')
            m = joblib.load(p) #[ngram_size, 1500]
            mean_m = np.mean(m, axis=0) #[1500]
            mean_lst.append(mean_m)
            
        with open(mean_file, 'wb') as fp:
            pickle.dump(mean_lst, fp)
    
    with open(mean_file, 'rb') as fp:
        data = pickle.load(fp)
        
    return data

def get_syn_score(result_dir):
    unique_p = [['null' for _ in range(1500)] for _ in range(13)]
    scores = [[0 for _ in range(1500)] for _ in range(13)]
    
    for factor_layer in range(13):
        for root, dirs, files in os.walk(join(CACHE_DIR, result_dir, f'dl_l{factor_layer}')):
            for name in files:
                if name == 'results.pkl': 
                    full_path = os.path.abspath(os.path.join(root, name))
                    segs = full_path.split('/')
                    factor_idx = int(segs[7][1:])
                    unique_p[factor_layer][factor_idx] = full_path
    
    for factor_layer in range(13):
        for factor_idx in range(1500):
            result_p = unique_p[factor_layer][factor_idx]
            with (open(result_p, "rb")) as openfile:
                f = pickle.load(openfile)
                scores[factor_layer][factor_idx] = f['top_score_synthetic']
    return scores

layer_idx_lst = [4, 4, 4, 4, 6, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10]
factor_idx_lst = [2, 16, 33, 30, 86, 125, 13, 42, 50, 102, 184, 195, 297, 322, 386]
    
def get_baseline_syn_score(result_dir):
    eval_all_ps = ['eval_dl_l4_i2/7343e8443347292dbbc1c7de8452dded36e694afd1bc9bca56778a3f0ad2dad2',
                'eval_dl_l4_i16/44efd61ddd3a509f7366bc21724e588b6ea9de4cf3f5da7f9c2e2f13bef3b3ec',
                'eval_dl_l4_i33/fc044e23e6b9a71cc06fbc00402f2d689d939378dfc17e8dfd8348b06922277e',
                'eval_dl_l4_i30/42ab3bb254bc6e5e48a2adc1853ae8380be46c12b5c075ae6887eed5bb1977a0',
                'eval_dl_l6_i86/edda0c5c9cb456ef40780efd95d60c66e5ffd350cef5058533a9bb28d545efe7',
                'eval_dl_l8_i125/7176f576d5d897399553c1903aa47c14a23f2dc5b521a133a0d372b5efc83ec8',
                'eval_dl_l10_i13/ba9bd081cc6cff60e64d09b538efaccce81427daa71118c01b0bc3965e29abe1',
                'eval_dl_l10_i42/56703ced437364eb7f92fa503ef8be83b7fe77e8a6374cb0cd64b9eecf9f8db3',
                'eval_dl_l10_i50/daf4ed3f428c16a57940454c75f14dad7d9e8b1ef0e7ea9b4f9aa4719cbc2b67',
                'eval_dl_l10_i102/c2b764a4f772f7c2e2fdd470dbc3bfa041ca0e603b39c62aa6a7adc2b47c8839',
                'eval_dl_l10_i184/29cea206934c0b64cbf2042e4d25a2e7b5443e547df943cfb235863f67694e5d',
                'eval_dl_l10_i195/7410ac2404b07b060cb63b2a4c5e42c422992efa305620a339434fcfd67131ca',
                'eval_dl_l10_i297/3bf50a31b1326b1a005111031f6fc41f0dab3e12d239401adc162d83a3139bfa',
                'eval_dl_l10_i322/af61a05f073ac4cbbb8f407cbdc610050b681933b6953c6b76cf3609e55482d6',
                'eval_dl_l10_i386/ac6f1e90dd48fb9dd825f25a9b4bd5eaae8dab43d55bccc4e0ca026ba1fc243c']
    data = []
    for i, p in enumerate(eval_all_ps):
        with (open(join(CACHE_DIR, result_dir, 'eval', p, 'eval.pkl'), "rb")) as openfile:
            f = pickle.load(openfile)
            b_score = f['score_synthetic'][-1]
            
        obj = {'layer': layer_idx_lst[i], 'factor_idx': factor_idx_lst[i], 'perc_syn_score': b_score}
        data.append(obj)
    return data
    
    
if __name__ == '__main__':
    
    ngram_dir = 'ah-module-prompt/mprompt/modules/dictionary_learning'
    result_dir = 'ah-module-prompt/results'
    mean_lst = get_mean(ngram_dir)
    mean_lst = np.absolute(np.array(mean_lst))
    
    scores = get_syn_score(result_dir)
    perc = np.divide(np.array(scores), mean_lst)
    print(perc.shape)
    
    with open(join(SAVE_DIR_DICT, 'wiki_our_syn_perc_mean_score.pkl'), 'wb') as fp:
        pickle.dump(perc, fp)
        
    baselines = get_baseline_syn_score(result_dir)
    b_perc = []
    for item in baselines:
        l = item['layer']
        f = item['factor_idx']
        b_score = item['perc_syn_score']
        mean_score = mean_lst[l][f]
        obj = {'layer': l, 'factor_idx': f, 'perc_syn_score': b_score/mean_score}
        b_perc.append(obj)
    print(b_perc)
    
    with open(join(SAVE_DIR_DICT, 'wiki_baseline_syn_perc_mean_score.pkl'), 'wb') as fp:
        pickle.dump(b_perc, fp)
    '''
    with open(join(SAVE_DIR_DICT, 'wiki_normalized_syn_scores', 'wiki_baseline_syn_perc_score.pkl'), 'rb') as fp:
        b_perc = pickle.load(fp)
        
    with open(join(SAVE_DIR_DICT, 'wiki_normalized_syn_scores', 'wiki_our_syn_perc_score.pkl'), 'rb') as fp:
        perc = pickle.load(fp)
    
    print(b_perc)
    print(perc)
    '''
        
        
        
    
    
    

    
        
        
        
    '''
    task = 'sst2'
    layer_idx = 12
    
    file_dir = '/var/tmp/ah_cache/dl_all_ngrams'
    cache_file = join(CACHE_DIR, f'cache_ngrams_{task}_l{layer_idx}.jbl')
    ahat_at_layer = joblib.load(cache_file)
    print(ahat_at_layer.shape)
    '''

