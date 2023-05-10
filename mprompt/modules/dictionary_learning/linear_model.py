import os
import imodelsx
import joblib
import pickle as pkl
import numpy as np
from tqdm import tqdm
from os.path import dirname, join
from datasets import load_dataset
from spacy.lang.en import English
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


from mprompt.config import CACHE_DIR

cur_dir = dirname(os.path.abspath(__file__))

def get_unique_ngram(X_str, tok):
    # get all ngrams
    ngrams_list = imodelsx.util.generate_ngrams_list(
        X_str,
        ngrams=3,
        tokenizer_ngrams=tok,
        all_ngrams=True
    )

    # get unique ngrams
    ngrams_list = sorted(list(set(ngrams_list)))
    
    return ngrams_list

def load_ngram_coefs():
    coefs = []
    for i in tqdm(range(13)):
        p = join(CACHE_DIR, 'dl_all_ngrams', f'cache_ngrams_sst2_l{i}.jbl')
        m = joblib.load(p)
        coefs.append(m)
        print(m.shape)
    return np.array(coefs)


def featurize(X, ngram_coefs, ngram_idx_map):
    feat = []
    tok = English(max_length=10e10)
    ngram_size, factor_size = ngram_coefs[0].shape
    ngram_coefs = np.transpose(ngram_coefs, (1, 0, 2)) #[ngram_size, 13, factor_size]
    print(ngram_coefs.shape)
    
    for sent in X:
        sent_f = np.zeros((ngram_size, 13, factor_size))
        sent_ngrams = get_unique_ngram(sent, tok)
        print(sent)
        for w in sent_ngrams:
            print(w, ngram_idx_map[w])
            sent_f[ngram_idx_map[w], :, :] = ngram_coefs[ngram_idx_map[w], :, :]
        feat.append(sent_f)
        break
    
    return np.array(feat)
        
        
        

if __name__ == '__main__':
    
    num_instances = 30000
    dataset = load_dataset('glue', 'sst2')
    X = dataset['train']['sentence'][:num_instances]
    y = dataset['train']['label'][:num_instances]
    
    with open(join(cur_dir, 'sst2_unique_ngram_list.pkl'), 'rb') as fp:
        unique_ngram_list = pkl.load(fp)
    ngram_idx_map = { w:i for i, w in enumerate(unique_ngram_list)}
    
    ngram_coefs = load_ngram_coefs()
    print(ngram_coefs.shape)
    f = featurize(X, ngram_coefs, ngram_idx_map)
    print(f.shape)
    
    
    params = {
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    }

    model = SGDClassifier(loss="log_loss", penalty="l1")
    clf = GridSearchCV(model, param_grid=params)
    
    # Scale the data to be between -1 and 1
    scaler = StandardScaler()
    scaler.fit(f)
    f = scaler.transform(f)

    clf.fit(f, y)
    print(clf.best_score_)
    print(clf.best_estimator_)
    
    
    