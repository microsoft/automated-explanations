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

def load_sample_coefs():
    coefs = []
    # the wiki here is indeed sst2
    for i in tqdm(range(13)):
        p = join(CACHE_DIR, 'dl_sst2', f'cache_sst2_l{i}.jbl')
        m = joblib.load(p)
        coefs.append(m)
        print(m)
    return np.array(coefs)
        
        
        

if __name__ == '__main__':
    
    num_instances = 30000
    dataset = load_dataset('glue', 'sst2')
    y = dataset['train']['label'][:num_instances]
    
    sample_coefs = load_sample_coefs() #[13, 67349(sst2_train_set_size), 1500]
    print(sample_coefs.shape)
    
    '''
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
    '''
    
    
    