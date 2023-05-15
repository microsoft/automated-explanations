import re
import logging
from typing import List
import numpy as np
import random
from tqdm import tqdm
import pickle as pkl
from os.path import dirname, join
import os.path
import torch.cuda
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM, AutoTokenizer
from datasets import load_dataset

from sasc.modules.dictionary_learning import sparsify_PyTorch
from sasc.modules.dictionary_learning.utils import batch_up
import joblib
from sasc.config import CACHE_DIR

modules_dir = dirname(os.path.abspath(__file__))
SAVE_DIR_DICT = join(modules_dir, 'dictionary_learning')

class DictionaryModule():

    def __init__(self, layer_idx: int = 4, reg: float = 0.3, factor_idx: int = 2, task: str = 'wiki'):
        """
        Params
        ------
        factor_idx: int
            Which dictionary factor to predict
        layer_idx: int = 0
            Emb from which layer to predict
        """

        # hyperparams for loaded model
        self.checkpoint = 'bert-base-uncased'
        self.tokenizer = BertTokenizerFast.from_pretrained(self.checkpoint)
        # hyperparams for sparse code inference
        self.reg = reg

        self.factor_idx = factor_idx
        self.layer_idx = layer_idx
        self.task = task
        
        self.words = []
        self.word_to_sentence = {}
        self.sentence_to_word = {}
        self.sentences_str = []
        self.len_ls = []

        # load pretrained dictionary
        self.basis = torch.from_numpy(np.load(join(SAVE_DIR_DICT, 'dictionaries', 'example_dict_short.npy'))).cuda() #size: (768, 1500)

    
    def _cache_end2end(self, X: List[str], calc_ngram):
        """
        Returns
        -------
        resp: List[float]
        """
        #cache_file = join(CACHE_DIR, 'dl_all_ngrams',f'cache_ngrams_{self.task}_l{self.layer_idx}.jbl')
        cache_file = join(SAVE_DIR_DICT,f'cache_ngrams_l{self.layer_idx}.jbl')
        
        if calc_ngram and os.path.exists(cache_file):
            ahat_at_layer = joblib.load(cache_file)
            resp = ahat_at_layer[:, self.factor_idx]
        else:
            ahat = []
            model = BertModel.from_pretrained(self.checkpoint)
            torch.cuda.set_device(0)
            device = torch.device("cuda:{}".format(0))
            model = model.to(device)

            sentences_shards = list(batch_up(X, batch_size = 1000))

            for shard_num in tqdm(range(len(sentences_shards)), 'shards'):
                self.len_ls = []
                embs = []
                sentences_batched = list(batch_up(sentences_shards[shard_num], batch_size=128))

                for batch_idx in range(len(sentences_batched)):
                    batch = sentences_batched[batch_idx]
                    inputs_no_pad = self.tokenizer.batch_encode_plus(batch, add_special_tokens=False)
                    inputs_no_pad_ids = inputs_no_pad['input_ids'] #[batch_size, seq_len]
                    local_len_ls = [ len(s) for s in inputs_no_pad_ids]

                    self.len_ls.extend([ len(s) for s in inputs_no_pad_ids])

                    inputs = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', add_special_tokens=False, padding=True, truncation=True).to(device)
                    hidden_states = model(**inputs, output_hidden_states=True)[-1]

                    # collect word_embs from all layers
                    # hidden_states: [layer, batch_size, seq_len, 768]
                    hidden_at_layer = hidden_states[self.layer_idx].cpu().detach().numpy()
                    for i in range(len(hidden_at_layer)):
                        sentences_trunc = hidden_at_layer[i][:local_len_ls[i]]
                        for s in range(len(sentences_trunc)):
                            embs.append(sentences_trunc[s])

                X_set_batched = list(batch_up(embs, 100))
                X_sparse_set = [] #[total_word, 1500]
                for i in range(len(X_set_batched)):
                    batch = X_set_batched[i]
                    I_cuda = torch.from_numpy(np.stack(batch, axis=1)).cuda()
                    X_sparse = sparsify_PyTorch.FISTA(I_cuda, self.basis, self.reg, 500)[0].T
                    X_sparse_set.extend(X_sparse.cpu().detach().numpy())
                
                
                X_sparse_at_layer = np.stack(X_sparse_set) #[total_word_count, 1500]
                pt = 0
                for skip in self.len_ls:
                    sent_ahat = np.mean(X_sparse_at_layer[pt:pt+skip, :], axis=0) #[1500]
                    ahat.append(sent_ahat)
                    pt += skip

            ahat = np.stack(ahat)
            resp = ahat[:, self.factor_idx]

            if calc_ngram:
                os.makedirs(dirname(cache_file), exist_ok=True)
                joblib.dump(ahat, cache_file)  
        
        assert(len(resp) == len(X))
        return resp

    def _end2end(self, X: List[str]):
        """
        Returns
        -------
        resp: List[float]
        """
        resp = []

        model = BertModel.from_pretrained(self.checkpoint)
        torch.cuda.set_device(0)
        device = torch.device("cuda:{}".format(0))
        model = model.to(device)

        sentences_shards = list(batch_up(X, batch_size = 1000))

        for shard_num in tqdm(range(len(sentences_shards)), 'shards'):
            #self.sentences_str = []
            #self.words = []
            #self.word_to_sentence = {}
            #self.sentence_to_word = {}
            self.len_ls = []
            #n1=0
            #n2=0
            embs = []

            sentences_batched = list(batch_up(sentences_shards[shard_num], batch_size=128))

            for batch_idx in range(len(sentences_batched)):
                batch = sentences_batched[batch_idx]
                inputs_no_pad = self.tokenizer.batch_encode_plus(batch, add_special_tokens=False)
                inputs_no_pad_ids = inputs_no_pad['input_ids'] #[batch_size, seq_len]
                local_len_ls = [ len(s) for s in inputs_no_pad_ids]

                self.len_ls.extend([ len(s) for s in inputs_no_pad_ids])

                inputs = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', add_special_tokens=False, padding=True, truncation=True).to(device)

                #keep track of a map between the word in each sentence 
                #to each of those sentences for convinience
                #for tokens in inputs_no_pad_ids:
                #    tokenized = self.tokenizer.convert_ids_to_tokens(tokens)
                #    self.sentences_str.append(tokenized)
                #    self.words.extend(tokenized)
                #    w_index = []
                #    for j in range(len(tokenized)):
                #        self.word_to_sentence[n2] = (n1, j)
                #        w_index.append(n2)
                #        n2+=1
                #    self.sentence_to_word[n1] = w_index
                #    n1+=1

                hidden_states = model(**inputs, output_hidden_states=True)[-1]

                # select embedding from a specific layer
                # X: [batch_size, seq_len, 768]
                hidden_at_layer = hidden_states[self.layer_idx].cpu().detach().numpy()
                for i in range(len(hidden_at_layer)):
                    sentences_trunc = hidden_at_layer[i][:local_len_ls[i]]
                    for s in range(len(sentences_trunc)):
                        embs.append(sentences_trunc[s])

            X_set_batched = list(batch_up(embs, 100))
            X_sparse_set = [] #[total_word, 1500]
            for i in range(len(X_set_batched)):
                batch = X_set_batched[i]
                I_cuda = torch.from_numpy(np.stack(batch, axis=1)).cuda()
                X_sparse = sparsify_PyTorch.FISTA(I_cuda, self.basis, self.reg, 500)[0].T
                X_sparse_set.extend(X_sparse.cpu().detach().numpy())
            
            # select a specific factor
            factor_slice = [x[self.factor_idx] for x in X_sparse_set] #[total_word_count]
            # calculate sententence a_hat as average of tokens a_hat
            pt = 0
            for skip in self.len_ls:
                sent_ahat = np.mean(factor_slice[pt:pt+skip])
                resp.append(sent_ahat)
                pt += skip

        return resp
        
    def __call__(self, X: List[str], calc_ngram=False) -> np.ndarray:
        """
        Returns a scalar continuous response for each element of X
        """
        '''
        resp = []
        # get embeddings
        embs = self._get_embs(X) #[total_word_count, 768]
        torch.cuda.empty_cache()

        # sparse code inference
        X_set_batched = list(batch_up(embs, 100))
        X_sparse_set = []
        for i in tqdm(range(len(X_set_batched)), 'sparse_inference'):
            batch = X_set_batched[i]
            I_cuda = torch.from_numpy(np.stack(batch, axis=1)).cuda()
            X_sparse = sparsify_PyTorch.FISTA(I_cuda, self.basis, self.reg, 500)[0].T
            X_sparse_set.extend(X_sparse.cpu().detach().numpy())
        

        # select a specific factor
        factor_slice = [x[self.factor_idx] for x in X_sparse_set] #[total_word_count]

        # calculate sententence a_hat as average of tokens a_hat
        pt = 0
        for skip in self.len_ls:
            #print(pt, pt+skip)
            sent_ahat = np.mean(factor_slice[pt:pt+skip])
            resp.append(sent_ahat)
            pt += skip
        

        '''
        #torch.cuda.empty_cache()
        #resp = self._end2end(X)
        resp = self._cache_end2end(X, calc_ngram)
        

        return np.array(resp)
    
    def _get_embs(self, X: List[str]):
        """
        Returns
        -------
        X: List[str]
        embs: List[Tensor(seq_len, 768)]
            (n_examples, seq_len(varied acorss examples), 768)
        """

        model = BertModel.from_pretrained(self.checkpoint)
        torch.cuda.set_device(0)
        device = torch.device("cuda:{}".format(0))
        model = model.to(device)

        sentences_shards = list(batch_up(X, batch_size = 1000))
        
        n1=0
        n2=0
        embs = []

        for shard_num in tqdm(range(len(sentences_shards)), 'shards'):
            sentences_batched = list(batch_up(sentences_shards[shard_num], batch_size=64))
            for batch_idx in tqdm(range(len(sentences_batched)), 'collect hidden states'):
                batch = sentences_batched[batch_idx]
                inputs_no_pad = self.tokenizer.batch_encode_plus(batch, add_special_tokens=False)
                inputs_no_pad_ids = inputs_no_pad['input_ids'] #[batch_size, seq_len]
                local_len_ls = [ len(s) for s in inputs_no_pad_ids]

                self.len_ls.extend([ len(s) for s in inputs_no_pad_ids])

                inputs = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', add_special_tokens=False, padding=True, truncation=True).to(device)

                #keep track of a map between the word in each sentence 
                #to each of those sentences for convinience
                for tokens in inputs_no_pad_ids:
                    tokenized = self.tokenizer.convert_ids_to_tokens(tokens)
                    self.sentences_str.append(tokenized)
                    self.words.extend(tokenized)
                    w_index = []
                    for j in range(len(tokenized)):
                        self.word_to_sentence[n2] = (n1, j)
                        w_index.append(n2)
                        n2+=1
                    self.sentence_to_word[n1] = w_index
                    n1+=1

                hidden_states = model(**inputs, output_hidden_states=True)[-1]

                # select embedding from a specific layer
                # X: [batch_size, seq_len, 768]
                hidden_at_layer = hidden_states[self.layer_idx].cpu().detach().numpy()
                for i in range(len(hidden_at_layer)):
                    sentences_trunc = hidden_at_layer[i][:local_len_ls[i]]
                    for s in range(len(sentences_trunc)):
                        embs.append(sentences_trunc[s])

        return embs

    def _get_top_n_activate_word_context(self, X: List[str], n: int = 3):
        """
        Returns a list of activation info for debug / verbose purpose
        """
        my_df = []
        # get embeddings
        embs = self._get_embs(X) #[n_samples*seq_len(varies), 768]
        torch.cuda.empty_cache()

        # sparse code inference
        X_set_batched = list(batch_up(embs, 100))
        X_sparse_set = []
        for i in tqdm(range(len(X_set_batched)), 'sparse_inference'):
            batch = X_set_batched[i]
            I_cuda = torch.from_numpy(np.stack(batch, axis=1)).cuda()
            X_sparse = sparsify_PyTorch.FISTA(I_cuda, self.basis, self.reg, 500)[0].T
            X_sparse_set.extend(X_sparse.cpu().detach().numpy())

        # select a specific factor
        factor_slice = [x[self.factor_idx] for x in X_sparse_set] #[total_word_count]
        indx = np.argsort(-np.array(factor_slice))[:n]
        for i in indx:
            word = self.words[i]
            act = factor_slice[i]
            sent_position, word_position = self.word_to_sentence[i]
            sentence = self.sentences_str[sent_position]
            d = {'word':word,'index':word_position,'sent_index':sent_position,'score':act,'sent':self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(sentence))}
            my_df.append(d)
        
        return my_df
    
    def get_train_wiki_texts(self):
        random.seed(99)
        max_seq_length = 64
        num_instances = 300000

        tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, use_fast=True
        )
        dataset = load_dataset("wikitext", 'wikitext-103-v1')
        articles = []
        article = ''
        for text in dataset['train']['text']:
        # text = dataset['train'][i]['text']
            if re.match(r"^ = [^=]", text):
                articles.append(article)
                article = ''
            article = article + text
        articles_long = [ar for ar in articles if len(ar) > 2000]
        sentences_sample = random.sample(articles_long, int(len(articles_long)))

        tokens = []
        for arr in sentences_sample:
            blocks = batch_up(arr, 2000)
            for b in blocks:
                tokens.extend(tokenizer(b, add_special_tokens=False)['input_ids'])

        tokens_batch = batch_up(tokens, max_seq_length)

        sentences = []
        for batch in tokens_batch:
            sentences.append(tokenizer.decode(batch))
            if len(sentences) > num_instances:
                break
        np.save(join(SAVE_DIR_DICT, 'train_sentences.npy') , sentences) 

    
if __name__ == '__main__':
    
    
    #X = ['born on september 6',
    #     'made up my mind',
    #     'light up my life',
    #     'will not change my mind',
    #     'you lost your mind']
    # create train data and save file to dir
    #mod._get_train_wiki_texts()
    #df = mod._get_top_n_activate_word_context(X)
    #print(df)
    #mod._get_embs(X)
    #resp = mod(X)
    #print(resp)
    
    for i in tqdm(range(13)):
        mod = DictionaryModule(layer_idx=i, factor_idx=15)
        dataset = load_dataset('glue', 'sst2')
        X = dataset['train']['sentence']
        mod._cache_end2end(X=X, calc_ngram=True)
