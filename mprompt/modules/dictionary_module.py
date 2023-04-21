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
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict

from mprompt.modules.dictionary_learning import sparsify_PyTorch
import joblib
import scipy as sp
import sklearn
import json


modules_dir = dirname(os.path.abspath(__file__))
SAVE_DIR_DICT = join(modules_dir, 'dictionary_learning')

low_level = [(4, 2), (4, 16), (4, 33), (0, 30), (2, 30), (4, 30)]
mid_level = [(10, 42), (10, 50), (6, 86), (10, 102), (8, 125), (10, 184), (10, 195), (4, 13), (6, 13), (10, 13)]
high_level = [(10, 297), (10, 322), (10, 386)]
levels = low_level + mid_level + high_level

class DictionaryModule():

    def __init__(self, layer_idx: int = 4, reg: float = 0.3, factor_idx: int = 2):
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
    
        self.words = []
        self.word_to_sentence = {}
        self.sentence_to_word = {}
        self.sentences_str = []
        self.len_ls = []

        # load pretrained dictionary
        self.basis = torch.from_numpy(np.load(join(SAVE_DIR_DICT, 'dictionaries', 'example_dict_short.npy'))).cuda() #size: (768, 1500)

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
    
    def _cache_end2end(self, X: List[str], calc_ngram):
        """
        Returns
        -------
        resp: List[float]
        """
        cache_file = join(SAVE_DIR_DICT, f'cache_ngrams_l{self.layer_idx}.jbl')
        
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

def batch_up(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

def get_train_wiki_texts():
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
        
def batch_up(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

def get_exp_data(factor_idx: int, factor_layer: int) -> List[str]:
    low_p = ['dl_l4_i2/268f564b66e8a186cd2f9c2876925b1b753c266429fadd81be54e86b997bfe18',
            'dl_l4_i16/3a4b18cdb341ebf9bf2e91228aef3a2f4b21e39032eac673e4929e1fd33d1a1c',
            'dl_l4_i33/694c301c43123ca4788f6ca8b494f82f0d718dbfea71ad50ea0862e0fe5b60ed',
            #'dl_l0_i30/69ada07b0fa60795b132b2e0d73c26fdf6e94c3cf4dbd6c13cdae8eb75251827',
            #'dl_l2_i30/a5d4e81e190b4c7c5249ae985642129e7861f9b80ff4600e054a1b6f8c279215',
            'dl_l4_i30/814af8345dc0e3d0b8a5a3756c92c023bd1510c2e853924307eaf6cf94f210f6']

    mid_p = ['dl_l10_i42/4a3aee14a9a8a0464fc5e89d16657ddb8a07cb366d848bcc34d969ab40410d2f',
            'dl_l10_i50/34e4a15483dec6c9e87f087a48d6293c2c7f9ba978e0145c35929fc5e08daa01',
            'dl_l6_i86/c4cacbee9894aefe4f0fbc099a8ad181f508ece7ca0d70a288b4e4053e3b50ba',
            'dl_l10_i102/7a4c700acfd72a29c6278a8a401754a0bf2f970b41b0e119715dff82489dcd0e',
            'dl_l8_i125/e933cdd746cc8776b3a4268b5a700ec0d2601652da7d1231d8f5b199b10aad7b',
            'dl_l10_i184/606adb5ae6d97287263e32875249cc45b8cee54ee21af949ea04560a0f0d0532',
            'dl_l10_i195/8d0bf634dba071a6fcc274096a842bb2728fb0fdc12ae76d7e1de9625a67cea2',
            #'dl_l4_i13/c21294ec3f2b6bccb37a1e8585e8009f1522421784a0bc1e481474b0d3e009a2',
            #'dl_l6_i13/01cb8bfbc06d531eabd9e11f67a39c100cfb547ba054a98161435fe474b49818',
            'dl_l10_i13/16ac8b76300e0b45bef28d20bc6d83e88514839d5ec1f1e58853afd517087291']

    high_p = ['dl_l10_i297/089863daca700285fb97b75fb4f6380a8332a11a1aa5772804dc982dfad9b015',
            'dl_l10_i322/e02f845df54536abf3499bcc319f59bf693c8b6501136b79f8f742d81a3517e5',
            'dl_l10_i386/330771ba7e87a1bfd0278644d90e5aa27066fad1f90b2ecdaaa21a4cf36661a1']

    all_ps = low_p + mid_p + high_p
    RESULT_DIR = join('/'.join(os.path.dirname(modules_dir).split('/')[:-1]), 'results')

    exps = None
    for p in all_ps:
        seg = p.split('/')
        key = f'dl_l{factor_layer}_i{factor_idx}'
        if seg[0] == key:
            with (open(join(RESULT_DIR, p, 'results.pkl'), "rb")) as openfile:
                f = pkl.load(openfile)
            exps = f['explanation_init_strs']
            break

    control_strs = {}
    control_strs['strs_added'] = []
    control_strs['strs_removed'] = []
    # load control data
    for p in all_ps:
        seg = p.split('/')
        sub_seg = seg[0].split('_')
        key = f'i{factor_idx}'
        # collect data from all the other factor
        if sub_seg[-1] != key:
            with (open(join(RESULT_DIR, p, 'results.pkl'), "rb")) as openfile:
                f = pkl.load(openfile)
            control_strs['strs_added'].extend(f['top_strs_added'])
            control_strs['strs_removed'].extend(f['top_strs_removed'])

    # load baseline explanations
    with open(join(SAVE_DIR_DICT, 'baseline_exp.json'), 'r') as fp:
        baseline_exp = json.load(fp)
    for k, d in baseline_exp.items():
        if d['layer'] == factor_layer and d['factor_idx'] == factor_idx:
            exps.extend([d['exp']])
            break
    
    assert(exps != None)

    return exps, control_strs
    

def write_baseline_exp():
    low_exp = ['mind. Noun. the element of a person that enables them to be aware of the world and their experiences.',
            'park. Noun. a common first and last name.',
            'light. Noun. the natural agent that stimulates sight and makes things visible.',
            'left. Adjective or Verb. Mixed senses.',
            'left. Verb. leaving, exiting.',
            'left. Verb. leaving, exiting.']

    mid_exp = ['something unfortunate happened.',
            'Doing something again, or making something new again.',
            'Consecutive years, used in foodball season naming.',
            'African names.',
            'Describing someone in a paraphrasing style. Name, Career.',
            'Institution with abbreviation.',
            'Consecutive of noun (Enumerating).',
            'Numerical values.',
            'Close Parentheses.',
            'Unit exchange with parentheses.']

    high_exp = ['repetitive structure detector.',
            'biography, someone born in some year...',
            'war.']
    r = {}
    exps = low_exp + mid_exp + high_exp

    for i, (l, pos) in enumerate(levels):
        r[i] = {}
        r[i]['layer'] = l
        r[i]['factor_idx'] = pos
        r[i]['exp'] = exps[i]
    
    with open(join(SAVE_DIR_DICT, 'baseline_exp.json'), 'w') as fp:
        json.dump(r, fp)
    
if __name__ == '__main__':
    
    mod = DictionaryModule(layer_idx=10, factor_idx=322)
    X = ['born on september 6',
         'made up my mind',
         'light up my life',
         'will not change my mind',
         'you lost your mind']
    # create train data and save file to dir
    #mod._get_train_wiki_texts()
    # save baseline explanation file to dir
    #write_baseline_exp()
    #exp, data = get_exp_data(2, 4)
    #df = mod._get_top_n_activate_word_context(X)
    #print(df)
    #mod._get_embs(X)
    resp = mod(X)
    print(resp)
