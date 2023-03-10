import logging
from typing import List
import numpy as np
from tqdm import tqdm
from spacy.lang.en import English
import pickle as pkl
from os.path import dirname, join
import os.path
import torch.cuda
import numpy.random
from transformers import BertTokenizerFast, BertModel, BertForMaskedLM


# import sparsify
from dictionary_learning import sparsify_PyTorch
import scipy as sp
import sklearn


modules_dir = dirname(os.path.abspath(__file__))
SAVE_DIR_DICT = join(modules_dir, 'dictionary_learning')
#SAVE_DIR_FMRI = '/home/chansingh/mntv1/deep-fMRI/opt_model'
#NUM_TOP_VOXELS = 500

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

        #sentences_str = []
        #words = []
        #len_ls = []
        #word_to_sentence = {}
        #sentence_to_word = {}
        n1=0
        n2=0
        collect_sent_emb = []
        embs = []
        for i in tqdm(range(len(X))):
            # inputs_no_pad = {'input_ids', 'token_type_ids', 'attention_mask'}
            inputs_no_pad = self.tokenizer.encode_plus(X[i], add_special_tokens=False)
            inputs_no_pad_ids = inputs_no_pad['input_ids'] #[1, seq_len]
            self.len_ls.append(len(inputs_no_pad_ids))

            inputs = self.tokenizer.encode_plus(X[i],return_tensors='pt', add_special_tokens=False,padding=True,truncation=True).to(device)

            #keep track of a map between the word in each sentence 
            #to each of those sentences for convinience
            tokenized = self.tokenizer.convert_ids_to_tokens(inputs_no_pad_ids)
            self.sentences_str.append(tokenized)
            self.words.extend(tokenized)
            w_index = []
            for j in range(len(tokenized)):
                self.word_to_sentence[n2] = (n1, j)
                w_index.append(n2)
                n2+=1
            self.sentence_to_word[n1] = w_index
            n1+=1
            
            #print(sentences_str)
            #print(words)
            #print(word_to_sentence)
            #print(w_index)
            #print(sentence_to_word)

            hidden_states = model(**inputs, output_hidden_states=True)[-1]

            # select embedding from a specific layer
            # X: [1, seq_len, 768]
            hidden_at_layer = hidden_states[self.layer_idx].cpu().detach().numpy()
            collect_sent_emb.append(hidden_at_layer)

        for i in range(len(collect_sent_emb)): #loop thru sentences
            sent_emb_trunc = collect_sent_emb[i][:self.len_ls[i]]
            sent_emb_trunc = sent_emb_trunc.squeeze(0)
            for t in range(len(sent_emb_trunc)): #loop thru tokens
                embs.append(sent_emb_trunc[t])
        
        return embs

    def __call__(self, X: List[str]) -> np.ndarray:
        """
        Returns a scalar continuous response for each element of X
        """
        resp = []
        # get embeddings
        embs = self._get_embs(X) #[n_samples, seq_len(varies), 768]
        torch.cuda.empty_cache()

        # sparse code inference
        # I_cuda: [768, total_word_count]
        I_cuda = torch.from_numpy(np.stack(embs, axis=1)).cuda()
        # X_sparse (a_hat): [total_word_count, 1500]
        X_sparse = sparsify_PyTorch.FISTA(I_cuda, self.basis, self.reg, 500)[0].T # hyperparam: num_iter = 500
        X_sparse = X_sparse.cpu().detach().numpy()

        # select a specific factor
        factor_slice = [x[self.factor_idx] for x in X_sparse] #[total_word_count]
        print(factor_slice)
        print(self.words)
        # calculate sententence a_hat as average of tokens a_hat
        pt = 0
        for skip in self.len_ls:
            print(pt, pt+skip)
            sent_ahat = np.mean(factor_slice[pt:pt+skip])
            resp.append(sent_ahat)
            pt += skip
        
        return np.array(resp)

    def _get_top_n_activate_word_context(self, X: List[str], n: int = 3):
        """
        Returns a list of activation info
        """
        my_df = []
        # get embeddings
        embs = self._get_embs(X) #[n_samples, seq_len(varies), 768]
        torch.cuda.empty_cache()

        # sparse code inference
        # I_cuda: [768, total_word_count]
        I_cuda = torch.from_numpy(np.stack(embs, axis=1)).cuda()
        # X_sparse (a_hat): [total_word_count, 1500]
        X_sparse = sparsify_PyTorch.FISTA(I_cuda, self.basis, self.reg, 500)[0].T # hyperparam: num_iter = 500
        X_sparse = X_sparse.cpu().detach().numpy()

        # select a specific factor
        factor_slice = [x[self.factor_idx] for x in X_sparse] #[total_word_count]
        indx = np.argsort(-np.array(factor_slice))[:n]
        for i in indx:
            word = self.words[i]
            act = factor_slice[i]
            sent_position, word_position = self.word_to_sentence[i]
            sentence = self.sentences_str[sent_position]
            d = {'word':word,'index':word_position,'sent_index':sent_position,'score':act,'sent':self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(sentence))}
            my_df.append(d)
        
        return my_df

        


if __name__ == '__main__':
    
    mod = DictionaryModule()
    X = ['that snare shot sounded like somebody would kicked open the door to your mind',
         'i became very frustrated with that and finally made up my mind to start getting back into things',
         'theme park guests may use the hogwarts express to travel between hogsmead']
    df = mod._get_top_n_activate_word_context(X)
    print(df)
    #mod._get_embs(X)
    #resp = mod(X)
    #print(resp)
