from collections import defaultdict
import pandas as pd
import logging
from typing import List
import datasets
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import sklearn.preprocessing
from spacy.lang.en import English
import imodelsx
import imodelsx.util
import pickle as pkl
from os.path import dirname, join
import torch.cuda
import os.path
import torch
import numpy.random
from transformers import AutoModelForCausalLM, AutoTokenizer
import joblib
from huth.utils_ds import make_word_ds

modules_dir = dirname(os.path.abspath(__file__))
SAVE_DIR_FMRI = join(modules_dir, 'fmri')
NUM_TOP_VOXELS = 500

class fMRIModule():

    def __init__(self, voxel_num_best: int = 0, subject: str = 'UTS01'):
        """
        Params
        ------
        voxel_num_best: int
            Which voxel to predict (0 for best-predicted voxel, then 1, 2, ...1000)
        """

        # hyperparams for loaded model
        self.checkpoint = 'facebook/opt-30b'
        self.voxel_num_best = voxel_num_best
        self.subject = subject
        self.ndel = 4
    
        # select voxel index
        self.voxel_idxs = joblib.load(join(SAVE_DIR_FMRI, 'voxel_lists', f'{subject}_voxel_selectivity.jbl'))
        numpy.random.default_rng(seed=42).shuffle(self.voxel_idxs)
        self.voxel_idxs = self.voxel_idxs[:NUM_TOP_VOXELS]
        joblib.dump(self.voxel_idxs, join(SAVE_DIR_FMRI, 'voxel_lists', f'{subject}_voxel_selectivity_shuffled.jbl'))
        self.voxel_idx = self.voxel_idxs[voxel_num_best]

        # look at metadata stuff
        corrs = joblib.load(join(SAVE_DIR_FMRI, 'voxel_performances', f'{subject}_voxel_performance.jbl'))
        self.corr = corrs[self.voxel_idx]

        # load weights
        weights_file = join(SAVE_DIR_FMRI, 'model_weights', f'wt_{subject}.jbl')
        self.weights = joblib.load(weights_file)
        self.preproc = pkl.load(open(join(SAVE_DIR_FMRI, 'preproc.pkl'), 'rb'))
        self.weights = self.weights[:, self.voxel_idxs]
    

    def _get_embs(self, X: List[str]):
        """
        Returns
        -------
        embs: np.ndarray
            (n_examples, 7168)
        """
        if torch.cuda.device_count() > 0:
            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, device_map='auto', torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        embs = []
        for i in tqdm(range(len(X))):
            # have to do this or get some weird opt error
            text = tokenizer.encode(X[i])
            inputs = {}
            inputs['input_ids'] = torch.tensor([text]).int()
            inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape)
    
            # Ideally, you would use downsampled features instead of copying features across time delays
            emb = list(model(**inputs, output_hidden_states=True)[2])[33][0][-1].cpu().detach().numpy()
            embs.append(emb)
        return np.array(embs)

    def __call__(self, X: List[str], return_all=False) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        # get opt embeddings
        embs = self._get_embs(X)
        torch.cuda.empty_cache()

        # apply StandardScaler (pre-trained)
        embs = self.preproc.transform(embs)

        # apply fMRI transform
        embs_delayed = np.hstack([embs] * self.ndel)
        preds_fMRI = embs_delayed @ self.weights
        
        if return_all:
            # self.weights was already restricted to top voxels
            return preds_fMRI
            
        # select voxel
        else:
            pred_voxel = preds_fMRI[:, self.voxel_num_best]
            return pred_voxel

def get_roi(voxel_num_best: int = 0, roi_type: str = 'anat', subject: str = 'UTS01'):
    if roi_type == 'anat':
        rois = joblib.load(join(SAVE_DIR_FMRI, 'voxel_rois', 'voxel_anat_rois', f'{subject}_voxel_anat_rois.jbl'))
    elif roi_type == 'func':
        rois = joblib.load(join(SAVE_DIR_FMRI, 'voxel_rois', 'voxel_func_rois', f'{subject}_voxel_func_rois.jbl'))
    voxel_idxs = joblib.load(join(SAVE_DIR_FMRI, 'voxel_lists', f'{subject}_voxel_selectivity_shuffled.jbl'))
    voxel_idx = voxel_idxs[voxel_num_best]
    return rois.get(str(voxel_idx), '--')

def get_train_story_texts(subject: str='UTS01'):
    TRAIN_STORIES_01 = ['itsabox', 'odetostepfather', 'inamoment',  'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
    TRAIN_STORIES_02_03 = ['itsabox', 'odetostepfather', 'inamoment', 'afearstrippedbare', 'findingmyownrescuer', 'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'escapingfromadirediagnosis', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'marryamanwholoveshismother', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'thesecrettomarriage', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
    story_names_train = {
        'UTS01': TRAIN_STORIES_01,
        'UTS02': TRAIN_STORIES_02_03,
        'UTS03': TRAIN_STORIES_02_03,
    }[subject]

    grids = joblib.load(join(SAVE_DIR_FMRI, 'stories', 'grids_all.jbl'))
    trfiles = joblib.load(join(SAVE_DIR_FMRI, 'stories', 'trfiles_all.jbl'))
    wordseqs = make_word_ds(grids, trfiles)
    texts = [' '.join(wordseqs[story].data) for story in story_names_train]
    return texts

def cache_test_data():
    '''Format the test data as a supervised task (text, resp) using 4 delays
    '''
    TEST_STORIES = ['wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood']
    out = defaultdict(dict)
    for subject in tqdm(['UTS01', 'UTS02', 'UTS03']):
        voxel_idxs = joblib.load(join(SAVE_DIR_FMRI, 'voxel_lists', f'{subject}_voxel_selectivity.jbl'))[:NUM_TOP_VOXELS]
        grids = joblib.load(join(SAVE_DIR_FMRI, 'stories', 'grids_all.jbl'))
        trfiles = joblib.load(join(SAVE_DIR_FMRI, 'stories', 'trfiles_all.jbl'))
        wordseqs = make_word_ds(grids, trfiles)    

        # loop over stories
        running_words = {}
        for k in TEST_STORIES:
            # get words from last 4 TRs
            # given TR with time t, words between [t-8 sec, t-2 sec]
            wordseq = wordseqs[k]
            words = np.array(wordseq.data)
            tr_times = wordseq.tr_times[10:-5]
            num_delays = 4
            running_words[k] = []
            for i in range(len(tr_times)):
                tr_time_max = tr_times[max(0, i - 1)]
                tr_time_min = tr_times[max(0, i - num_delays)]
                valid_times = (tr_time_min <= wordseq.data_times) & (wordseq.data_times <= tr_time_max)
                running_words[k].append(' '.join(words[valid_times]))

        # get resp
        resp = joblib.load(join(SAVE_DIR_FMRI, 'responses', f'{subject}_responses.jbl')) # these are already normalized
        resp = {k: resp[k][:, voxel_idxs] for k in TEST_STORIES} # narrow down the stories/voxels

        out[subject]['words'] = sum(list(running_words.values()), [])
        out[subject]['resp'] = np.concatenate(list(resp.values()))
        assert len(out[subject]['words']) == out[subject]['resp'].shape[0]
    joblib.dump(out, join(SAVE_DIR_FMRI, 'stories', 'running_words.jbl'))

def cache_preprocessor():
    embs_dict = joblib.load(join(SAVE_DIR_FMRI, 'stimulus_features', 'OPT_features.jbl'))
    embs = np.concatenate([embs_dict[k] for k in embs_dict])
    preproc = sklearn.preprocessing.StandardScaler()
    preproc.fit(embs)
    pkl.dump(preproc, open(join(SAVE_DIR_FMRI, 'preproc.pkl'), 'wb'))

if __name__ == '__main__':
    # cache_preprocessor()
    # cache_test_data()
    # story_text = get_train_story_texts()
    
    mod = fMRIModule()
    # X = ['I am happy', 'I am sad', 'I am angry']
    # print(X[0][:50])
    # resp = mod(X[:3])
    # print(resp)
    # print(mod.corrs[:20])
