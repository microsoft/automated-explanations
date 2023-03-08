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
        NUM_TOP_VOXELS = 500
        self.voxel_idxs = joblib.load(join(SAVE_DIR_FMRI, 'voxel_lists', f'{subject}_voxel_selectivity.jbl'))
        numpy.random.default_rng(seed=42).shuffle(self.voxel_idxs)
        self.voxel_idxs = self.voxel_idxs[:NUM_TOP_VOXELS]
        self.voxel_idx = self.voxel_idxs[voxel_num_best]

        # look at metadata stuff
        corrs = joblib.load(join(SAVE_DIR_FMRI, 'voxel_performances', f'{subject}_voxel_performance.jbl'))
        rois_anat = joblib.load(join(SAVE_DIR_FMRI, 'voxel_rois', 'voxel_anat_rois', f'{subject}_voxel_anat_rois.jbl'))
        rois_func = joblib.load(join(SAVE_DIR_FMRI, 'voxel_rois', 'voxel_func_rois', f'{subject}_voxel_func_rois.jbl'))

        self.corr = corrs[self.voxel_idx]
        # self.roi_anat = rois_anat[str(self.voxel_idx)]
        # self.roi_func = rois_func[str(self.voxel_idx)]]

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
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint, device_map='auto', torch_dtype=torch.float16)
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
        print('embs.shape', embs.shape)

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

def get_test_ngrams(voxel_num_best: int = 0):
    top_ngrams = pd.read_pickle(join(SAVE_DIR_FMRI, 'top_ngrams.pkl'))
    return top_ngrams['voxel_top_' + str(voxel_num_best)].values

def get_roi(voxel_num_best: int = 0):
    rois = pd.read_pickle(join(SAVE_DIR_FMRI, 'roi_dict.pkl'))
    return rois.get(voxel_num_best, '--')

def get_train_story_texts(subject: str='UTS01'):
    # TEST_STORIES = ['wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood']
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

def cache_preprocessor():
    embs_dict = joblib.load(join(SAVE_DIR_FMRI, 'stimulus_features', 'OPT_features.jbl'))
    embs = np.concatenate([embs_dict[k] for k in embs_dict])
    preproc = sklearn.preprocessing.StandardScaler()
    preproc.fit(embs)
    pkl.dump(preproc, open(join(SAVE_DIR_FMRI, 'preproc.pkl'), 'wb'))

if __name__ == '__main__':
    # story_text = get_train_story_texts()
    # cache_preprocessor()
    mod = fMRIModule()
    X = ['I am happy', 'I am sad', 'I am angry']
    print(X[0][:50])
    resp = mod(X[:3])
    print(resp)
    # print(mod.corrs[:20])
