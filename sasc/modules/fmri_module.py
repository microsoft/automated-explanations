from sasc.config import RESULTS_DIR, SAVE_DIR_FMRI
from collections import defaultdict
import pandas as pd
import logging
from typing import List, Union
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
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import joblib
from copy import deepcopy

modules_dir = dirname(os.path.abspath(__file__))

NUM_TOP_VOXELS = 500

"""
When adding a new model, need to:
- run cache_preprocessor() on the extracted features
"""
VOXELS_IDXS_DICT = {
    subject: joblib.load(
        join(SAVE_DIR_FMRI, "voxel_lists", f"{subject}_voxel_selectivity.jbl")
    )
    for subject in ["UTS01", "UTS02", "UTS03"]
}

STABILITY_SCORES_DICT = joblib.load(
    join(RESULTS_DIR, "sasc", "fmri_stability_scores.jbl"))


def convert_module_num_to_voxel_num(module_num: int, subject: str):
    voxel_idxs = deepcopy(VOXELS_IDXS_DICT[subject])
    numpy.random.default_rng(seed=42).shuffle(voxel_idxs)
    return voxel_idxs[module_num]


def add_stability_score(module_num: int, subject: str):
    return STABILITY_SCORES_DICT[subject][module_num]


class fMRIModule:
    def __init__(
        self,
        voxel_num_best: int = 0,
        subject: str = "UTS01",
        checkpoint="facebook/opt-30b",
        init_model=True,
        restrict_weights=True,
    ):
        """
        Params
        ------
        voxel_num_best: int
            Which voxel to predict (0 for best-predicted voxel, then 1, 2, ...1000)
        """

        # load opt model & tokenizer
        assert checkpoint in ["facebook/opt-30b",
                              "decapoda-research/llama-30b-hf"]
        self.checkpoint = checkpoint
        self.model_dir = {
            "facebook/opt-30b": "opt_model",
            "decapoda-research/llama-30b-hf": "llama_model",
        }[checkpoint]
        if init_model:
            if checkpoint == "decapoda-research/llama-30b-hf":
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    self.checkpoint)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, device_map="auto", torch_dtype=torch.float16
            )

        # load fmri-specific stuff
        self._init_fmri(subject, restrict_weights)
        self._init_fmri_voxel(voxel_num_best, subject)

    def _init_fmri(self, subject: str, restrict_weights: bool = True):
        print("initializing fmri...")
        # select voxel index
        # voxel_idxs = joblib.load(
        #     join(SAVE_DIR_FMRI, "voxel_lists", f"{subject}_voxel_selectivity.jbl")
        # )
        # numpy.random.default_rng(seed=42).shuffle(voxel_idxs)
        # voxel_idxs = voxel_idxs[:NUM_TOP_VOXELS]
        # joblib.dump(
        #     voxel_idxs,
        #     join(
        #         SAVE_DIR_FMRI,
        #         "voxel_lists",
        #         f"{subject}_voxel_selectivity_shuffled.jbl",
        #     ),
        # )
        self.voxel_idxs = joblib.load(
            join(
                SAVE_DIR_FMRI,
                "voxel_lists",
                f"{subject}_voxel_selectivity_shuffled.jbl",
            )
        )

        # load weights
        weights_file = join(
            SAVE_DIR_FMRI, self.model_dir, "model_weights", f"wt_{subject}.jbl"
        )
        self.weights = joblib.load(weights_file)
        if restrict_weights:
            self.weights = self.weights[:, self.voxel_idxs]
        self.preproc = pkl.load(
            open(join(SAVE_DIR_FMRI, self.model_dir, "preproc.pkl"), "rb")
        )
        self.ndel = 4

        # load corrs
        self.corrs = joblib.load(
            join(
                SAVE_DIR_FMRI,
                self.model_dir,
                "voxel_performances",
                f"{subject}_voxel_performance.jbl",
            )
        )
        if self.checkpoint == "decapoda-research/llama-30b-hf":
            self.corrs = self.corrs[0]

    def _init_fmri_voxel(self, voxel_num_best: Union[int, np.ndarray[int]], subject: str):
        if isinstance(voxel_num_best, np.ndarray):
            voxel_num_best = voxel_num_best.astype(int)
        self.voxel_num_best = voxel_num_best
        self.subject = subject

        # load corr performance
        if isinstance(voxel_num_best, int):
            self.corr = self.corrs[self.voxel_idxs[voxel_num_best]]

    def _get_embs(self, X: List[str]):
        """
        Returns
        -------
        embs: np.ndarray
            (n_examples, 7168)
        """
        embs = []
        layer = {
            "facebook/opt-30b": 33,
            "decapoda-research/llama-30b-hf": 18,
        }[self.checkpoint]
        for i in tqdm(range(len(X))):
            text = self.tokenizer.encode(X[i])
            inputs = {}
            inputs["input_ids"] = torch.tensor([text]).int()
            inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape)

            # Ideally, you would use downsampled features instead of copying features across time delays
            emb = (
                list(self.model(**inputs, output_hidden_states=True)
                     [2])[layer][0][-1]
                .cpu()
                .detach()
                .numpy()
            )
            embs.append(emb)
        return np.array(embs)

    def __call__(self, X: List[str] = None, embs=None, return_all=False) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        self.voxel_num_best may be a list, in which case it will return a 2d array (len(X), len(self.voxel_num_best))
        """
        if embs is None:
            # get opt embeddings
            embs = self._get_embs(X)
            torch.cuda.empty_cache()

        # apply StandardScaler (pre-trained)
        embs = self.preproc.transform(embs)

        # apply fMRI transform
        embs_delayed = np.hstack([embs] * self.ndel)
        preds_fMRI = embs_delayed @ self.weights

        if return_all:
            return preds_fMRI  # self.weights was already restricted to top voxels
        else:
            pred_voxel = preds_fMRI[
                :, np.array(self.voxel_num_best).astype(int)
            ]  # select voxel (or potentially many voxels)
            return pred_voxel


def get_roi(voxel_num_best: int = 0, roi_type: str = "anat", subject: str = "UTS01"):
    if roi_type == "anat":
        rois = joblib.load(
            join(
                SAVE_DIR_FMRI,
                "voxel_rois",
                "voxel_anat_rois",
                f"{subject}_voxel_anat_rois.jbl",
            )
        )
    elif roi_type == "func":
        rois = joblib.load(
            join(
                SAVE_DIR_FMRI,
                "voxel_rois",
                "voxel_func_rois",
                f"{subject}_voxel_func_rois.jbl",
            )
        )
    voxel_idxs = joblib.load(
        join(SAVE_DIR_FMRI, "voxel_lists",
             f"{subject}_voxel_selectivity_shuffled.jbl")
    )
    voxel_idx = voxel_idxs[voxel_num_best]
    return rois.get(str(voxel_idx), "--")


def get_train_story_texts(subject: str = "UTS01"):
    TRAIN_STORIES_01 = [
        "itsabox",
        "odetostepfather",
        "inamoment",
        "hangtime",
        "ifthishaircouldtalk",
        "goingthelibertyway",
        "golfclubbing",
        "thetriangleshirtwaistconnection",
        "igrewupinthewestborobaptistchurch",
        "tetris",
        "becomingindian",
        "canplanetearthfeedtenbillionpeoplepart1",
        "thetiniestbouquet",
        "swimmingwithastronauts",
        "lifereimagined",
        "forgettingfear",
        "stumblinginthedark",
        "backsideofthestorm",
        "food",
        "theclosetthatateeverything",
        "notontheusualtour",
        "exorcism",
        "adventuresinsayingyes",
        "thefreedomridersandme",
        "cocoonoflove",
        "waitingtogo",
        "thepostmanalwayscalls",
        "googlingstrangersandkentuckybluegrass",
        "mayorofthefreaks",
        "learninghumanityfromdogs",
        "shoppinginchina",
        "souls",
        "cautioneating",
        "comingofageondeathrow",
        "breakingupintheageofgoogle",
        "gpsformylostidentity",
        "eyespy",
        "treasureisland",
        "thesurprisingthingilearnedsailingsoloaroundtheworld",
        "theadvancedbeginner",
        "goldiethegoldfish",
        "life",
        "thumbsup",
        "seedpotatoesofleningrad",
        "theshower",
        "adollshouse",
        "canplanetearthfeedtenbillionpeoplepart2",
        "sloth",
        "howtodraw",
        "quietfire",
        "metsmagic",
        "penpal",
        "thecurse",
        "canadageeseandddp",
        "thatthingonmyarm",
        "buck",
        "wildwomenanddancingqueens",
        "againstthewind",
        "indianapolis",
        "alternateithicatom",
        "bluehope",
        "kiksuya",
        "afatherscover",
        "haveyoumethimyet",
        "firetestforlove",
        "catfishingstrangerstofindmyself",
        "christmas1940",
        "tildeath",
        "lifeanddeathontheoregontrail",
        "vixenandtheussr",
        "undertheinfluence",
        "beneaththemushroomcloud",
        "jugglingandjesus",
        "superheroesjustforeachother",
        "sweetaspie",
        "naked",
        "singlewomanseekingmanwich",
        "avatar",
        "whenmothersbullyback",
        "myfathershands",
        "reachingoutbetweenthebars",
        "theinterview",
        "stagefright",
        "legacy",
        "canplanetearthfeedtenbillionpeoplepart3",
        "listo",
        "gangstersandcookies",
        "birthofanation",
        "mybackseatviewofagreatromance",
        "lawsthatchokecreativity",
        "threemonths",
        "whyimustspeakoutaboutclimatechange",
        "leavingbaghdad",
    ]
    TRAIN_STORIES_02_03 = [
        "itsabox",
        "odetostepfather",
        "inamoment",
        "afearstrippedbare",
        "findingmyownrescuer",
        "hangtime",
        "ifthishaircouldtalk",
        "goingthelibertyway",
        "golfclubbing",
        "thetriangleshirtwaistconnection",
        "igrewupinthewestborobaptistchurch",
        "tetris",
        "becomingindian",
        "canplanetearthfeedtenbillionpeoplepart1",
        "thetiniestbouquet",
        "swimmingwithastronauts",
        "lifereimagined",
        "forgettingfear",
        "stumblinginthedark",
        "backsideofthestorm",
        "food",
        "theclosetthatateeverything",
        "escapingfromadirediagnosis",
        "notontheusualtour",
        "exorcism",
        "adventuresinsayingyes",
        "thefreedomridersandme",
        "cocoonoflove",
        "waitingtogo",
        "thepostmanalwayscalls",
        "googlingstrangersandkentuckybluegrass",
        "mayorofthefreaks",
        "learninghumanityfromdogs",
        "shoppinginchina",
        "souls",
        "cautioneating",
        "comingofageondeathrow",
        "breakingupintheageofgoogle",
        "gpsformylostidentity",
        "marryamanwholoveshismother",
        "eyespy",
        "treasureisland",
        "thesurprisingthingilearnedsailingsoloaroundtheworld",
        "theadvancedbeginner",
        "goldiethegoldfish",
        "life",
        "thumbsup",
        "seedpotatoesofleningrad",
        "theshower",
        "adollshouse",
        "canplanetearthfeedtenbillionpeoplepart2",
        "sloth",
        "howtodraw",
        "quietfire",
        "metsmagic",
        "penpal",
        "thecurse",
        "canadageeseandddp",
        "thatthingonmyarm",
        "buck",
        "thesecrettomarriage",
        "wildwomenanddancingqueens",
        "againstthewind",
        "indianapolis",
        "alternateithicatom",
        "bluehope",
        "kiksuya",
        "afatherscover",
        "haveyoumethimyet",
        "firetestforlove",
        "catfishingstrangerstofindmyself",
        "christmas1940",
        "tildeath",
        "lifeanddeathontheoregontrail",
        "vixenandtheussr",
        "undertheinfluence",
        "beneaththemushroomcloud",
        "jugglingandjesus",
        "superheroesjustforeachother",
        "sweetaspie",
        "naked",
        "singlewomanseekingmanwich",
        "avatar",
        "whenmothersbullyback",
        "myfathershands",
        "reachingoutbetweenthebars",
        "theinterview",
        "stagefright",
        "legacy",
        "canplanetearthfeedtenbillionpeoplepart3",
        "listo",
        "gangstersandcookies",
        "birthofanation",
        "mybackseatviewofagreatromance",
        "lawsthatchokecreativity",
        "threemonths",
        "whyimustspeakoutaboutclimatechange",
        "leavingbaghdad",
    ]
    story_names_train = {
        "UTS01": TRAIN_STORIES_01,
        "UTS02": TRAIN_STORIES_02_03,
        "UTS03": TRAIN_STORIES_02_03,
    }[subject]

    grids = joblib.load(join(SAVE_DIR_FMRI, "stories", "grids_all.jbl"))
    trfiles = joblib.load(join(SAVE_DIR_FMRI, "stories", "trfiles_all.jbl"))
    from huth.utils_ds import make_word_ds

    wordseqs = make_word_ds(grids, trfiles)
    texts = [" ".join(wordseqs[story].data) for story in story_names_train]
    return texts


def cache_test_data():
    """Format the test data as a supervised task (text, resp) using 4 delays"""
    TEST_STORIES = ["wheretheressmoke",
                    "onapproachtopluto", "fromboyhoodtofatherhood"]
    out = defaultdict(dict)
    for subject in tqdm(["UTS01", "UTS02", "UTS03"]):
        voxel_idxs = joblib.load(
            join(SAVE_DIR_FMRI, "voxel_lists",
                 f"{subject}_voxel_selectivity.jbl")
        )[:NUM_TOP_VOXELS]
        grids = joblib.load(join(SAVE_DIR_FMRI, "stories", "grids_all.jbl"))
        trfiles = joblib.load(
            join(SAVE_DIR_FMRI, "stories", "trfiles_all.jbl"))
        from huth.utils_ds import make_word_ds

        wordseqs = make_word_ds(grids, trfiles)

        # loop over stories
        running_words = {}
        for k in TEST_STORIES:
            # get words from last 4 TRs
            # given TR with time t, words between [t-8 sec, t-2 sec]
            wordseq = wordseqs[k]
            words = np.array(wordseq.data)
            tr_times = wordseq.tr_times[10:-5]
            num_trs = 4
            running_words[k] = []
            for i in range(len(tr_times)):
                tr_time_max = tr_times[max(0, i - 1)]
                tr_time_min = tr_times[max(0, i - num_trs)]
                valid_times = (tr_time_min <= wordseq.data_times) & (
                    wordseq.data_times <= tr_time_max
                )
                running_words[k].append(" ".join(words[valid_times]))

        # get resp
        # these are already normalized
        resp = joblib.load(join(SAVE_DIR_FMRI, "responses",
                           f"{subject}_responses.jbl"))
        resp = {
            k: resp[k][:, voxel_idxs] for k in TEST_STORIES
        }  # narrow down the stories/voxels

        out[subject]["words"] = sum(list(running_words.values()), [])
        out[subject]["resp"] = np.concatenate(list(resp.values()))
        assert len(out[subject]["words"]) == out[subject]["resp"].shape[0]
    joblib.dump(out, join(SAVE_DIR_FMRI, "stories", "running_words.jbl"))


def cache_preprocessor(model_dir="llama_model"):
    embs_dict = joblib.load(
        join(SAVE_DIR_FMRI, model_dir, "stimulus_features.jbl"))
    embs = np.concatenate([embs_dict[k] for k in embs_dict])
    preproc = sklearn.preprocessing.StandardScaler()
    preproc.fit(embs)
    pkl.dump(preproc, open(join(SAVE_DIR_FMRI, model_dir, "preproc.pkl"), "wb"))


if __name__ == "__main__":
    # cache_preprocessor()
    cache_test_data()

    # story_texts = get_train_story_texts()
    # story_text = ' '.join(story_texts)
    # print(len(set(story_text.split())))
    # with open('story_text.txt', 'w') as f:
    #     f.write(story_text)

    # mod = fMRIModule(
    # voxel_num_best=[1, 2, 3], checkpoint="decapoda-research/llama-30b-hf"
    # )
    # X = ["I am happy", "I am sad", "I am angry"]
    # print(X[0][:50])
    # resp = mod(X[:3])
    # print(resp.shape)
    # print(resp)

    for subj in ["UTS01", "UTS02", "UTS03"]:
        print(
            joblib.load(
                f"/home/chansingh/mntv1/deep-fMRI/rj_models/llama_model/voxel_performances/{subj}_voxel_performance.jbl"
            )[0].mean()
        )
