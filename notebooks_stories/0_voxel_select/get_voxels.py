import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
from typing import Dict, List
import numpy as np
import sasc.viz
from pprint import pprint
import joblib
from collections import defaultdict
from sasc.config import RESULTS_DIR, REPO_DIR
from typing import Tuple
import sys
import json

VOXEL_DICT = {
    "UTS01": [
        41,  # time period (numeric),
        378,  # college days
        186,  # location
        322,  # embarrassment
        434,  # laughter or amusement
        244,  # emotional response
        34,  # fear and anticipation
        258,  # lonelineness
        365,  # family and friends
        # 204,  # family and friends
        106,  # communication
        187,  # relationships
        # 399,  # relationships
        # 122,  # relationships
        # 162,  # communication
        # 261,  # relationships
        # 332,  # family and friends
        # 432,  # relationships
        # 142,  # family and friends
        # 340,  # family and friends
        # 71,  # family and friends
        # 291,  # family and relationships
        # 264,  # family and relationships
        # 374,  # family and friends
        # 400,  # relationships
        # 145,  # relationships
        # 479,  # family and relationships
        197,  # flirting and relationships
        # 144,  # family relationships
        489,  # physical movement
        # 469,  # physical contact
        # 171,  # physical movement
        # 46,  # physical contact
        # 394,  # physical movement
        # 456,  # physical contact
        # 121,  # movement
        # 456,  # physical contact
        # 376,  # physical contact
        311,  # vulgarities
        232,  # violence and injury
        203,  # food and drink
        # 321,  # body positioning
        484,  # repetition
    ],
    "UTS02": [
        # 8 voxels with 3 reps is ~15 mins
        337,
        122,
        168,
        171,
        79,
        299,
        368,
        398,  # automatically picked (clustered then ran synthetic stories)
        # 9 handpicked
        426,  # laughter
        155,  # death
        179,  # emotion
        248,  # negativity
        212,  # counting time
        339,  # physical injury or trauma
        154,  # hair and clothing
        442,  # birthdays
        342,  # rejection
    ],
    "UTS03": [
        171,  # time passing
        # 39, # birthdays, birth years
        253,  # measurements (distance and time)
        377,  # distance or proximity
        # 404, # travel and movement
        56,  # location or place
        161,  # surprise, confusion
        47,  # emotional expressions
        # 305,  # self-reflection
        # 110, # laughter
        439,  # profanity
        337,  # love and joy
        494,  # relationships and emotions
        339,  # family and friends
        # 354, # relationships
        343,  # speaking / responding
        # 481, # food and drinks
        # 215, # physical movement
        148,  # physical injury
        # 121, # physical actions
        # 393, # physical movement
        # 331, # physical contact
        # 288, # physical contact
        99,  # body language
        246,  # lying, deception
        400,  # unhealthy
        395,  # vomiting, sickness
        24,  # hair and clothing
    ],
}


INTERACTIONS_DICT = {
    "UTS02": [
        # related (>0.4)
        (212, 171),  # time, measurements
        # medium (~0.2)
        (339, 337),  # physical injury or trauma, moments
        (426, 442),  # laughter, birthdays
        # unrelated (< -0.3)
        (122, 299),  # locations, communication
        (398, 79),  # emotional expression, food preparation
    ],
}

def _voxels_to_rows(
    voxels: List[Tuple], polysemantic_ngrams: Dict = None
) -> pd.DataFrame:
    """Add extra data (like ngrams) to each row"""
    r = pd.read_pickle(join(RESULTS_DIR, "results_fmri_full_1500.pkl")).sort_values(
        by=["top_score_synthetic"], ascending=False
    )
    # put all voxel data into rows DataFrame
    rows = []
    expls = []
    for vox in voxels:
        expl, subj, vox_num = vox
        vox_num = int(vox_num)
        # try:
        row = r[(r.subject == subj) & (r.module_num == vox_num)].iloc[0]
        if polysemantic_ngrams is not None:
            row['top_ngrams_module_correct'] = polysemantic_ngrams[tuple(vox)]
        rows.append(row)
        expls.append(expl)

        # except:
        # print("skipping", vox)
    rows = pd.DataFrame(rows)
    rows["expl"] = expls
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 200):
    # display(rows[['subject', 'module_num', 'expl', 'top_explanation_init_strs', 'top_ngrams_module_correct']])
    return rows


def get_rows_voxels(subject: str, setting="default"):
    """Select rows from fitted voxels

    Params
    ------
    subject : str
        UTS01, UTS02, UTS03
    setting : str
        default, interactions
    """

    # UTS02 - Pilot voxels
    if setting in ["default", "interactions"]:
        VOXEL_DICT_FNAMES = {
            "UTS02": "notebooks_stories/0_voxel_select/pilot/uts02_concepts_pilot_mar22.json",
            "UTS01": "notebooks_stories/0_voxel_select/pilot/uts01_concepts_pilot_jun14.json",
            "UTS03": "notebooks_stories/0_voxel_select/pilot/uts03_concepts_pilot_jun14.json",
        }

    elif setting == "polysemantic":
        VOXEL_DICT_FNAMES = {
            "UTS02": "notebooks_stories/0_voxel_select/polysemantic_UTS02.json",
        }
    voxels_dict = json.load(open(join(REPO_DIR, VOXEL_DICT_FNAMES[subject]), "r"))
    vals = pd.DataFrame([tuple(x) for x in sum(list(voxels_dict.values()), [])])
    vals.columns = ["expl", "subject", "module_num"]

    if setting in ["default", "polysemantic"]:

        # filter vals
        if setting == 'default':
            voxel_nums = VOXEL_DICT[subject]
            print("len(voxel_nums)", len(voxel_nums), "nunique", len(np.unique(voxel_nums)))
            vals = vals[vals["module_num"].isin(voxel_nums)]
            for voxel_num in voxel_nums:
                if not voxel_num in vals["module_num"].values:
                    print("missing", voxel_num)
            assert vals["module_num"].nunique() == vals.shape[0], "no duplicates"
            assert len(vals) == len(voxel_nums), "all voxels found"
            assert len(voxel_nums) == 17
        else:
            print('len(vals)', len(vals), 'nunique voxels', vals['module_num'].nunique())

        # add extra data (like ngrams) to each row
        if setting == "polysemantic":
            polysemantic_ngrams = joblib.load(
                join(
                    REPO_DIR,
                    f"notebooks_stories/0_voxel_select/polysemantic_ngrams_{subject}.pkl",
                )
            )
        else:
            polysemantic_ngrams = None
        rows = _voxels_to_rows(vals.values, polysemantic_ngrams=polysemantic_ngrams)
        return rows

    elif setting == "interactions":
        # interactions
        rows_list = []
        for i in range(2):
            voxel_nums = [x[i] for x in INTERACTIONS_DICT[subject]]
            print(
                "len(voxel_nums)",
                len(voxel_nums),
                "nunique",
                len(np.unique(voxel_nums)),
            )
            v = vals[vals["module_num"].isin(voxel_nums)]
            for voxel_num in voxel_nums:
                if not voxel_num in v["module_num"].values:
                    print("missing", voxel_num)
            assert v["module_num"].nunique() == v.shape[0], "no duplicates"
            assert len(v) == len(voxel_nums), "all voxels found"

            rows_list.append(_voxels_to_rows(v.values))
        return tuple(rows_list)
