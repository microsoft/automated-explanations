import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
from typing import List
import numpy as np
import sasc.notebook_helper as notebook_helper
import sasc.viz
from pprint import pprint
import joblib
from collections import defaultdict
from sasc.config import RESULTS_DIR, REPO_DIR
from typing import Tuple
import imodelsx.sasc.llm
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


def _voxels_to_rows(voxels: List[Tuple]) -> pd.DataFrame:
    r = pd.read_pickle(join(RESULTS_DIR, "results_fmri_full_1500.pkl")).sort_values(
        by=["top_score_synthetic"], ascending=False
    )
    # put all voxel data into rows DataFrame
    rows = []
    expls = []
    for vox in voxels:
        expl, subj, vox_num = vox
        vox_num = int(vox_num)
        try:
            rows.append(r[(r.subject == subj) & (r.module_num == vox_num)].iloc[0])
            expls.append(expl)
        except:
            print("skipping", vox)
    rows = pd.DataFrame(rows)
    rows["expl"] = expls
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 200):
    # display(rows[['subject', 'module_num', 'expl', 'top_explanation_init_strs', 'top_ngrams_module_correct']])
    return rows


def get_rows_voxels(subject: str):
    """Select rows from fitted voxels"""

    # UTS02 - Pilot voxels
    PILOT_FNAMES = {
        "UTS02": "notebooks_stories/voxel_select/uts02_concepts_pilot_mar22.json",
        "UTS01": "notebooks_stories/voxel_select/uts01_concepts_pilot_jun14.json",
        "UTS03": "notebooks_stories/voxel_select/uts03_concepts_pilot_jun14.json",
    }
    voxels_dict = json.load(open(join(REPO_DIR, PILOT_FNAMES[subject]), "r"))
    vals = pd.DataFrame([tuple(x) for x in sum(list(voxels_dict.values()), [])])
    vals.columns = ["expl", "subject", "module_num"]
    voxel_nums = VOXEL_DICT[subject]
    print("len(voxel_nums)", len(voxel_nums), "nunique", len(np.unique(voxel_nums)))
    vals = vals[vals["module_num"].isin(voxel_nums)]
    for voxel_num in voxel_nums:
        if not voxel_num in vals["module_num"].values:
            print("missing", voxel_num)
    assert vals["module_num"].nunique() == vals.shape[0], "no duplicates"
    assert len(vals) == len(voxel_nums), "all voxels found"
    assert len(voxel_nums) == 17

    # add extra data (like ngrams) to voxels
    rows = _voxels_to_rows(vals.values)
    return rows


if __name__ == "__main__":
    generate_paragraphs = False

    # iterate over seeds
    seeds = list(range(1, 8))
    # random.shuffle(seeds)
    for subject in ['UTS01', 'UTS03']:
        for seed in seeds:
            for version in ["v5_noun"]:
                STORIES_DIR = join(RESULTS_DIR, "pilot_v1")

                EXPT_NAME = f"{subject.lower()}___jun14___seed={seed}"
                EXPT_DIR = join(STORIES_DIR, EXPT_NAME)
                os.makedirs(EXPT_DIR, exist_ok=True)

                # get voxels
                rows = get_rows_voxels(subject=subject)

                # shuffle order (this is the only place randomness is applied)
                rows = rows.sample(frac=1, random_state=seed, replace=False)
                rows.to_csv(join(EXPT_DIR, f"rows.csv"), index=False)

                # get prompts
                expls = rows.expl.values
                examples_list = rows.top_ngrams_module_correct
                prompts = notebook_helper.get_prompts(
                    expls, examples_list, version, n_examples=4
                )
                for p in prompts:
                    print(p)
                PV = notebook_helper.get_prompt_templates(version)
                with open(join(EXPT_DIR, "prompts.txt"), "w") as f:
                    f.write("\n\n".join(prompts))

                # generate paragraphs
                paragraphs = imodelsx.sasc.llm.get_paragraphs(
                    prompts,
                    checkpoint="gpt-4-0314",
                    prefix_first=PV["prefix_first"],
                    prefix_next=PV["prefix_next"],
                    cache_dir="/home/chansingh/cache/llm_stories",
                )

                for i in tqdm(range(len(paragraphs))):
                    para = paragraphs[i]
                    print(para)
                    # pprint(para)

                # save
                rows["prompt"] = prompts
                rows["paragraph"] = paragraphs
                joblib.dump(rows, join(EXPT_DIR, "rows.pkl"))
                with open(join(EXPT_DIR, "story.txt"), "w") as f:
                    f.write("\n\n".join(paragraphs))
