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
import openai
from pprint import pprint
import joblib
from collections import defaultdict
from sasc.config import RESULTS_DIR, REPO_DIR
from typing import Tuple
import imodelsx.sasc.llm
import json
# openai.api_key_path = os.path.expanduser('~/.OPENAI_KEY')

def get_rows_voxels(seed, n_voxels_per_category=4):
    '''Select rows from fitted voxels
    '''
    r = (pd.read_pickle(join(RESULTS_DIR, 'results_fmri.pkl'))
        .sort_values(by=['top_score_synthetic'], ascending=False))
    r['id'] = "('" + r['top_explanation_init_strs'].str.replace(' ', '_').str.slice(stop=20) + "', '" + r['subject'] + "', " + r['module_num'].astype(str) + ")"

    def _voxels_to_rows(voxels: List[Tuple]) -> pd.DataFrame:
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
                print('skipping', vox)
        rows = pd.DataFrame(rows)
        rows['expl'] = expls
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 200):
            # display(rows[['subject', 'module_num', 'expl', 'top_explanation_init_strs', 'top_ngrams_module_correct']])
        return rows

    # manually pick some voxels
    # with pd.option_context('display.max_rows', None, 'display.max_colwidth', 200):
    #     display(r.sort_values(by=['top_score_synthetic'], ascending=False)[
    #         ['top_explanation_init_strs', 'subject', 'module_num', 'top_score_synthetic', 'frac_top_ngrams_module_correct', 'id', 'top_ngrams_module_correct']
    #     ].round(3).reset_index(drop=True).head(50))


    # expls = ['baseball','animals','water','movement','religion','time','technology']
    # interesting_expls = ['food', 'numbers', 'physical contact', 'time', 'laughter', 'age', 'clothing']
    # voxels = [('movement', 'UTS01',	7), ('numbers', 'UTS03', 55), ('time', 'UTS03', 19), ('relationships', 'UTS01', 21),
            #   ('sounds', 'UTS03', 35), ('emotion', 'UTS03', 23), ('food', 'UTS03', 46)]
    # voxels = [('numbers', 'UTS03', 55), ('time', 'UTS03', 19),
            #   ('sounds', 'UTS03', 35), ('emotion', 'UTS03', 23), ('food', 'UTS03', 46)]
    # voxels = [('movement', 'UTS01',	7),('relationships', 'UTS01', 21) ('passing of time	UTS02	4)]
    # voxels = [('relationships', 'UTS02', 9), ('time', 'UTS02', 4), ('looking or staring', 'UTS03', 57), ('food and drinks', 'UTS01', 52), ('hands and arms', 'UTS01', 46)]
    # rows = _voxels_to_rows(voxels)
    # return rows

    # mar 21 - voxels spread across categories
    # voxels = [
    #     # belong to previous categories
    #     ('hands and arms', 'UTS01', 46),
    #     ('measurements and numbers', 'UTS02', 48),
    #     ('locations', 'UTS03', 87),
    #     ('time', 'UTS02', 4),
    #     ('physical injury or discomfort', 'UTS01', 35),
    #     ('feelings and emotions', 'UTS02', 104),
    #     ('relationships', 'UTS02', 9),

    #     # new voxels
    #     ('food and drinks', 'UTS01', 52),
    #     ('sound', 'UTS02', 81),
    #     ('hands and arms', 'UTS01', 46),
    # ]
    # rows = _voxels_to_rows(voxels)
    # return rows

    # mar 22 - UTS02 voxels in different categories
    voxels_dict = json.load(open(join(REPO_DIR, f'notebooks_stories/voxel_select/uts02_concepts_pilot_mar22.json'), 'r'))
    d = defaultdict(list)

    # randomly shuffle the categories order + voxels within each category
    # return n_voxels_per_category per category
    # rng = np.random.default_rng(seed)
    # voxels_dict_keys = list(voxels_dict.keys())
    # rng.shuffle(voxels_dict_keys)
    # print(voxels_dict_keys)
    # idxs_list = [rng.choice(len(voxels_dict[k]), n_voxels_per_category, replace=False) for k in voxels_dict_keys]
    # for i, k in enumerate(voxels_dict_keys):
    #     idxs = idxs_list[i]
    #     d['voxels'].extend([tuple(vox) for vox in np.array(voxels_dict[k])[idxs]])
    #     d['category'].extend([k] * n_voxels_per_category)
    # d = pd.DataFrame(d)
    # # print(d.)
    # voxels = d.voxels.values.tolist()
    # rows = _voxels_to_rows(voxels)
    # return rows, idxs_list, voxels

    # mar 24 - UTS02 voxels after screening
    vals = pd.DataFrame([tuple(x) for x in sum(list(voxels_dict.values()), [])])
    vals.columns = ['expl', 'subject', 'module_num']
    voxel_nums = [
        # 8 voxels with 3 reps is ~15 mins
        337, 122, 168, 171, 79, 299, 368, 398, # automatically picked (clustered then ran synthetic stories)
        
        # handpicked
        426, # laughter
        155, # death
        179, # emotion
        248, # negativity
        212, # counting time
        339, # physical injury or trauma
        154, # hair and clothing
        442, # birthdays
        342, # rejection
    ]
    vals = vals[vals['module_num'].isin(voxel_nums)]
    assert vals['module_num'].nunique() == vals.shape[0], 'no duplicates'
    # display(vals.reset_index())
    voxels = vals.sample(frac=1, random_state=seed, replace=False).values

    rows = _voxels_to_rows(voxels)
    return rows
    

def get_rows_huth():
    '''Select rows corresponding to 2016 categories
    '''
    huth2016_categories = json.load(open('huth2016clusters.json', 'r'))
    r = pd.DataFrame.from_dict({'expl': huth2016_categories.keys(), 'top_ngrams_module_correct': huth2016_categories.values()})
    return r


if __name__ == '__main__':
    # version = 'v4'
    # EXPT_NAME = 'huth2016clusters_mar21_i_time_traveled'
    # rows = get_rows_huth()

    # EXPT_NAME = 'relationships_mar9'
    # EXPT_NAME, version = ('voxels_mar21_hands_arms_emergency', 'v4_noun')
    # rows = get_rows_voxels(seed=1)

    # this expt iterates by selecting a fixed # of voxels in each category
    # seed = 10
    # EXPT_NAME, version = (f'uts02_concepts_pilot_mar22_seed={seed}', 'v4_noun')
    # rows, idxs_list, voxels = get_rows_voxels(seed=seed, n_voxels_per_category=4)


    generate_paragraphs = False

    # iterate over seeds
    seeds = list(range(1, 8))
    random.shuffle(seeds)
    for seed in seeds:
        # for version in ['v4_noun', 'v5_noun']:
        for version in ['v4_noun', 'v5_noun']:

            STORIES_DIR = join(RESULTS_DIR, 'pilot_v1')
            EXPT_NAME = f'uts02_pilot_gpt4_mar28___ver={version}___seed={seed}'
            EXPT_DIR = join(STORIES_DIR, EXPT_NAME)
            os.makedirs(EXPT_DIR, exist_ok=True)


            rows = get_rows_voxels(seed=seed)
            rows.to_csv(join(EXPT_DIR, f'rows.csv'), index=False)
            # print(rows.shape)
            # display(rows.head())
            # print(rows)

            expls = rows.expl.values
            examples_list = rows.top_ngrams_module_correct
            prompts = notebook_helper.get_prompts(expls, examples_list, version, n_examples=4)
            for p in prompts:
                print(p)
            PV = notebook_helper.get_prompt_templates(version)

            if generate_paragraphs:
                # generate paragraphs
                paragraphs = imodelsx.sasc.llm.get_paragraphs(
                    prompts,
                    checkpoint='gpt-4-0314',
                    prefix_first=PV['prefix_first'], prefix_next=PV['prefix_next'],
                )
                rows['prompt'] = prompts
                rows['paragraph'] = paragraphs
                for i in tqdm(range(len(paragraphs))):
                    para = paragraphs[i]
                    print(para)
                    # pprint(para)

                # save
                joblib.dump(rows, join(STORIES_DIR, EXPT_NAME, 'rows.pkl'))
                with open(join(EXPT_DIR, 'story.txt'), 'w') as f:
                    f.write('\n\n'.join(rows.paragraph.values))
                with open(join(EXPT_DIR, 'prompts.txt'), 'w') as f:
                    f.write('\n\n'.join(rows.prompt.values))