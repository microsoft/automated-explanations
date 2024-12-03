import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
from typing import List
import numpy as np
import sasc.generate_helper
import sasc.viz
from pprint import pprint
import joblib
from collections import defaultdict
from sasc.config import RESULTS_DIR, REPO_DIR
from typing import Tuple
import sys
import json
sys.path.append('../0_voxel_select')

sys.path.append(join(REPO_DIR, "notebooks_stories", "0_voxel_select"))


def get_rows_and_prompts_default(
    subject,
    setting,
    seed,
    n_examples_per_prompt_to_consider,
    n_examples_per_prompt,
    version,
    fname_suffix,
):
    # get voxels
    rows = get_voxels.get_rows_voxels(
        subject=subject, setting=setting, fname_suffix=fname_suffix)

    # shuffle order (this is the 1st place randomness is applied)
    rows = rows.sample(frac=1, random_state=seed, replace=False)

    # get prompt inputs
    expls = rows.expl.values
    examples_list = rows.top_ngrams_module_correct
    # n_examples from each list of examples (this is the 2nd and last place randomness is applied)
    # for pilot v0, just selected the first
    examples_list = sasc.generate_helper.select_top_examples_randomly(
        examples_list,
        n_examples_per_prompt_to_consider,
        n_examples_per_prompt,
        seed,
    )

    # get prompts
    PV = sasc.generate_helper.get_prompt_templates(version)
    prompts = sasc.generate_helper.get_prompts(expls, examples_list, version)
    if 'prompt_suffix' in rows.columns:
        prompts = [p + row.prompt_suffix for p,
                   row in zip(prompts, rows.itertuples())]
    for p in prompts:
        print(p)

    return rows, prompts, PV


def get_rows_and_prompts_interactions(
    subject,
    setting,
    seed,
    n_examples_per_prompt_to_consider,
    n_examples_per_prompt,
    version,
):
    # get voxels
    rows1, rows2 = get_voxels.get_rows_voxels(subject=subject, setting=setting)
    print(rows1.expl.values)
    print(rows2.expl.values)

    # shuffle order (this is the 1st place randomness is applied)
    rows1 = rows1.sample(frac=1, random_state=seed, replace=False)
    rows2 = rows2.sample(frac=1, random_state=seed, replace=False)

    # get prompt inputs
    expls1 = rows1.expl.values
    expls2 = rows2.expl.values
    kwargs = dict(
        n_examples_per_prompt_to_consider=n_examples_per_prompt_to_consider,
        n_examples_per_prompt=n_examples_per_prompt,
        seed=seed,
    )

    examples_list1 = sasc.generate_helper.select_top_examples_randomly(
        rows1["top_ngrams_module_correct"].values.tolist(), **kwargs
    )
    examples_list2 = sasc.generate_helper.select_top_examples_randomly(
        rows2["top_ngrams_module_correct"].values.tolist(), **kwargs
    )
    prompts = sasc.generate_helper.get_prompts_interaction(
        expls1,
        expls2,
        examples_list1,
        examples_list2,
        version=version,
    )
    for p in prompts:
        print(p)
    PV = sasc.generate_helper.get_prompt_templates_interaction(version)

    return rows1, rows2, prompts, PV


if __name__ == "__main__":
    import get_voxels
    generate_paragraphs = False

    VERSIONS = {
        # "default": "v5_noun",
        "default": "v4_noun",
        "interactions": "v0",
        "polysemantic": "v5_noun",
        'qa': 'v6_noun',
        'roi': 'v6_noun',
    }
    # iterate over seeds
    seeds = range(1, 7)
    # seeds = range(7, 12)
    # seeds = range(1, 10)
    # seeds = range(2, 3)
    # random.shuffle(seeds)

    # original stories
    # n_examples_per_prompt = 3
    # n_examples_per_prompt_to_consider = 6

    # increased for roi stories
    n_examples_per_prompt = 5
    n_examples_per_prompt_to_consider = 9
    fname_suffix = '_v2'
    for setting in [
        # "interactions",
        # "default",
        # "polysemantic",
        # 'qa',
        'roi',
    ]:  # default, interactions, polysemantic
        for subject in [
            # "UTS02",
            "UTS03",
        ]:  # , "UTS03"]:  # ["UTS01", "UTS03"]:
            for seed in seeds:
                # for version in ["v5_noun"]:
                version = VERSIONS[setting]
                STORIES_DIR = join(RESULTS_DIR, "stories")

                # EXPT_NAME = f"{subject.lower()}___qa_may31___seed={seed}"
                # EXPT_NAME = f"{subject.lower()}___roi_may31___seed={seed}"
                EXPT_NAME = f"{subject.lower()}___roi_nov30___seed={seed}{fname_suffix}"
                EXPT_DIR = join(STORIES_DIR, setting, EXPT_NAME)
                os.makedirs(EXPT_DIR, exist_ok=True)

                if setting in ["default", "polysemantic", 'qa', 'roi',]:
                    rows, prompts, PV = get_rows_and_prompts_default(
                        subject,
                        setting,
                        seed,
                        n_examples_per_prompt_to_consider,
                        n_examples_per_prompt,
                        version,
                        fname_suffix,
                    )
                    rows.to_csv(join(EXPT_DIR, f"rows.csv"), index=False)
                    rows.to_pickle(join(EXPT_DIR, f"rows.pkl"))
                    print(prompts)
                    # exit(0)

                elif setting == "interactions":
                    rows1, rows2, prompts, PV = get_rows_and_prompts_interactions(
                        subject,
                        setting,
                        seed,
                        n_examples_per_prompt_to_consider,
                        n_examples_per_prompt,
                        version,
                    )

                    # repeat
                    reps = [rows1.iloc[0]]
                    for i in range(0, len(rows1)):
                        reps.append(rows1.iloc[i])
                        reps.append(rows2.iloc[i])
                        reps.append(rows1.iloc[i])
                    rows1_rep = pd.concat(
                        reps,
                        ignore_index=True,
                        axis=1,
                    ).transpose()
                    rows1.to_csv(join(EXPT_DIR, f"rows1.csv"), index=False)
                    rows1.to_pickle(join(EXPT_DIR, f"rows1.pkl"))
                    rows2.to_csv(join(EXPT_DIR, f"rows2.csv"), index=False)
                    rows2.to_pickle(join(EXPT_DIR, f"rows2.pkl"))
                    rows1_rep.to_pickle(join(EXPT_DIR, f"rows.pkl"))

                for p in prompts:
                    print('\n' + p)
                with open(join(EXPT_DIR, "prompts.txt"), "w") as f:
                    f.write("\n\n".join(prompts))

                # save
                continue
                # generate paragraphs
                paragraphs = sasc.generate_helper.get_paragraphs(
                    prompts,
                    checkpoint="gpt-4",
                    prefix_first=PV["prefix_first"] if "prefix_first" in PV else None,
                    prefix_next=PV["prefix_next"] if "prefix_next" in PV else None,
                    cache_dir="/home/chansingh/cache/llm_stories_may8",
                )

                with open(join(EXPT_DIR, "story.txt"), "w") as f:
                    f.write("\n\n".join(paragraphs))
                joblib.dump(
                    {"prompts": prompts, "paragraphs": paragraphs},
                    join(EXPT_DIR, "prompts_paragraphs.pkl"),
                )
