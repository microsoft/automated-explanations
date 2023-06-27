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

sys.path.append(join(REPO_DIR, "notebooks_stories", "0_voxel_select"))
import pilot


def get_rows_and_prompts_default(
    subject,
    setting,
    seed,
    n_examples_per_prompt_to_consider,
    n_examples_per_prompt,
    version,
):
    # get voxels
    rows = pilot.get_rows_voxels(subject=subject, setting=setting)

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
    prompts = sasc.generate_helper.get_prompts(expls, examples_list, version)
    for p in prompts:
        print(p)
    PV = sasc.generate_helper.get_prompt_templates(version)

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
    rows1, rows2 = pilot.get_rows_voxels(subject=subject, setting=setting)

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
    examples_list1 = sasc.generate_helper.select_top_examples_randomly(rows1['top_ngrams_module_correct'], **kwargs)
    examples_list2 = sasc.generate_helper.select_top_examples_randomly(rows2['top_ngrams_module_correct'], **kwargs)
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
    generate_paragraphs = False

    VERSIONS = {
        "default": "v5_noun",
        "interactions": "v0",
    }
    # iterate over seeds
    seeds = list(range(1, 8))
    # random.shuffle(seeds)
    n_examples_per_prompt = 3
    n_examples_per_prompt_to_consider = 5
    for setting in ["interactions"]:  # default
        for subject in ["UTS02"]:  # ["UTS01", "UTS03"]:
            for seed in seeds:
                # for version in ["v5_noun"]:
                version = VERSIONS[setting]
                STORIES_DIR = join(RESULTS_DIR, "pilot_v1")

                EXPT_NAME = f"{subject.lower()}___jun14___seed={seed}"
                EXPT_DIR = join(STORIES_DIR, EXPT_NAME, setting)
                os.makedirs(EXPT_DIR, exist_ok=True)

                if setting == "default":
                    rows, prompts, PV = get_rows_and_prompts_default(
                        subject,
                        setting,
                        seed,
                        n_examples_per_prompt_to_consider,
                        n_examples_per_prompt,
                        version,
                    )
                    rows.to_csv(join(EXPT_DIR, f"rows.csv"), index=False)
                    with open(join(EXPT_DIR, "prompts.txt"), "w") as f:
                        f.write("\n\n".join(prompts))
                elif setting == "interactions":
                    rows1, rows2, prompts, PV = get_rows_and_prompts_interactions(
                        subject,
                        setting,
                        seed,
                        n_examples_per_prompt_to_consider,
                        n_examples_per_prompt,
                        version,
                    )
                    rows1.to_csv(join(EXPT_DIR, f"rows1.csv"), index=False)
                    rows2.to_csv(join(EXPT_DIR, f"rows2.csv"), index=False)
                    with open(join(EXPT_DIR, "prompts.txt"), "w") as f:
                        f.write("\n\n".join(prompts))

                # generate paragraphs
                paragraphs = sasc.generate_helper.get_paragraphs(
                    prompts,
                    checkpoint="gpt-4-0314",
                    prefix_first=PV["prefix_first"] if 'prefix_first' in PV else None,
                    prefix_next=PV["prefix_next"] if 'prefix_next' in PV else None,
                    cache_dir="/home/chansingh/cache/llm_stories",
                )

                for i in tqdm(range(len(paragraphs))):
                    para = paragraphs[i]
                    print(para)
                    # pprint(para)

                # save
                # rows["prompt"] = prompts
                # rows["paragraph"] = paragraphs
                # joblib.dump(rows, join(EXPT_DIR, "rows.pkl"))
                with open(join(EXPT_DIR, "story.txt"), "w") as f:
                    f.write("\n\n".join(paragraphs))
                joblib.dump(
                    {"prompts": prompts, "paragraphs": paragraphs},
                    join(EXPT_DIR, "prompts_paragraphs.pkl"),
                )
