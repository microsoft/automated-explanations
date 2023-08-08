"""Process the explanations of each fMRI voxel
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import imodelsx.process_results
import sys
import bert_score
import numpy as np
from typing import List

tqdm.pandas()
from copy import deepcopy
from sasc.modules.fmri_module import get_roi
import sasc.evaluate as evaluate
from sasc.config import RESULTS_DIR
import joblib
from sasc.modules.fmri_module import SAVE_DIR_FMRI
import imodelsx.util
from sasc.modules.emb_diff_module import EmbDiffModule
import scipy.stats
from sasc.evaluate import D5_Validator
import torch.cuda
from sasc.config import CACHE_DIR
from sasc.modules.fmri_module import convert_module_num_to_voxel_num


def add_expl_preds(r):
    """Calculate match between expl and test resp"""
    dsets = joblib.load(join(SAVE_DIR_FMRI, "stories", "running_words.jbl"))
    r["neg_dists_expl_test"] = [[] for _ in range(r.shape[0])]
    r["rankcorr_expl_test"] = np.nan

    mod = EmbDiffModule(task_str="")
    for i in tqdm(range(r.shape[0])):
        row = r.iloc[i]
        expl = row["top_explanation_init_strs"]
        subject = row["subject"]

        # get resp
        dset = dsets[subject]
        resp = dset["resp"][:, row["module_num"]]

        # check cache
        cache_fname = join(
            CACHE_DIR,
            "expl_preds",
            f'{subject}_{expl.replace("/", "__").replace(" ", "_")[:150]}.jbl',
        )
        os.makedirs(join(CACHE_DIR, "expl_preds"), exist_ok=True)
        loaded = False
        if os.path.exists(cache_fname):
            try:
                neg_dists = joblib.load(cache_fname)
                loaded = True
            except:
                pass

        if not loaded:
            mod._init_task(task_str=expl)
            strs_list = dset["words"]
            neg_dists = [
                mod(
                    imodelsx.util.generate_ngrams_list(
                        strs_list[j], ngrams=3, all_ngrams=True
                    ),
                    verbose=False,
                ).mean()
                for j in range(len(strs_list))
            ]
            joblib.dump(neg_dists, cache_fname)

        # neg dist closer to 0 should elicit higher response
        r["neg_dists_expl_test"].iloc[i] = neg_dists
        r["rankcorr_expl_test"].iloc[i] = scipy.stats.spearmanr(
            neg_dists, resp, nan_policy="omit", alternative="greater"
        ).statistic
    return r

    # mod = None
    # torch.cuda.empty_cache()
    # val = D5_Validator()
    # frac_valid = [
    #     np.mean(
    #         val.validate_w_scores(
    #             expl,
    #             imodelsx.util.generate_ngrams_list(strs_list[j], ngrams=3, all_ngrams=False)))
    #     for j in tqdm(range(len(strs_list)))
    # ]


if __name__ == "__main__":
    # results_dir = "/home/chansingh/mntv1/mprompt/mar13/"
    # suffix = '_opt'
    results_dir = "/home/chansingh/mntv1/mprompt/aug1_llama/"
    suffix = "_llama"

    print("Loading results...")
    r = imodelsx.process_results.get_results_df(results_dir, use_cached=False)
    print(f"Loaded {r.shape[0]} results")
    for num in [25, 50, 75, 100]:
        r[f"top_ngrams_module_{num}"] = r["explanation_init_ngrams"].apply(
            lambda x: x[:num]
        )
        # r[f'top_ngrams_test_{num}'] = r.apply(lambda row: get_test_ngrams(voxel_num_best=row.module_num)[:num], axis=1)

    print(f"Adding roi...")
    r["roi_anat"] = r.progress_apply(
        lambda row: get_roi(
            voxel_num_best=row.module_num, roi_type="anat", subject=row.subject
        ),
        axis=1,
    )
    r["roi_func"] = r.progress_apply(
        lambda row: get_roi(
            voxel_num_best=row.module_num, roi_type="func", subject=row.subject
        ),
        axis=1,
    )

    # Calculate train ngram correctness
    print(f"Finding matching ngrams_module...")
    num_top_ngrams_expl = 75
    (
        correct_ngrams_module_scores,
        correct_ngrams_module_list,
    ) = evaluate.calc_frac_correct_score(
        r, col_ngrams=f"top_ngrams_module_{num_top_ngrams_expl}"
    )
    r["top_ngrams_module_correct"] = correct_ngrams_module_list
    r["frac_top_ngrams_module_correct"] = r["top_ngrams_module_correct"].apply(
        lambda x: len(x) / num_top_ngrams_expl
    )

    # Add score normalized by std over ngrams
    scores_std = {}
    for subject in ["UTS01", "UTS02", "UTS03"]:
        suff = "_llama" if suffix == "_llama" else ""
        ngram_scores_filename = join(
            CACHE_DIR, "cache_ngrams", f"fmri_{subject}{suffix}.pkl"
        )
        ngram_scores = joblib.load(ngram_scores_filename)
        ngram_scores_std = np.std(ngram_scores, axis=0)
        scores_std[subject] = deepcopy(ngram_scores_std)

    r["top_score_std"] = r.apply(
        lambda x: scores_std[x["subject"]][x["module_num"]], axis=1
    )
    r["top_score_normalized"] = r["top_score_synthetic"] / r["top_score_std"]
    r["voxel_num"] = r.apply(
        lambda row: convert_module_num_to_voxel_num(row["module_num"], row["subject"]),
        axis=1,
    )

    # Save results
    r.to_pickle(join(RESULTS_DIR, f"results_fmri_1500{suffix}.pkl"))

    ############# Secondary metric ########################
    # Add explanation<>test response match
    r = pd.read_pickle(join(RESULTS_DIR, f"results_fmri_1500{suffix}.pkl"))
    torch.cuda.empty_cache()
    print("Saved original results, now computing expl<>resp match...")
    r = add_expl_preds(r)
    r.to_pickle(join(RESULTS_DIR, f"results_fmri_full_1500{suffix}.pkl"))

    # Unnecessary metrics
    # r['top_ngrams_test_correct_score'] = correct_ngrams_module_scores # these scores are basically just 0/1 for each ngram
    # r['expl_test_bert_score'] = r.progress_apply(lambda row: m4_evaluate.test_ngrams_bert_score(
    # row['top_explanation_init_strs'], row['top_ngrams_test'].tolist()), axis=1)

    # Calculate test ngram correctness
    # num_top_ngrams_test = 75
    # test_correct_score_list, correct_ngrams_test_list = m4_evaluate.calc_frac_correct_score(r, col_ngrams=f'top_ngrams_test_{num_top_ngrams_test}')
    # r['top_ngrams_test_correct'] = correct_ngrams_test_list
    # r['frac_top_ngrams_test_correct'] = r['top_ngrams_test_correct'].apply(lambda x: len(x) / num_top_ngrams_test)
