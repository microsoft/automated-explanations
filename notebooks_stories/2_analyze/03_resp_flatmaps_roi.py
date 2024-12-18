from collections import defaultdict
import os
import matplotlib.pyplot as plt
from os.path import join, dirname
from tqdm import tqdm
import joblib
import numpy as np
from sasc import analyze_helper
import sasc.viz
from sasc.config import RESULTS_DIR
from sasc import config


if __name__ == "__main__":
    pilot_name = 'pilot5_story_data.pkl'
    pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20240604')
    stories_data_dict = joblib.load(
        join(config.RESULTS_DIR, 'processed', pilot_name))

    # load responses
    default_story_idxs = np.where(
        (np.array(stories_data_dict['story_setting']) == 'roi')
    )[0]
    resp_np_files = [stories_data_dict['story_name_new'][i].replace('_resps', '')
                     for i in default_story_idxs]
    resps_dict = {
        k: np.load(join(pilot_data_dir, k))
        for k in tqdm(resp_np_files)
    }

    resp_chunks_list = defaultdict(list)
    for story_num in default_story_idxs:
        rows = stories_data_dict["rows"][story_num]

        # get resp_chunks
        resp_story = resps_dict[
            stories_data_dict["story_name_new"][story_num].replace(
                '_resps', '')
        ].T  # (voxels, time)
        timing = stories_data_dict["timing"][story_num]
        if 'paragraphs' in stories_data_dict.keys():
            paragraphs = stories_data_dict["paragraphs"][story_num]
        else:
            paragraphs = stories_data_dict["story_text"][story_num].split(
                "\n\n")

        assert len(paragraphs) == len(
            rows), f"{len(paragraphs)} != {len(rows)}"
        resp_chunks = analyze_helper.get_resps_for_paragraphs(
            timing, paragraphs, resp_story, offset=2)
        assert len(resp_chunks) <= len(paragraphs)

        for i, roi in enumerate(rows['roi'].values):
            resp_chunks_list[roi].append(resp_chunks[i].mean(axis=1))

    resp_avg_dict = {
        roi: np.array(resp_chunks_list[roi]).mean(axis=0)
        for roi in resp_chunks_list.keys()
    }
    rw = rows.sort_values(by="roi")
    # expls = rw["expl"].values
    rois = rw["roi"].values
    # rw['resp_chunks'] = [resp_chunks_arr[i]
    #  for i in range(len(resp_chunks_arr))]
    # rw['resp_chunks'] = resp_chunks_arr
    os.makedirs(join(RESULTS_DIR, 'processed', 'flatmaps_roi'), exist_ok=True)
    joblib.dump(resp_avg_dict, join(RESULTS_DIR, 'processed', 'flatmaps_roi',
                                    'resp_avg_dict.pkl'))

    for roi in tqdm(rois):
        # joblib.dump(
        # resp_avg_dict[roi],
        # join(RESULTS_DIR, 'processed', 'flatmaps_roi', f"avg_resp_{i}_{roi}.jl"))
        sasc.viz.quickshow(
            resp_avg_dict[roi],
            subject="UTS02",
            fname_save=join(
                RESULTS_DIR,
                "processed",
                'flatmaps_roi',
                f"flatmap_{roi}.png",
            ),
            title=rois[i],
        )
        plt.cla()
        plt.close()

    # resp_avg_dict = joblib.load(join(RESULTS_DIR, 'processed', 'flatmaps_roi',
    #  'resp_avg_dict.pkl'))

    # print(resp_avg_dict.keys())
