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
    out_dir = join(RESULTS_DIR, "processed", "flatmaps")
    # pilot_name = 'pilot_story_data.pkl'
    pilot_name = 'pilot5_story_data.pkl'

    stories_data_dict = joblib.load(
        join(config.RESULTS_DIR, 'processed', pilot_name))
    if pilot_name == 'pilot_story_data.pkl':
        pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20230504')
    elif pilot_name == 'pilot5_story_data.pkl':
        pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20240604')

    default_story_idxs = np.where(
        (np.array(stories_data_dict['story_setting']) == 'default') |
        # (np.array(stories_data_dict['story_setting']) == 'roi') |
        (np.array(stories_data_dict['story_setting']) == 'qa')
    )[0]
    print('story_idxs', default_story_idxs)
    resp_np_files = [stories_data_dict['story_name_new'][i].replace('_resps', '')
                     for i in default_story_idxs]
    resps_dict = {
        k: np.load(join(pilot_data_dir, k))
        for k in tqdm(resp_np_files)
    }

    # get chunked resps
    resp_chunks_list = []
    for story_num in default_story_idxs:
        rows = stories_data_dict["rows"][story_num]
        # get resp_chunks
        resp_story = resps_dict[
            stories_data_dict["story_name_new"][story_num].replace(
                '_resps', '')
        ].T  # (voxels, time)
        timing = stories_data_dict["timing"][story_num]
        timing['time_running'] = timing['time_running']
        if 'paragraphs' in stories_data_dict.keys():
            paragraphs = stories_data_dict["paragraphs"][story_num]
        else:
            paragraphs = stories_data_dict["story_text"][story_num].split(
                "\n\n")
        assert len(paragraphs) == len(
            rows), f"{len(paragraphs)} != {len(rows)}"
        resp_chunks = analyze_helper.get_resps_for_paragraphs(
            timing, paragraphs, resp_story, offset=2, validate=True)
        assert len(resp_chunks) <= len(paragraphs)
        args = np.argsort(rows["expl"].values)
        resp_chunks_list.append([resp_chunks[i].mean(axis=1) for i in args])

    resp_chunks_arr = np.array(resp_chunks_list).mean(axis=0)
    # rw = rw.sort_values(by="expl")
    rows = rows.sort_values(by="expl")
    expls = rows["expl"].values
    rows['resp_chunks'] = [resp_chunks_arr[i]
                           for i in range(len(resp_chunks_arr))]
    if 'module_num' not in rows.columns:
        rows['module_num'] = None
    resp_avg_dict = {
        (rows.iloc[i]['expl'], rows.iloc[i]['module_num']): resp_chunks_arr[i] for i in range(len(resp_chunks_arr))
    }
    # rw['resp_chunks'] = resp_chunks_arr
    os.makedirs(out_dir, exist_ok=True)
    pilot_name_abbrev = pilot_name.split("_")[0]
    joblib.dump(resp_avg_dict, join(
        out_dir, f'resps_avg_dict_{pilot_name_abbrev}.pkl'))

    for i in tqdm(range(resp_chunks_arr.shape[0])):
        # joblib.dump(
        #     resp_chunks_arr[i],
        #     join(out_dir, f"avg_resp_{i}_{expls[i]}.jl"))
        sasc.viz.quickshow(
            resp_chunks_arr[i],
            subject="UTS02",
            fname_save=join(
                out_dir, f"flatmap_{pilot_name_abbrev}_{i}_{expls[i]}.pdf"
            ),
            title=expls[i],
        )
        plt.cla()
        plt.close()
