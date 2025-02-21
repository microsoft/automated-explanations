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
    FULL_SETTINGS = [
        # ('UTS02', 'default', 'pilot_story_data.pkl'),
        # ('UTS03', 'default', 'pilot3_story_data.pkl'),
        # ('UTS03', 'default', 'pilot3_story_data.pkl'),
        # ('UTS01', 'default', 'pilot4_story_data.pkl'),

        # ('UTS02', 'qa', 'pilot5_story_data.pkl'),
        ('UTS02', 'roi', 'pilot5_story_data.pkl'),
        # ('UTS02', 'roi', 'pilot6_story_data.pkl'),
        # ('UTS03', 'roi', 'pilot7_story_data.pkl'),
        # ('UTS03', 'roi', 'pilot8_story_data.pkl'),
    ]
    for idx in range(len(FULL_SETTINGS)):

        subject, setting, pilot_name = FULL_SETTINGS[idx]

        # pilot_name = 'pilot5_story_data.pkl'
        pilot_name_abbrev = pilot_name.split("_")[0]
        out_dir = join(
            RESULTS_DIR, "processed", "flatmaps_all", subject, setting + '_' + pilot_name_abbrev)

        stories_data_dict = joblib.load(
            join(config.RESULTS_DIR, 'processed', pilot_name))
        if pilot_name == 'pilot_story_data.pkl':
            pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20230504')
        elif pilot_name == 'pilot3_story_data.pkl':
            pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20231106')
        elif pilot_name == 'pilot4_story_data.pkl':
            pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20240509')
        elif pilot_name == 'pilot5_story_data.pkl':
            pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20240604')
        elif pilot_name == 'pilot6_story_data.pkl':
            # pilot_data_dir = 'out'
            pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20241202')
        elif pilot_name == 'pilot7_story_data.pkl':
            pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20241204')
            # pilot_data_dir = join('out')
        elif pilot_name == 'pilot8_story_data.pkl':
            pilot_data_dir = join(config.PILOT_STORY_DATA_DIR, '20241204')
            # pilot_data_dir = join('out')

        default_story_idxs = np.where(
            (np.array(stories_data_dict['story_setting']) == setting)
            # (np.array(stories_data_dict['story_setting']) == 'default')
            #  |
            # (np.array(stories_data_dict['story_setting']) == 'roi') |
            # (np.array(stories_data_dict['story_setting']) == 'qa')
        )[0]
        # print('story_idxs', default_story_idxs)
        resp_np_files = [stories_data_dict['story_name_new'][i].replace('_resps', '')
                         for i in default_story_idxs]
        resps_dict = {
            k: np.load(join(pilot_data_dir, k))
            for k in tqdm(resp_np_files)
        }

        # get chunked resps
        resp_chunks_list = []
        resp_chunks_list_full = []
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
            if pilot_name in ['pilot3_story_data.pkl']:
                paragraphs = [sasc.analyze_helper.remove_repeated_words(
                    p) for p in paragraphs]
            assert len(paragraphs) == len(
                rows), f"{len(paragraphs)} != {len(rows)}"
            resp_chunks = analyze_helper.get_resps_for_paragraphs(
                timing, paragraphs, resp_story, offset=2, validate=True,
                split_hyphens=pilot_name in ["pilot6_story_data.pkl", "pilot7_story_data.pkl", "pilot8_story_data.pkl"])
            assert len(resp_chunks) <= len(paragraphs)
            args = np.argsort(rows["expl"].values)
            if len(resp_chunks) == len(args) - 1:
                resp_chunks.append(np.full(resp_chunks[0].shape, np.nan))
            resp_chunks_list.append(
                [np.nanmean(resp_chunks[i], axis=1) for i in args])
            resp_chunks_list_full.append(
                [resp_chunks[i] for i in args])
            # print(resp_chunks[0].shape)

        # resp_chunks_list: (num_stories, num_paragraphs, num_voxels, num_trs)
        # want resp_chunks_concat averaged over num_stories and concatenated over num_trs (num_paragraphs, num_voxels, num_trs)
        resp_chunks_concat = []
        num_paragraphs = len(resp_chunks_list[0])
        for i in range(num_paragraphs):
            resp_chunks_concat.append(
                np.concatenate([resp_chunks_list_full[j][i] for j in range(len(resp_chunks_list_full))], axis=-1))
        # resp_chunks_concat: (num_paragraphs, num_voxels, num_stories*num_trs)
        # print(resp_chunks_concat)

        # print('shape', np.concatenate(resp_chunks_list).shape)
        resp_chunks_arr = np.nanmean(np.array(resp_chunks_list), axis=0)
        # resp_chunks_arr = np.array([np.nanmean(x, axis=-1)
        #    for x in resp_chunks_concat])
        # rw = rw.sort_values(by="expl")
        rows = rows.sort_values(by="expl")
        expls = rows["expl"].values
        rows['resp_chunks'] = [resp_chunks_arr[i]
                               for i in range(len(resp_chunks_arr))]

        # save average responses
        if 'module_num' not in rows.columns:
            rows['module_num'] = None
        resp_avg_dict = {
            (rows.iloc[i]['expl'], rows.iloc[i]['module_num']): resp_chunks_arr[i] for i in range(len(resp_chunks_arr))
        }
        resp_concat_dict = {
            (rows.iloc[i]['expl'], rows.iloc[i]['module_num']): resp_chunks_concat[i] for i in range(len(resp_chunks_concat))
        }
        # rw['resp_chunks'] = resp_chunks_arr
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(resp_avg_dict, join(
            out_dir, f'resps_avg_dict_{pilot_name_abbrev}.pkl'))
        joblib.dump(resp_concat_dict, join(
            out_dir, f'resps_concat_dict_{pilot_name_abbrev}.pkl'))

        # for i in tqdm(range(resp_chunks_arr.shape[0])):
        #     # joblib.dump(
        #     #     resp_chunks_arr[i],
        #     #     join(out_dir, f"avg_resp_{i}_{expls[i]}.jl"))
        #     os.makedirs(out_dir, exist_ok=True)
        #     sasc.viz.quickshow(
        #         resp_chunks_arr[i],
        #         subject=subject,
        #         fname_save=join(
        #             out_dir, f"flatmap_{i}_{expls[i]}.pdf"
        #         ),
        #         title=expls[i],
        #     )
        #     plt.cla()
        #     plt.close()
        # print('Finished saving flatmaps')
