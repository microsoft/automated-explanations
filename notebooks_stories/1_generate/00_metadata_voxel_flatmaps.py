import os
import matplotlib.pyplot as plt
from os.path import join, dirname
from tqdm import tqdm
import joblib
import numpy as np
import sasc.viz
import pandas as pd


path_to_current_file = dirname(os.path.abspath(__file__))
path_to_repo = dirname(path_to_current_file)
story_generate = __import__("01_generate_story")
from sasc.modules.fmri_module import convert_module_num_to_voxel_num

if __name__ == "__main__":
    subject = "UTS02"

    print("loading resp of correct size...")
    n_voxels_dict = {
        "UTS01": 1000,
        "UTS02": 1000,
        "UTS03": 1000,
    }
    # get n_voxels
    pilot_data_dir = "/home/chansingh/mntv1/deep-fMRI/story_data/20230504"
    resp_np_files = os.listdir(pilot_data_dir)
    n_voxels = np.load(join(pilot_data_dir, resp_np_files[0])).shape[1]
    print(subject, "n_voxels", n_voxels)

    # get voxel_nums
    # story_data = joblib.load(join(path_to_repo, "results/pilot_story_data.pkl"))
    # rows = story_data["rows"][0]
    # voxel_nums = rows["voxel_num"].values

    setting = "polysemantic"
    rows, _, _ = story_generate.get_rows_and_prompts_default(
        subject=subject,
        setting=setting,
        seed=1,
        n_examples_per_prompt_to_consider=35,
        n_examples_per_prompt=3,
        version="v5_noun",
    )
    rows["voxel_num"] = rows.apply(
        lambda row: convert_module_num_to_voxel_num(row["module_num"], row["subject"]),
        axis=1,
    )
    voxel_nums = rows["voxel_num"].values
    print(subject, setting, "voxel_nums", voxel_nums)

    resp_arr = np.zeros(n_voxels)
    resp_arr[voxel_nums] = 1
    sasc.viz.quickshow(
        resp_arr,
        subject="UTS02",
        fname_save=join(path_to_repo, "results", "pilot_plots", "voxel_locations"),
    )
    plt.cla()
    plt.close()
