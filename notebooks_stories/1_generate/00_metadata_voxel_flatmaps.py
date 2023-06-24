import os
import matplotlib.pyplot as plt
from os.path import join, dirname
from tqdm import tqdm
import joblib
import numpy as np
import sasc.viz
import pandas as pd
story_generate = __import__('01_story_generate')

subject = 'UTS02'
rows = story_generate.get_rows_voxels(subject=subject)
rows['top_ngrams_module_correct'] = rows['top_ngrams_module_correct'].apply(lambda x: x[:5])
path_to_current_file = dirname(os.path.abspath(__file__))
path_to_repo = dirname(path_to_current_file)


if __name__ == "__main__":
    print("loading rsep of correct size...")
    n_voxels_dict = {
        "UTS01": 1000,
        "UTS02": 1000,
        "UTS03": 1000,
    }
    pilot_data_dir = "/home/chansingh/mntv1/deep-fMRI/story_data/20230504"
    resp_np_files = os.listdir(pilot_data_dir)
    n_voxels = np.load(join(pilot_data_dir, resp_np_files[0])).shape[1]
    resp_arr = np.zeros(n_voxels)
    print("resp_arr.shape", resp_arr.shape)

    story_data = joblib.load(join(path_to_repo, "results/pilot_story_data.pkl"))
    rows = story_data["rows"][0]
    voxel_nums = rows["voxel_num"].values

    resp_arr[voxel_nums] = 1
    sasc.viz.quickshow(
        resp_arr,
        subject="UTS02",
        fname_save=join(path_to_repo, "results", "pilot_plots", "voxel_locations"),
    )
    plt.cla()
    plt.close()
