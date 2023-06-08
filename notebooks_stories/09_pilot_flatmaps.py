import os
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm
import joblib
import numpy as np
import cortex


def quickshow(X: np.ndarray, subject="UTS03", fname_save=None):
    """
    Actual visualizations
    Note: for this to work, need to point the cortex config filestore to the `ds003020/derivative/pycortex-db` directory.
    This might look something like `/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/UTS03/anatomicals/`
    """
    vol = cortex.Volume(X, subject, xfmname=f"{subject}_auto")
    # , with_curvature=True, with_sulci=True)
    cortex.quickshow(vol, with_rois=True, cmap="PuBu")
    if fname_save is not None:
        plt.savefig(fname_save)
        plt.savefig(fname_save.replace(".pdf", ".png"))
        plt.close()


if __name__ == '__main__':
    # load data and corresponding resps
    pilot_data_dir = "/home/chansingh/mntv1/deep-fMRI/story_data/20230504"
    resp_np_files = os.listdir(pilot_data_dir)
    resps_dict = {k: np.load(join(pilot_data_dir, k)) for k in tqdm(resp_np_files)}

    # viz mean resp for all voxels (this has been normalized, so is extremely close to zero)
    """
    resps_arr = np.array([np.mean(arr, axis=0) for arr in list(resps_dict.values())]).mean(
        axis=0
    )
    # joblib.dump(resps_arr, "_resps.pkl")
    quickshow(resps_arr, subject="UTS02", fname_save="_resps_flatmap.pdf")
    """

    # viz mean driving resp for each of the 16 voxels
    for i in range(16):
