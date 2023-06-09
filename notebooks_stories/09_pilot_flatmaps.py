import os
import matplotlib.pyplot as plt
from os.path import join, dirname
from tqdm import tqdm
import joblib
import numpy as np
import cortex
import story_helper
path_to_current_file = dirname(os.path.abspath(__file__))
path_to_repo = dirname(path_to_current_file)


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
    story_data = joblib.load(join(path_to_repo, 'results/pilot_story_data.pkl'))
    resp_chunks_list = []
    for story_num in range(6): # range(1, 7)
        rows = story_data["rows"][story_num]
        rw = rows[
            [
                "expl",
                "module_num",
                "top_explanation_init_strs",
                "subject",
                "fmri_test_corr",
                "top_score_synthetic",
                "roi_anat",
                "roi_func",
            ]
        ]
        paragraphs = story_data["story_text"][story_num].split("\n\n")
        assert len(paragraphs) == len(rw), (len(paragraphs), len(rw))
        timing = story_data["timing"][story_num]

        resp_story = resps_dict[story_data["story_name_new"][story_num]].T  # (voxels, time)
        resp_chunks = story_helper.get_resp_chunks(timing, paragraphs, resp_story)

        args = np.argsort(rw["expl"].values)
        resp_chunks_list.append([resp_chunks[i].mean(axis=1) for i in args])

    resp_chunks_arr = np.array(resp_chunks_list).mean(axis=0)
    expls = rw.sort_values(by="expl")["expl"].values
    for i in range(resp_chunks_arr.shape[0]):
        quickshow(resp_chunks_arr[i], subject="UTS02", fname_save=join(path_to_repo, 'results', 'pilot_plots', f"_resps_flatmap_{i}_{expls[i]}.pdf"))
        plt.cla()
        plt.close()
        
    
