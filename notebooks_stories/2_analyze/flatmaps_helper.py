from copy import deepcopy
import joblib
import numpy as np
import sys
import os
from os.path import abspath, dirname, join
# try:
# from sasc.config import FMRI_DIR, STORIES_DIR, RESULTS_DIR, CACHE_DIR, cache_ngrams_dir, regions_idxs_dir
# except ImportError:
repo_path = dirname(dirname(dirname(abspath(__file__))))
RESULTS_DIR = join(repo_path, 'results')

sys.path.append('../notebooks')
VOX_COUNTS = {
    'S02': 94251,
    'S03': 95556,
}

ROI_EXPLANATIONS_S03 = {
    'EBA': 'Body parts',
    'IPS': 'Descriptive elements of scenes or objects',
    'OFA': 'Conversational transitions',
    'OPA': 'Direction and location descriptions',
    'OPA_only': 'Self-reflection and growth',
    'PPA': 'Scenes and settings',
    'PPA_only': 'Garbage, food, and household items',
    'RSC': 'Travel and location names',
    'RSC_only': 'Location names',
    'sPMv': 'Dialogue and responses',
}


def load_flatmaps(normalize_flatmaps, load_timecourse=False, explanations_only=False):
    # S02
    gemv_flatmaps_default = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps", 'resps_avg_dict_pilot.pkl'))
    gemv_flatmaps_roi_qa = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps", 'resps_avg_dict_pilot5.pkl'))
    gemv_flatmaps_roi_custom = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", 'UTS02', 'roi_pilot6', 'resps_avg_dict_pilot6.pkl'))
    # gemv_flatmaps_dict = gemv_flatmaps_default | gemv_flatmaps_roi_qa | gemv_flatmaps_roi_custom
    gemv_flatmaps_dict_S02 = gemv_flatmaps_roi_custom

    # S03
    gemv_flatmaps_default = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'default', 'resps_avg_dict_pilot3.pkl'))
    gemv_flatmaps_roi_custom1 = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'roi_pilot7', 'resps_avg_dict_pilot7.pkl'))
    gemv_flatmaps_roi_custom2 = joblib.load(join(
        RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'roi_pilot8', 'resps_avg_dict_pilot8.pkl'))
    # gemv_flatmaps_dict = gemv_flatmaps_default | gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2
    gemv_flatmaps_dict_S03 = gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2

    if load_timecourse:
        gemv_flatmaps_roi_custom = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", 'UTS02', 'roi_pilot6', 'resps_concat_dict_pilot6.pkl'))

        gemv_flatmaps_dict_S02_timecourse = gemv_flatmaps_roi_custom

        gemv_flatmaps_roi_custom1 = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'roi_pilot7', 'resps_concat_dict_pilot7.pkl'))
        gemv_flatmaps_roi_custom2 = joblib.load(join(
            RESULTS_DIR, "processed", "flatmaps_all", 'UTS03', 'roi_pilot8', 'resps_concat_dict_pilot8.pkl'))
        gemv_flatmaps_dict_S03_timecourse = gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2

        return gemv_flatmaps_dict_S02, gemv_flatmaps_dict_S03, gemv_flatmaps_dict_S02_timecourse, gemv_flatmaps_dict_S03_timecourse

    # normalize flatmaps
    if normalize_flatmaps:
        for k, v in gemv_flatmaps_dict_S03.items():
            flatmap_unnormalized = gemv_flatmaps_dict_S03[k]
            gemv_flatmaps_dict_S03[k] = (
                flatmap_unnormalized - flatmap_unnormalized.mean()) / flatmap_unnormalized.std()
        for k, v in gemv_flatmaps_dict_S02.items():
            flatmap_unnormalized = gemv_flatmaps_dict_S02[k]
            gemv_flatmaps_dict_S02[k] = (
                flatmap_unnormalized - flatmap_unnormalized.mean()) / flatmap_unnormalized.std()

    return gemv_flatmaps_dict_S02, gemv_flatmaps_dict_S03


def load_custom_rois(subject, suffix_setting='_fedorenko'):
    '''
    Params
    ------
    subject: str
        'S02' or 'S03'
    suffix_setting: str
        '' - load custom communication rois
        '_fedorenko' - load fedorenko rois
        '_spotlights' - load spotlights rois (there are a ton of these)
    '''
    if suffix_setting == '':
        # rois_dict = joblib.load(join(regions_idxs_dir, f'rois_{subject}.jbl'))
        # rois = joblib.load(join(FMRI_DIR, 'brain_tune/voxel_neighbors_and_pcs/', 'communication_rois_UTS02.jbl'))
        rois = joblib.load(join(FMRI_DIR, 'brain_tune/voxel_neighbors_and_pcs/',
                                f'communication_rois_v2_UT{subject}.jbl'))
        rois_dict_raw = {i: rois[i] for i in range(len(rois))}
        if subject == 'S02':
            raw_idxs = [
                [0, 7],
                [3, 4],
                [1, 5],
                [2, 6],
            ]
        elif subject == 'S03':
            raw_idxs = [
                [0, 7],
                [3, 4],
                [2, 5],
                [1, 6],
            ]
        return {
            'comm' + str(i): np.vstack([rois_dict_raw[j] for j in idxs]).sum(axis=0)
            for i, idxs in enumerate(raw_idxs)
        }
    elif suffix_setting == '_fedorenko':
        if subject == 'S03':
            rois_fedorenko = joblib.load(join(
                FMRI_DIR, 'brain_tune/voxel_neighbors_and_pcs/', 'lang_localizer_UTS03.jbl'))
        return {
            'fed' + str(i): rois_fedorenko[i] for i in range(len(rois_fedorenko))
        }
        # rois_dict = rois_dict_raw
    elif suffix_setting == '_spotlights':
        rois_spotlights = joblib.load(join(
            FMRI_DIR, 'brain_tune/voxel_neighbors_and_pcs/', f'all_spotlights_UT{subject}.jbl'))
        return {'spot' + str(i): rois_spotlights[i][-1]
                for i in range(len(rois_spotlights))}


def load_known_rois(subject):
    nonzero_entries_dict = joblib.load(
        join(regions_idxs_dir, f'rois_{subject}.jbl'))
    rois_dict = {}
    for k, v in nonzero_entries_dict.items():
        mask = np.zeros(VOX_COUNTS[subject])
        mask[v] = 1
        rois_dict[k] = deepcopy(mask)
    if subject == 'S03':
        rois_dict['OPA'] = rois_dict['TOS']
    return rois_dict
