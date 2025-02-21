from copy import deepcopy
import joblib
import numpy as np
import sys
from os.path import abspath, dirname, join
try:
    from sasc.config import FMRI_DIR, STORIES_DIR, RESULTS_DIR, CACHE_DIR, cache_ngrams_dir, regions_idxs_dir
except ImportError:
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

FED_DRIVING_EXPLANATIONS_S03 = {
    0: 'Relationships',
    1: 'Positive Emotional Reactions',
    2: 'Body parts',
    3: 'Dialogue',
}

FED_DRIVING_EXPLANATIONS_S02 = {
    0: 'Secretive Or Covert Actions',
    1: 'Introspection',
    2: 'Relationships',
    3: 'Sexual and Romantic Interactions',
}


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
        elif subject == 'S02':
            rois_fedorenko = joblib.load(join(
                FMRI_DIR, 'brain_tune/voxel_neighbors_and_pcs/', 'lang_localizer_UTS02_aligned.jbl'))
        return {
            'Lang-' + str(i): rois_fedorenko[i] for i in range(len(rois_fedorenko))
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
