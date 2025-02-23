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
    4: 'Negative Emotional Reactions',
}

FED_DRIVING_EXPLANATIONS_S02 = {
    0: 'Secretive Or Covert Actions',
    1: 'Introspection',
    2: 'Relationships',
    3: 'Sexual and Romantic Interactions',
    4: 'Dialogue',
}


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
