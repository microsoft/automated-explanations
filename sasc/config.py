import os.path
from os.path import join, dirname, abspath
# if os.path.exists('/home/chansingh'):
path_to_file = dirname(dirname(abspath(__file__)))
# REPO_DIR = '/home/chansingh/automated-explanations/'
REPO_DIR = path_to_file
RESULTS_DIR = join(REPO_DIR, 'results')
FMRI_DIR = '/home/chansingh/mntv1/deep-fMRI/'

STORIES_DIR = join(RESULTS_DIR, "stories")
SAVE_DIR_FMRI = join(FMRI_DIR, 'sasc', 'rj_models')
# '/home/chansingh/mntv1/mprompt/cache'
CACHE_DIR = join(FMRI_DIR, 'sasc', 'mprompt', 'cache')
# else:
# RESULTS_DIR = '/accounts/campus/aliyahhsu/module-prompt/results'
# SAVE_DIR_FMRI = '/var/tmp/ah_cache'
# REPO_DIR = '/accounts/campus/aliyahhsu/module-prompt'
# CACHE_DIR = '/var/tmp/ah_cache'


# brain_tune
PILOT_STORY_DATA_DIR = '/home/chansingh/mntv1/deep-fMRI/brain_tune/story_data'

cache_ngrams_dir = join(FMRI_DIR, 'sasc/mprompt/cache/cache_ngrams')
regions_idxs_dir = join(FMRI_DIR, 'sasc/brain_regions')

if __name__ == '__main__':
    print('REPO_DIR:', REPO_DIR)