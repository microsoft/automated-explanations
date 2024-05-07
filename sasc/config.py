import os.path
from os.path import join
# if os.path.exists('/home/chansingh'):
REPO_DIR = '/home/chansingh/automated-explanations/'
RESULTS_DIR = '/home/chansingh/automated-explanations/results/'
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
