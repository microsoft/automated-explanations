import os.path
if os.path.exists('/home/chansingh'):
    RESULTS_DIR = '/home/chansingh/mprompt/results/'
    SAVE_DIR_FMRI = '/home/chansingh/mntv1/deep-fMRI/opt_model/'
    REPO_DIR = '/home/chansingh/mprompt/'
    CACHE_DIR = '/home/chansingh/mntv1/mprompt/cache'
else:
    RESULTS_DIR = '/accounts/campus/aliyahhsu/module-prompt/results'
    SAVE_DIR_FMRI = '/var/tmp/ah_cache'
    REPO_DIR = '/accounts/campus/aliyahhsu/module-prompt'
    CACHE_DIR = '/var/tmp/ah_cache'
    

