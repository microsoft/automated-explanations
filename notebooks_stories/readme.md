There were several iterations of these experiments, causing these scripts to get messy. Here's the structure:

## 0_voxel
- This contains code for selecting which voxels to run the experiments on.
- These voxel selections are made based on the results of sasc outputs `results/fmri_results_merged.pkl`
- Voxels are selected using human judgement, as well as scores (stability score started being used in pilot3 onwards)

## 1_generate
- This contains code for generating the stories for the experiments

## 2_analyze_pilot
- This analyzes the responses to the stories
- Important: responses must be aligned to the right part of the stories
  - Must account for trimming
  - Must account for story paragraph metadata not matching perfectly with the `timings_processed.csv` file, which is what actually gets used

Note: to run the code, you need to first run `pip install -e .` from the main repo directory. Then, place the folders from the [fMRI data collection](https://app.box.com/folder/211367364142) into join(sasc.config.FMRI_DIR, 'story_data')