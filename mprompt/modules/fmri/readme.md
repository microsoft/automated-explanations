Contains results for fitting subject `UTS03` on the [deep-fMRI-dataset](https://github.com/HuthLab/deep-fMRI-dataset) using embeddings from BERT with 4 delays (see https://github.com/csinva/fmri for more details).

- `weights.npz` contains the weights for the fitted model. These weights are already averaged over the delays, so that they can be directly applied to embeddings.
- `corrs.npz` contains the test correlations for each voxel
- `preproc.pkl` contains the preprocessing that is applied to the embeddings before applying the weights
- `roi_dict.pkl` contains a dictionary mapping voxel indices to ROIs
- `top_ngrams.pkl` contains a dictionary mapping voxel indices to the top ngrams for that voxel, based on responses