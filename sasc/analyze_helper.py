from collections import defaultdict
import numpy as np
import re
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import imodelsx.linear_finetune
import string
from typing import List
import pandas as pd


def _remove_punc(s):
    return s.translate(str.maketrans("", "", string.punctuation))


def get_start_end_indexes_for_paragraphs(timing: pd.DataFrame, paragraphs: List[str], validate=False, split_hyphens=False):
    """Returns start/end indexes for each paragraph in the story.
    If the entire story was not played, the number of start/end times will be less than the number of paragraphs.
    """
    idx = 0
    start_times = []
    end_times = []
    # display(timing['word'][5])
    for para in paragraphs:
        if split_hyphens:
            words = re.split(r'\s|-|—', para)
        else:
            words = para.split()
        start_times.append(timing["time_running"][idx])
        for word in words:
            if validate:
                word_in_timings = timing["word"][idx]
                word_in_story = word
                # print(timing['word'][idx])
                assert _remove_punc(word_in_timings) == _remove_punc(word_in_story), (
                    idx,
                    word,
                    str(timing["word"][idx]),
                )
            idx += 1
            if idx == len(timing):
                # print('break!!!!')
                break
            # print(idx, len(timing))
        end_times.append(timing["time_running"][idx - 1])

        if idx == len(timing):
            break

    start_indexes = (np.array(start_times) //
                     2).astype(int)  # - offset_trim
    end_indexes = (np.array(end_times) // 2).astype(int)  # - offset_trim
    start_indexes = start_indexes.clip(min=0)
    end_indexes = end_indexes.clip(min=0)
    return start_indexes, end_indexes


def get_resps_for_paragraphs(
        timing, paragraphs, resp_story, offset=2,
        apply_offset=True, trim=5, validate=False,
        split_hyphens=False,
) -> List[np.ndarray]:
    '''Return responses for each paragraph (after applying offset)

    Params
    ------
    offset: int
        Number of TRs to remove from the beginning and end of each paragraph.
        The offset is applied to each paragraph separately.

    Returns
    -------
    resp_chunks : List[np.ndarray]
        List of responses for each paragraph.
        Each response is a 2D array of shape (n_features, n_timepoints)
    '''
    resp_chunks = []
    start_indexes, end_indexes = get_start_end_indexes_for_paragraphs(
        timing, paragraphs, validate=validate, split_hyphens=split_hyphens)

    for i in range(len(start_indexes)):
        # get paragraph
        start_idx = max(0, start_indexes[i] - trim)
        end_idx = end_indexes[i] - trim
        resp_paragraph = resp_story[:, start_idx: end_idx]

        # apply offset
        if apply_offset:
            while resp_paragraph.shape[1] <= 2 * offset:
                offset -= 1
            resp_paragraph = resp_paragraph[:, offset:-offset]
        resp_chunks.append(resp_paragraph)

        # find the middle 3 values of resp_paragraph
        # # # mid = resp_paragraph.shape[1] // 2
        # # resp_middle = resp_paragraph[:, mid - 1 : mid + 2]
        # mat[:, i] = resp_middle.mean(axis=1)

    return resp_chunks


def _get_word_chunks(t):
    '''
    Split words based on timing into bins of 2 seconds
    '''
    word_chunks = []
    current_time = 0
    for i in range(len(t)):
        if i == 0:
            word_chunks.append([t['word'][i]])
        elif t['time_running'][i] - current_time < 2:
            word_chunks[-1].append(t['word'][i])
        else:
            word_chunks.append([t['word'][i]])
            current_time += 2
    return word_chunks


def compute_word_chunk_deltas_for_single_paragraph(
        start_times, end_times, voxel_resp,
        word_chunks_contain_example_ngrams, voxel_num,
        deltas=range(1, 8)
):

    i_start_voxel = start_times[voxel_num]
    i_end_voxel = end_times[voxel_num] + 1
    if i_end_voxel > len(voxel_resp):
        i_end_voxel = len(voxel_resp)
    idxs = np.arange(i_start_voxel, i_end_voxel)
    idxs_wc = np.where(word_chunks_contain_example_ngrams[idxs])[0]

    word_chunk_deltas = defaultdict(list)
    for delta_num in deltas:
        for idx in idxs_wc:
            if idx + delta_num < len(idxs):
                word_chunk_deltas[delta_num].append(
                    voxel_resp[idxs][idx + delta_num] - voxel_resp[idxs][idx]
                )
    return word_chunk_deltas


def find_all_examples_within_quotes(x: str):
    '''
    return a list of strings that are within quotes
    e.g. This is a "hello" world string about "cars" -> ["hello", "cars"]
    '''

    # find all indices of quotes
    idxs = []

    idxs = [m.start() for m in re.finditer('"', x)]
    if len(idxs) % 2 != 0:
        raise ValueError("Uneven number of quotes")

    # find all strings within quotes
    examples = []
    for i in range(0, len(idxs), 2):
        examples.append(x[idxs[i] + 1: idxs[i + 1]])

    return examples


def save_figs_to_single_pdf(filename):
    p = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(p, format="pdf", bbox_inches="tight")
    p.close()


def sort_expls_semantically(expls: List[str], device='cpu'):
    # from sasc.config import RESULTS_DIR, REPO_DIR
    # r = pd.read_pickle(join(RESULTS_DIR, "results_fmri_full_1500.pkl"))
    # expls_full = r['top_explanation_init_strs']
    expls_full = expls
    pca = PCA(n_components=1)
    lf = imodelsx.linear_finetune.LinearFinetune(device=device)
    embs_full = lf._get_embs(expls_full)
    pca.fit(embs_full)

    embs = lf._get_embs(expls)
    coefs = pca.transform(embs).flatten()
    ordering = np.argsort(coefs).flatten()
    return ordering


def remove_repeated_words(paragraph):
    words = paragraph.split()
    new_words = [words[0]]
    for w in words[1:]:
        if w != new_words[-1]:
            new_words.append(w)
    return ' '.join(new_words)
