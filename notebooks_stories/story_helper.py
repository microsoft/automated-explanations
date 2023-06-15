import numpy as np
import re
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def get_start_end_times(timing, paragraphs):
    idx = 0
    start_times = []
    end_times = []
    for para in paragraphs:
        words = para.split()
        start_times.append(timing["time_running"][idx])
        for word in words:
            assert timing["word"][idx] == word, (idx, timing["word"][idx], word)
            idx += 1
            if idx == len(timing):
                break
        end_times.append(timing["time_running"][idx - 1])
    start_times = (np.array(start_times) // 2).astype(int)
    end_times = (np.array(end_times) // 2).astype(int)
    return start_times, end_times


def get_resp_chunks(timing, paragraphs, resp_story, offset=8, apply_offset=True):
    # return responses for each paragraph (after applying offset)

    resp_chunks = []
    start_times, end_times = get_start_end_times(timing, paragraphs)

    for i in range(len(start_times)):
        resp_paragraph = resp_story[:, start_times[i] : end_times[i]]
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


def find_all_examples_within_quotes(x: str):
    # return a list of strings that are within quotes
    # e.g. This is a "hello" world string about "cars" -> ["hello", "cars"]

    # find all indices of quotes
    idxs = []

    idxs = [m.start() for m in re.finditer('"', x)]
    if len(idxs) % 2 != 0:
        raise ValueError("Uneven number of quotes")

    # find all strings within quotes
    examples = []
    for i in range(0, len(idxs), 2):
        examples.append(x[idxs[i] + 1 : idxs[i + 1]])

    return examples

def save_figs_to_single_pdf(filename):
    p = PdfPages(filename)
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs: 
        fig.savefig(p, format='pdf', bbox_inches='tight') 
    p.close()  
