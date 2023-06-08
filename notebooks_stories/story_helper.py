import numpy as np

def get_resp_chunks(timing, paragraphs, resp_story, OFFSET=8):
    # return responses for each paragraph (after applying offset)

    resp_chunks = []
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

    for i in range(len(start_times)):
        resp_paragraph = resp_story[:, start_times[i] : end_times[i]]
        while resp_paragraph.shape[1] <= 2 * OFFSET:
            OFFSET -= 1
        resp_chunks.append(resp_paragraph[:, OFFSET:-OFFSET])
    return resp_chunks