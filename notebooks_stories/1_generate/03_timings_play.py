import time
import os
from os.path import join
import pandas as pd
from sasc.config import RESULTS_DIR
setting = "qa"
EXPT_NAMES = [
    k
    for k in os.listdir(join(RESULTS_DIR, "stories", setting))
    # if "uts03" in k or "uts01" in k
]


def display_words(words_and_timings):
    for word, duration in words_and_timings:
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')
        # Display the word
        print(word)
        # Wait for the specified duration
        time.sleep(duration)


# Example list of words and their respective timings (in seconds)
words_and_timings = pd.read_csv(
    join(RESULTS_DIR, 'stories', setting, EXPT_NAMES[0], 'timings_processed.csv'))
words_and_timings = words_and_timings[['word', 'timing']].values.tolist()
display_words(words_and_timings)
