import time
import os


def display_words(words_and_timings):
    for word, duration in words_and_timings:
        # Clear the console
        os.system('cls' if os.name == 'nt' else 'clear')
        # Display the word
        print(word)
        # Wait for the specified duration
        time.sleep(duration)


# Example list of words and their respective timings (in seconds)
words_and_timings = [
    ("Hello", 1),
    ("World", 2),
    ("This", 1.5),
    ("is", 1),
    ("a", 0.5),
    ("test", 2)
]

display_words(words_and_timings)
