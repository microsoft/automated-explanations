import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def colorize(words: List[str], color_array: np.ndarray[float],
             char_width_max=60, title: str=None, subtitle: str=None):
    '''
    Colorize a list of words based on a color array.
    color_array
        an array of numbers between 0 and 1 of length equal to words
    '''
    cmap = matplotlib.cm.get_cmap('viridis')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    char_width = 0
    for word, color in zip(words, color_array):
        char_width += len(word)
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
        if char_width >= char_width_max:
            colored_string += '</br>'
            char_width = 0

    if subtitle:
        colored_string = f'<h5>{subtitle}</h5>\n' + colored_string
    if title:
        colored_string = f'<h3>{title}</h3>\n' + colored_string
    return colored_string

def moving_average(a, n=3):
    assert n % 2 == 1, 'n should be odd'
    diff = n // 2
    vals = []
    # calculate moving average in a window 2
    # (1, 4)
    for i in range(diff, len(a) + diff):
        l = i - diff
        r = i + diff + 1
        vals.append(np.mean(a[l: r]))
    return np.nan_to_num(vals)