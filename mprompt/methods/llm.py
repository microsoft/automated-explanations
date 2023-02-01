import re
from typing import List
import numpy as np
import openai
import os.path
from os.path import join
import pickle as pkl
    

def summarize_ngrams(
    ngrams_list: List[str],
    prefix_str = 'Here is a list of phrases:',
    suffix_str = 'What is a common theme among these phrases?\nThe common theme among these phrases is',
    num_ngrams: int=40,
    seed: int=0,
):
    """Refine a keyphrase by making a call to gpt-3
    """
    # check cache
    # os.makedirs(cache_dir, exist_ok=True)
    # cache_file = join(cache_dir, f'{keyphrase_str}___{seed}.pkl')
    # print(f'{cache_file=}')
    # if os.path.exists(cache_file):
        # return pkl.load(open(cache_file, 'rb'))

    bullet_list_ngrams =  '- ' + '\n- '.join(ngrams_list[:num_ngrams])
    prompt = prefix_str + '\n\n' + bullet_list_ngrams + '\n\n' + suffix_str
    print(prompt)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.1,
        top_p=1,
        frequency_penalty=0.25,  # maximum is 2
        presence_penalty=0,
        stop=["."]
    )
    summary = response['choices'][0]['text']

    # clean up summary
    summary = summary.strip()
    # if summary.startswith('that'):

    return summary

    '''
    # clean up the keyphrases
    # (split the string s on any numeric character)
    ks = [
        k.replace('.', '').strip()
        for k in re.split(r'\d', refined_keyphrase) if k.strip()
    ]

    ks = list(set(ks))  # remove duplicates
    ks = [k.lower() for k in ks if len(k) > 2]

    pkl.dump(ks, open(cache_file, 'wb'))
    return ks
    '''


if __name__ == '__main__':
    summary = summarize_ngrams(['apple', 'banana', 'cat', 'dog'])
    print('summary', repr(summary))
