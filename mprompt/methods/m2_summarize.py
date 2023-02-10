import re
from typing import Any, List, Mapping, Optional
import numpy as np
import openai
import os.path
from os.path import join
import pickle as pkl
from langchain.llms.base import LLM
from mprompt.llm import get_llm
from langchain import PromptTemplate

def summarize_ngrams(
    llm: LLM,
    ngrams_list: List[str],
    num_summaries: int=2,
    prefix_str='Here is a list of phrases:',
    suffix_str='What is a common theme among these phrases?\nThe common theme among these phrases is',
    num_top_ngrams: int = 40,
    # seed: int = 0,
) -> List[str]:
    """Refine a keyphrase by making a call to the llm
    """
    bullet_list_ngrams = '- ' + '\n- '.join(ngrams_list[:num_top_ngrams])
    prompt = prefix_str + '\n\n' + bullet_list_ngrams + '\n\n' + suffix_str
    print(prompt)
    
    summaries = []
    for i in range(num_summaries):
        summary = llm(prompt)

        # clean up summary
        summary = summary.strip()
        # if summary.startswith('that'):

        '''
        # clean up the keyphrases
        # (split the string s on any numeric character)
        ks = [
            k.replace('.', '').strip()
            for k in re.split(r'\d', refined_keyphrase) if k.strip()
        ]

        ks = list(set(ks))  # remove duplicates
        ks = [k.lower() for k in ks if len(k) > 2]
        '''

        summaries.append(summary)

    return summaries


if __name__ == '__main__':
    llm = get_llm(checkpoint='google/flan-t5-xxl')
    summary = summarize_ngrams(llm, ['cat', 'dog', 'bird', 'elephant', 'cheetah'])
    print('summary', repr(summary))