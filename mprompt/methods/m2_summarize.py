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
    args,
    llm: LLM,
    ngrams_list: List[str],
    num_summaries: int = 2,
    prefix_str='Here is a list of phrases:',
    suffix_str='What is a common theme among these phrases?\nThe common theme among these phrases is',
    num_top_ngrams: int = 30,
    num_top_ngrams_to_consider: int = 50,
    # seed: int = 0,
) -> List[str]:
    """Refine a keyphrase by making a call to the llm
    """
    rng = np.random.default_rng(args.seed)

    summaries = []
    for i in range(num_summaries):
        # randomly sample num_top_ngrams (preserving ordering)
        idxs = np.sort(
            rng.choice(np.arange(
                min(num_top_ngrams_to_consider, len(ngrams_list))),  # choose from this many ngrams
                size=num_top_ngrams, # choose this many ngrams
                replace=False
            )
        )
        bullet_list_ngrams = '- ' + '\n- '.join(np.array(ngrams_list)[idxs])
        prompt = prefix_str + '\n\n' + bullet_list_ngrams + '\n\n' + suffix_str
        print(prompt)
        summary = llm(prompt)

        # clean up summary
        summary = summary.strip()
        # if summary.startswith('that'):

        if summary.endswith('.'):
            summary = summary[:-1]

        for k in ['that', 'they']:
            if summary.startswith(k):
                summary = summary[len(k):].strip()
        
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

    # remove replicates
    summaries = list(set(summaries))
    return summaries


if __name__ == '__main__':
    llm = get_llm(checkpoint='google/flan-t5-xxl')
    summary = summarize_ngrams(
        llm, ['cat', 'dog', 'bird', 'elephant', 'cheetah'])
    print('summary', repr(summary))
