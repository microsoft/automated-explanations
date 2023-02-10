import re
from typing import Any, List, Mapping, Optional, Tuple
import numpy as np
import openai
import os.path
from os.path import join
import pickle as pkl
from langchain.llms.base import LLM
from mprompt.llm import get_llm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain import PromptTemplate

def generate_synthetic_strs(
    llm: LLM,
    explanation_str: str,
    num_synthetic_strs: int = 20,
) -> Tuple[List[str], List[str]]:
    """Generate text_added and text_removed via call to an LLM.
    Note: might want to pass in a custom text to edit in this function.
    Issue: flan-t5-xxl is only able to generate one sentence before stopping
    """

    template = '''
Generate {num_synthetic_strs} sentences that {blank_or_do_not}contain the concept of "{concept}":

1. The'''
    prompt_template = PromptTemplate(
        input_variables=['num_synthetic_strs', 'blank_or_do_not', 'concept'],
        template=template,
    )    
    strs_added = []
    strs_removed = []
    for blank_or_do_not in ['', 'do not ']:
        prompt = prompt_template.format(
            num_synthetic_strs=num_synthetic_strs,
            blank_or_do_not=blank_or_do_not,
            concept=explanation_str,
        )

        # note: this works works with openai model
        # but tends to stop after generating just one text with non-openai
        # need to fix generation function to generate more...
        synthetic_text_numbered_str = llm(prompt, max_tokens=1000)
        print(synthetic_text_numbered_str)

        # need to parse output for generations here....
        # (split the string s on any numeric character)
        synthetic_strs = re.split(r'\d', synthetic_text_numbered_str)
        print('synthetic_strs=', synthetic_strs)

        # ks = list(set(ks))  # remove duplicates
        # ks = [k.lower() for k in ks if len(k) > 2] # lowercase & len > 2
        # return ks
        # synthetic_str = synthetic_str.strip()
        # ....

        for s in synthetic_strs:
            if blank_or_do_not == '':
                strs_added.append(s)
            else:
                strs_removed.append(s)
    return strs_added, strs_removed



if __name__ == '__main__':
    llm = get_llm(checkpoint='facebook/opt-iml-max-30b')
    strs_added, strs_removed = generate_synthetic_strs(
        llm,
        explanation_str='anger',
        num_synthetic_strs=20)
    print(f'{strs_added=} {strs_removed=}')
