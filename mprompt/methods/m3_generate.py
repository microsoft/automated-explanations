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
    num_synthetic_strs: int = 2,
    prefix_str_add='Generate 20 sentences that contain the concept of',
    prefix_str_remove='Generate 20 sentences that contain the concept of',
) -> Tuple[List[str], List[str]]:
    """Generate text_added and text_removed via call to an LLM.
    Note: might want to pass in a custom text to edit in this function.
    """
    template = '''
Generate {num_synthetic_strs} senteces that {blank_or_do_not}contain the concept of "{concept}":

1.'
'''

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

        synthetic_text_numbered_str = llm(prompt)
        print(synthetic_text_numbered_str)

        # need to parse output for generations here....
        # (split the string s on any numeric character)
        ks = [
            k.replace('.', '').replace('"', '').strip()
            for k in re.split(r'\d', synthetic_text_numbered_str) if k.strip()
        ]

        # ks = list(set(ks))  # remove duplicates
        # ks = [k.lower() for k in ks if len(k) > 2] # lowercase & len > 2
        # return ks
        # synthetic_str = synthetic_str.strip()
        # ....

        if blank_or_do_not == '':
            strs_added.append(synthetic_str)
        else:
            strs_removed.append(synthetic_str)

    return strs_added, strs_removed



if __name__ == '__main__':
    llm = get_llm(checkpoint='google/flan-t5-xxl')
    strs_added, strs_removed = generate_synthetic_strs(
        llm,
        explanation_str='anger',
        num_synthetic_strs=1)
    print(f'{strs_added=} {strs_removed=}')
