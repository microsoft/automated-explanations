import re
from typing import Any, List, Mapping, Optional, Tuple
import numpy as np
import openai
import os.path
from os.path import join
import pickle as pkl
from langchain.llms.base import LLM
from mprompt.methods.llm import get_llm
from transformers import T5Tokenizer, T5ForConditionalGeneration


def generate_synthetic_intervention_strs(
    llm: LLM,
    explanation_str: str,
    num_synthetic_strs: int = 2,
    prefix_str_add='Generate text with the given concept:',
    prefix_str_remove='Generate text without the given concept:',
) -> Tuple[List[str], List[str]]:
    """Generate text_added and text_removed via call to an LLM.
    Note: might want to pass in a custom text to edit in this function.
    """
    strs_added = []
    strs_removed = []

    for prefix_str in [prefix_str_add, prefix_str_remove]:
        prompt = prefix_str + '\n\n' + f'Concept:{explanation_str}\n\n' + 'Text:'
        for i in range(num_synthetic_strs):

            synthetic_str = llm(prompt)

            # clean up synthetic string
            synthetic_str = synthetic_str.strip()
            # ....

            if prefix_str == prefix_str_add:
                strs_added.append(synthetic_str)
            else:
                strs_removed.append(synthetic_str)

    return strs_added, strs_removed


if __name__ == '__main__':
    llm = get_llm(checkpoint='google/flan-t5-xl')
    strs_added, strs_removed = generate_synthetic_intervention_strs(
        llm,
        explanation_str='anger',
        num_synthetic_strs=1)
    print(f'{strs_added=} {strs_removed=}')
