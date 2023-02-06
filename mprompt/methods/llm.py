import re
from typing import Any, List, Mapping, Optional
import numpy as np
import openai
import os.path
from os.path import join
import pickle as pkl
import langchain
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache
from transformers import T5Tokenizer, T5ForConditionalGeneration
langchain.llm_cache = InMemoryCache()


def llm_openai() -> LLM:
    from langchain.llms import OpenAI
    return OpenAI(
        model_name='text-davinci-003',
        max_tokens=100,
        # stop='.',
    )


def llm_flan(checkpoint='google/flan-t5-xl') -> LLM:
    class LLM_HF(LLM):
        # langchain forces us to initialize stuff in this kind of weird way
        _checkpoint: str = checkpoint
        _max_tokens = 100
        _tokenizer = T5Tokenizer.from_pretrained(_checkpoint)
        _model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, device_map="auto")

        @property
        def _llm_type(self) -> str:
            return "custom"

        # langchain wants _call instead of __call__
        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            input_ids = self._tokenizer(
                prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self._model.generate(
                input_ids, max_length=self._max_tokens)
            return self._tokenizer.decode(outputs[0])

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return vars(self)

    return LLM_HF()


def summarize_ngrams(
    llm: LLM,
    ngrams_list: List[str],
    prefix_str='Here is a list of phrases:',
    suffix_str='What is a common theme among these phrases?\nThe common theme among these phrases is',
    num_ngrams: int = 40,
    seed: int = 0,
):
    """Refine a keyphrase by making a call to the llm
    """
    bullet_list_ngrams = '- ' + '\n- '.join(ngrams_list[:num_ngrams])
    prompt = prefix_str + '\n\n' + bullet_list_ngrams + '\n\n' + suffix_str
    print(prompt)
    summary = llm(prompt)

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
    llm = llm_flan(checkpoint='google/flan-t5-xxl')
    summary = summarize_ngrams(llm, ['cat', 'dog', 'bird', 'elephant', 'cheetah'])
    print('summary', repr(summary))
