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

def get_llm(checkpoint='openai'):
    if checkpoint == 'openai':
        return llm_openai()
    else:
        return llm_hf(checkpoint)

def llm_openai() -> LLM:
    from langchain.llms import OpenAI
    return OpenAI(
        model_name='text-davinci-003',
        max_tokens=100,
        # stop='.',
    )

def llm_hf(checkpoint='google/flan-t5-xl') -> LLM:
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






