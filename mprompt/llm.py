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
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
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

def _get_tokenizer(checkpoint):
    if 'facebook/opt' in checkpoint:
        # opt can't use fast tokenizer...https://huggingface.co/docs/transformers/model_doc/opt
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    else:
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

def llm_hf(checkpoint='google/flan-t5-xl') -> LLM:
    class LLM_HF(LLM):
        # langchain forces us to initialize stuff in this kind of weird way
        _checkpoint: str = checkpoint
        _max_tokens = 100
        _tokenizer = _get_tokenizer(_checkpoint)
        if 'google/flan' in checkpoint:
            _model = AutoModel.from_pretrained(
                checkpoint, device_map="auto")
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                checkpoint, device_map="auto")
        _offset_decode = len('</s>') if 'facebook/opt' in checkpoint else 0


        @property
        def _llm_type(self) -> str:
            return "custom_hf_llm_for_langchain"

        # langchain wants _call instead of __call__
        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            input_ids = self._tokenizer(
                prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self._model.generate(
                input_ids, max_length=self._max_tokens)
            out_str = self._tokenizer.decode(outputs[0])
            return out_str[self._offset_decode + len(prompt):]

        def get_logit_for_target_token(self, prompt: str, target_token: str) -> float:
            inputs = self._tokenizer(prompt, return_tensors="pt").to('cuda')
            logits = self._model(**inputs)['logits']  # (batch_size, seq_len, vocab_size)
            token_output_id = self._tokenizer.convert_tokens_to_ids(target_token)
            logit_target = logits[0, -1, token_output_id]
            # print(logit_target, 'id', token_output_id)
            token_idx_max = logits[0, -1].argmax()
            # print(f'{token_idx_max=} {self._tokenizer.convert_ids_to_tokens([token_idx_max])=}')
            # print('target_token inner', target_token, token_output_id)
            return logit_target.item()

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return vars(self)
        

    return LLM_HF()


if __name__ == '__main__':
    llm = llm_hf(checkpoint='facebook/opt-125m')
    # print(prompt)
    # print(answer)
    for (prompt, answer) in [
        ('Question: Is a cat an animal?\nAnswer:', ' Yes.'),
        ('Question: Is a dog an animal?\nAnswer:', ' Yes.'),
        ('Question: Is a cat a fruit?\nAnswer:', ' No.'),
        ('Question: Is a dog a fruit?\nAnswer:', ' No.'),
    ]:
        target_token = ' Yes.'
        # answer = llm(prompt) # this is weird, adds some kind of asynchrony or something
        logit_target = llm.get_logit_for_target_token(prompt, target_token=target_token)
        # print(prompt.strip(), answer, f'logit for logit_target: {logit_target:0.2f}')
        print(repr(prompt), logit_target)





