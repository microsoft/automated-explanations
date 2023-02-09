import re
from typing import Any, List, Mapping, Optional
import numpy as np
import openai
import os.path
from os.path import join
import pickle as pkl
import langchain
from scipy.special import softmax
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
langchain.llm_cache = InMemoryCache()

def get_llm(checkpoint='openai'):
    if checkpoint.startswith('text-da'):
        return llm_openai()
    else:
        return llm_hf(checkpoint)

def llm_openai(checkpoint='text-davinci-003') -> LLM:
    from langchain.llms import OpenAI
    return OpenAI(
        model_name=checkpoint,
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
            _model = T5ForConditionalGeneration.from_pretrained(
                checkpoint, device_map="auto")
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                checkpoint, device_map="auto")

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
            if 'facebook/opt' in checkpoint:
                return out_str[len('</s>') + len(prompt):]
            elif 'google/flan' in checkpoint:
                return out_str[len('<pad>'):out_str.index('</s>')]
            else:
                return out_str

        def _get_logit_for_target_token(self, prompt: str, target_token_str: str) -> float:
            """Get logits for each target token
            This is weird when token_output_ids represents multiple tokens
            It currently will only take the first token
            """
            # Get first token id in target_token_str
            target_token_id = self._tokenizer(target_token_str)['input_ids'][0]

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            logits = self._model(**inputs)['logits'].detach().cpu()  # shape is (batch_size, seq_len, vocab_size)
            probs_next_token = softmax(logits[0, -1, :].numpy().flatten())  # shape is (vocab_size,)
            # token_output_ids = self.tokenizer.convert_tokens_to_ids(target_tokens)
            return probs_next_token[target_token_id]

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




