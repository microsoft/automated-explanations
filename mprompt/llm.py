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
import torch
from langchain.cache import InMemoryCache
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, StoppingCriteriaList, MaxLengthCriteria

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
        # opt can't use fast tokenizer
        # https://huggingface.co/docs/transformers/model_doc/opt
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    else:
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


def llm_hf(checkpoint='google/flan-t5-xl') -> LLM:
    class LLM_HF():
        def __init__(self):
            _checkpoint: str = checkpoint
            self._tokenizer = _get_tokenizer(_checkpoint)
            if 'google/flan' in checkpoint:
                self._model = T5ForConditionalGeneration.from_pretrained(
                    checkpoint, device_map="auto", torch_dtype=torch.float16)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    checkpoint, device_map="auto",
                    torch_dtype=torch.float16)

        def __call__(self, prompt: str, stop: Optional[List[str]] = None, max_tokens=1000) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            input_ids = self._tokenizer(
                prompt, return_tensors="pt").input_ids.to("cuda")
            # stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_tokens)])
            # outputs = self._model.generate(input_ids, max_length=max_tokens, stopping_criteria=stopping_criteria)
            outputs = self._model.generate(
                input_ids,
                # max_new_tokens=max_tokens,
                do_sample=True,
            )
            out_str = self._tokenizer.decode(outputs[0])
            if 'facebook/opt' in checkpoint:
                return out_str[len('</s>') + len(prompt):]
            elif 'google/flan' in checkpoint:
                print('full', out_str)
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

            # get prob of target token
            inputs = self._tokenizer(
                prompt, return_tensors="pt").to(self._model.device)
            # shape is (batch_size, seq_len, vocab_size)
            logits = self._model(**inputs)['logits'].detach().cpu()
            # shape is (vocab_size,)
            probs_next_token = softmax(logits[0, -1, :].numpy().flatten())
            # token_output_ids = self._tokenizer.convert_tokens_to_ids(target_tokens)
            # str(self._tokenizer.convert_ids_to_tokens([np.argmax(probs_next_token)])[0])
            return probs_next_token[target_token_id]

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return vars(self)

        @property
        def _llm_type(self) -> str:
            return "custom_hf_llm_for_langchain"

    return LLM_HF()
