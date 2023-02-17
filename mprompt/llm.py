from transformers import T5Tokenizer, T5ForConditionalGeneration, StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from langchain.cache import InMemoryCache
import re
from typing import Any, List, Mapping, Optional
import numpy as np
import openai
import os.path
from os.path import join, dirname
import os
import pickle as pkl
import langchain
from scipy.special import softmax
import openai
from langchain.llms.base import LLM
import hashlib
import torch
repo_dir = join(dirname(dirname(__file__)))

langchain.llm_cache = InMemoryCache()


def get_llm(checkpoint):
    if checkpoint.startswith('text-da'):
        return llm_openai()
    else:
        return llm_hf(checkpoint)


def llm_openai(checkpoint='text-davinci-003') -> LLM:
    class LLM_OpenAI():
        def __init__(self, checkpoint,
                     cache_dir=join(repo_dir, 'results', 'cache_openai')):
            self.checkpoint = checkpoint
            self.cache_dir = cache_dir

        def __call__(self, prompt: str,
                     max_new_tokens=250, seed=1, do_sample=True):
            # cache
            os.makedirs(self.cache_dir, exist_ok=True)
            hash_str = hashlib.sha256(prompt.encode()).hexdigest()
            cache_file = join(
                self.cache_dir, f'{hash_str}__num_tok={max_new_tokens}__seed={seed}.pkl')
            cache_file_raw = join(
                self.cache_dir, f'raw_{hash_str}__num_tok={max_new_tokens}__seed={seed}.pkl')
            if os.path.exists(cache_file):
                return pkl.load(open(cache_file, 'rb'))

            response = openai.Completion.create(
                engine=self.checkpoint,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=1,
                frequency_penalty=0.25,  # maximum is 2
                presence_penalty=0,
                # stop=["101"]
            )
            response_text = response['choices'][0]['text']

            pkl.dump(response_text, open(cache_file, 'wb'))
            pkl.dump({'prompt': prompt, 'response_text': response_text},
                     open(cache_file_raw, 'wb'))
            return response_text

    return LLM_OpenAI(checkpoint)


def _get_tokenizer(checkpoint):
    if 'facebook/opt' in checkpoint:
        # opt can't use fast tokenizer
        # https://huggingface.co/docs/transformers/model_doc/opt
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    else:
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


def llm_hf(checkpoint='google/flan-t5-xl') -> LLM:
    class LLM_HF():
        def __init__(self, checkpoint):
            _checkpoint: str = checkpoint
            self._tokenizer = _get_tokenizer(_checkpoint)
            if 'google/flan' in checkpoint:
                self._model = T5ForConditionalGeneration.from_pretrained(
                    checkpoint, device_map="auto", torch_dtype=torch.float16)
            elif checkpoint == 'gpt-xl':
                self._model = AutoModelForCausalLM.from_pretrained(checkpoint)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    checkpoint, device_map="auto",
                    torch_dtype=torch.float16)

        def __call__(self, prompt: str, stop: Optional[List[str]] = None,
                     max_new_tokens=20, do_sample=False) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            inputs = self._tokenizer(
                prompt, return_tensors="pt",
                return_attention_mask=True
            ).to(self._model.device)  # .input_ids.to("cuda")
            # stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_tokens)])
            # outputs = self._model.generate(input_ids, max_length=max_tokens, stopping_criteria=stopping_criteria)
            # print('pad_token', self._tokenizer.pad_token)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                # pad_token=self._tokenizer.pad_token,
                pad_token_id=self._tokenizer.pad_token_id,
                # top_p=0.92,
                # top_k=0
            )
            out_str = self._tokenizer.decode(outputs[0])
            if 'facebook/opt' in checkpoint:
                return out_str[len('</s>') + len(prompt):]
            elif 'google/flan' in checkpoint:
                print('full', out_str)
                return out_str[len('<pad>'):out_str.index('</s>')]
            else:
                return out_str[len(prompt):]

        def _get_logit_for_target_token(self, prompt: str, target_token_str: str) -> float:
            """Get logits target_token_str
            This is weird when token_output_ids represents multiple tokens
            It currently will only take the first token
            """
            # Get first token id in target_token_str
            target_token_id = self._tokenizer(target_token_str)['input_ids'][0]

            # get prob of target token
            inputs = self._tokenizer(
                prompt, return_tensors="pt",
                return_attention_mask=True,
                padding=False,
                truncation=False,
            ).to(self._model.device)
            # shape is (batch_size, seq_len, vocab_size)
            logits = self._model(**inputs)['logits'].detach().cpu()
            # shape is (vocab_size,)
            probs_next_token = softmax(logits[0, -1, :].numpy().flatten())
            return probs_next_token[target_token_id]

            # Get first token id in target_token_str
            # target_token_id1 = self._tokenizer(' True.')['input_ids'][0]
            # target_token_id2 = self._tokenizer(' False.')['input_ids'][0]

            # # get prob of target token
            # inputs = self._tokenizer(
            #     prompt, return_tensors="pt",
            #     return_attention_mask=True,
            #     padding=False,
            #     truncation=False,
            # ).to(self._model.device)
            # # shape is (batch_size, seq_len, vocab_size)
            # logits = self._model(**inputs)['logits'].detach().cpu()
            # # shape is (vocab_size,)
            # probs_next_token = softmax(logits[0, -1, :].numpy().flatten())
            # return probs_next_token[target_token_id1] - probs_next_token[target_token_id2]

        @ property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return vars(self)

        @ property
        def _llm_type(self) -> str:
            return "custom_hf_llm_for_langchain"

    return LLM_HF(checkpoint)


if __name__ == '__main__':
    llm = get_llm('text-davinci-003')
    text = llm('What do these have in common? Horse, ')
    print('text', text)
