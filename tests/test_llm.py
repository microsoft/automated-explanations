import numpy as np
from mprompt.llm import llm_hf
from langchain import PromptTemplate


def test_llm_hf(
    checkpoint='facebook/opt-125m',
    # checkpoint='facebook/opt-6.7b',
    # checkpoint='EleutherAI/gpt-j-6B',
):
    # load llm
    llm = llm_hf(checkpoint)

    # test prompts
    # LABELS = ['ĠYes', 'ĠNo']
    LABELS = [' False', ' True']

    prompt_template = PromptTemplate(
        input_variables=["input"],
        template='True or False: A {input} is an animal.\nAnswer:',
    )

    INPUTS_AND_LABELS = [
        ('cat', 1),
        ('dog', 1),
        ('banana', 0),
        ('pear', 0)
    ]

    # test generation
    for input, label in INPUTS_AND_LABELS:
        question = prompt_template.format(input=input)
        answer = LABELS[label]
        # prompt = 'Question: Is a cat an animal?\nAnswer:'
        a = llm(question)
        if '.' in a:
            a = a[:a.index('.')]  # only look at up to the period
        if '\n' in a:
            a = a[:a.index('\n')]  # only look at up to the period
        print(repr(question), '->', repr(a))

        # assert answer.lower() in a, f'Should contain the correct answer {repr(answer)} in the generation {repr(a)}'

        # Test synthetic model
        # Note: OPT Tokenizer maps many common tokens to 2 (unknown)
        # e.g. " Yes." -> 2, " No." -> 2
        # e.g. "Yes." -> 2, "No." -> 2
        logits = {}
        print(prompt_template)
        for target_token in LABELS:
            # print('target token', target_token)
            logits[target_token] = llm.get_logit_for_target_token(
                question, target_token=target_token)
        print((logits[LABELS[1]] - logits[LABELS[0]]))
        # assert logits[answer] > np.min(list(logits.values())), 'Correct answer logit should be greater than incorrect'


if __name__ == '__main__':
    test_llm_hf()
