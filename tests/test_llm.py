import numpy as np
from mprompt.llm import llm_hf


def test_llm_hf():
    # load llm
    llm = llm_hf(checkpoint='facebook/opt-6.7b')
    # llm = llm_hf(checkpoint='facebook/opt-125m')

    # test prompts
    prompt_questions = [
        ('Question: Is a cat an animal?\nAnswer: ', 'Yes'),
        ('Question: Is a dog an animal?\nAnswer: ', 'Yes'),
        ('Question: Is a cat a fruit?\nAnswer: ', 'No'),
        ('Question: Is a dog a fruit?\nAnswer: ', 'No'),
    ]

    # test generation
    for (prompt, answer) in prompt_questions:
        # prompt = 'Question: Is a cat an animal?\nAnswer:'
        answer = llm(prompt) # this is weird, adds some kind of asynchrony or something
        # print('answer', answer)
        if answer == 'Yes':
            assert 'yes' in answer.lower()
        elif answer == 'No':
            assert 'no' in answer.lower()


    # Test synthetic model
    # Note: OPT Tokenizer maps many common tokens to 2 (unknown)
    # e.g. " Yes." -> 2, " No." -> 2
    # e.g. "Yes." -> 2, "No." -> 2
    for (prompt, answer) in prompt_questions:
        logits = {}
        print(prompt)
        for target_token in ['Yes', 'No']:
            print('target token', target_token)
            logits[target_token] = llm.get_logit_for_target_token(
                prompt, target_token=target_token)
        print(logits, logits.values())
        assert logits[answer] > np.min(list(logits.values())), 'Correct answer logit should be greater than incorrect'


if __name__ == '__main__':
    test_llm_hf()