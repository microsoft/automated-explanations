import numpy as np
from mprompt.llm import llm_hf
from mprompt.modules.synthetic_groundtruth import SyntheticModule


def test_synthetic(
    checkpoint='facebook/opt-6.7b',
):

    mod = SyntheticModule(
        task_str='animal',
        checkpoint=checkpoint
    )

    INPUTS_POS = ['cat', 'dog', 'giraffe', 'horse']
    INPUTS_NEG = ['apple', 'orange', 'pear', 'three', '4', 'five']
    probs_pos = mod(INPUTS_POS)
    probs_neg = mod(INPUTS_NEG)
    print(probs_pos)
    print(probs_neg)


if __name__ == '__main__':
    test_synthetic()
