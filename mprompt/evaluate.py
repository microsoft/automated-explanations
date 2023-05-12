import numpy as np
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm, trange
import torch.nn
from typing import List
import mprompt.data.data
import bert_score


def compute_score_contains_keywords(args, explanation_strs):
    score_contains_keywords = []
    task_str = mprompt.data.data.get_task_str(
        args.module_name, args.module_num)
    check_func = mprompt.data.data.get_groundtruth_keywords_check_func(
        task_str)

    # compute whether each explanation contains any of the synthetic keywords
    for explanation_str in explanation_strs:
        score_contains_keywords.append(check_func(explanation_str))

    return score_contains_keywords

def compute_score_bert(args, explanation_strs):
    # get groundtruth explanation
    task_str = mprompt.data.data.get_task_str(
        args.module_name, args.module_num)
    keyword_groundtruth = mprompt.data.data.get_groundtruth_keyword(task_str)

    # compute bert score with groundtruth explanation
    scores_tup_PRF = bert_score.score(
        explanation_strs, [keyword_groundtruth] * len(explanation_strs),
        model_type='microsoft/deberta-xlarge-mnli')
    scores_bert = scores_tup_PRF[2].detach().cpu().numpy().tolist()

    return scores_bert

def test_ngrams_bert_score(explanation: str, top_ngrams_test: List[str]):
    n = len(top_ngrams_test)
    P, R, F1 = bert_score.score(
        [explanation] * n,
        top_ngrams_test,
        model_type='microsoft/deberta-xlarge-mnli'
    )
    bscore = F1.detach().cpu().numpy().mean()
    return bscore

class D5_Validator:
    """Validator class from D5 paper https://github.com/ruiqi-zhong/D5
    """
    def __init__(self, model_path: str = 'ruiqi-zhong/d5_t5_validator', batch_size: int = 32, verbose: bool = False):
        '''
        model_path is the path to the T5 model weights used for validation
        can also any other model name
        the default is the best model we have trained
        '''
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        print('loading model weights')
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path, device_map='auto', torch_dtype=torch.float16) #.to(device)
        print('done')
        self.validator_template = '''Check whether the TEXT satisfies a PROPERTY
. Respond with Yes or No. When uncertain, output No. 
Now complete the following example -
input: PROPERTY: {hypothesis}
TEXT: {text}
output:'''
        self.batch_size = batch_size
        self.verbose = verbose

    def validate_w_scores(self, explanation: str, top_ngrams_test: List[str]) -> List[float]:
        '''
        Returns
        -------
        list of scores
            each score is a float between 0 and 1
            probability that the explanation is true given the text for each input
        '''
        
        prompts = []
        for top_ngram in top_ngrams_test:
            prompts.append(self.validator_template.format(
                hypothesis=explanation, text=top_ngram))
        # print('prompts', prompts)

        # return prompts
        all_scores = []
        with torch.no_grad():
            self.model.eval()
            num_batches = (len(prompts) - 1) // self.batch_size + 1
            if self.verbose:
                pbar = trange(num_batches)
                pbar.set_description('inference')
            else:
                pbar = range(num_batches)

            for batch_idx in pbar:
                input_prompts = prompts[batch_idx *
                                        self.batch_size: (batch_idx + 1) * self.
batch_size]
                inputs = self.tokenizer(input_prompts,
                                        return_tensors="pt",
                                        padding="longest",
                                        max_length=1024,
                                        truncation=True,
                                        ).to(self.model.device)
                generation_result = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=2,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                YES_NO_TOK_IDX = [150, 4273]
                scores = self.softmax(generation_result.scores[0][:, YES_NO_TOK_IDX])[
                    :, 1].detach().cpu().numpy().tolist()
                all_scores = all_scores + scores
            return all_scores

def calc_frac_correct_score(r: pd.DataFrame, col_explanation='top_explanation_init_strs', col_ngrams='top_ngrams_test_100'):
    """Note: validator takes a lot of memory.
    """
    validator = D5_Validator()
    test_correct_score_list = []
    correct_ngrams_test_list = []
    for i in tqdm(range(r.shape[0])):
        # get expl and ngrams
        row = r.iloc[i]
        explanation = row[col_explanation]
        assert isinstance(explanation, str), explanation
        ngrams = row[col_ngrams]
        if isinstance(ngrams, list):
            ngrams = np.array(ngrams)
        assert isinstance(ngrams, np.ndarray), ngrams

        scores = validator.validate_w_scores(explanation, ngrams.tolist())
        test_correct_score_list.append(np.mean(scores))
        correct_ngrams_test_list.append(ngrams[np.array(scores) > 0.5])
    return test_correct_score_list, correct_ngrams_test_list

if __name__ == '__main__':
    # scores_tup_PRF = bert_score.score(['hello world', 'hi world', 'hey world', 'hello worlds'], [
                                    #   'hello worlds'] * 4, model_type='microsoft/deberta-xlarge-mnli')
    # scores = scores_tup_PRF[2].detach().cpu().numpy()
    # print(scores)
    # print([score[2].item()
    #   for score in bert_score.score(['hello world', 'hi world'], ['hello worlds'] * 2,
    #    model_type='microsoft/deberta-xlarge-mnli')
    #    ])
    validator = D5_Validator()