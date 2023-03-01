import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import trange
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


YES_NO_TOK_IDX = [150, 4273]
MAX_SOURCE_LENGTH = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
MAX_TARGET_LENGTH = 2
TEMPERATURE = 0.001
from typing import Dict
sm = torch.nn.Softmax(dim=-1)


class D5_Validator:
    """Validator class from T5 paper https://github.com/ruiqi-zhong/D5
    """
    def __init__(self, model_path: str = 'ruiqi-zhong/d5_t5_validator', batch_size: int = BATCH_SIZE, verbose: bool = False):
        '''
        model_path is the path to the T5 model weights used for validation
        can also any other model name
        the default is the best model we have trained
        '''
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        print('loading model weights')
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        print('done')
        self.validator_template = '''Check whether the TEXT satisfies a PROPERTY. Respond with Yes or No. When uncertain, output No. 
Now complete the following example -
input: PROPERTY: {hypothesis}
TEXT: {text}
output:'''
        self.batch_size = batch_size
        self.verbose = verbose

    def validate_w_scores(self, input_dicts: List[Dict[str, str]]) -> List[float]:
        '''
        input_dicts is a list of dictionaries, each dictionary has two keys: "explanation" (h) and "text" (x), mapping to the hypothesis and text to be validated
        returns a list of scores, each score is a float between 0 and 1, corresponding to the probability that the hypothesis is true given the text for each input dictionary
        note that it is an iterator, so you can use it in a for loop and save the results whenever some input dictionaries are processed
        '''
        prompts = []
        for i, input_dict in enumerate(input_dicts):
            hypothesis, text = input_dict['explanation'], input_dict['text']
            prompts.append(self.validator_template.format(
                hypothesis=hypothesis, text=text))

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
                                        self.batch_size: (batch_idx + 1) * self.batch_size]
                inputs = self.tokenizer(input_prompts,
                                        return_tensors="pt",
                                        padding="longest",
                                        max_length=MAX_SOURCE_LENGTH,
                                        truncation=True,
                                        ).to(device)
                generation_result = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_TARGET_LENGTH,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                scores = sm(generation_result.scores[0][:, YES_NO_TOK_IDX])[
                    :, 1].detach().cpu().numpy().tolist()
                for s in scores:
                    yield s


if __name__ == '__main__':
    scores_tup_PRF = bert_score.score(['hello world', 'hi world', 'hey world', 'hello worlds'], [
                                      'hello worlds'] * 4, model_type='microsoft/deberta-xlarge-mnli')
    scores = scores_tup_PRF[2].detach().cpu().numpy()
    print(scores)
    # print([score[2].item()
    #   for score in bert_score.score(['hello world', 'hi world'], ['hello worlds'] * 2,
    #    model_type='microsoft/deberta-xlarge-mnli')
    #    ])
