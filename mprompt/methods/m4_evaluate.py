import mprompt.data.data
import bert_score

def compute_score_contains_keywords(args, explanation_strs):
    score_contains_keywords = []
    task_str = mprompt.data.data.get_task_str(args.module_name, args.module_num)
    check_func = mprompt.data.data.get_groundtruth_keywords_check_func(task_str)

    # compute whether each explanation contains any of the synthetic keywords
    for explanation_str in explanation_strs:
        score_contains_keywords.append(check_func(explanation_str))

    return score_contains_keywords

def compute_score_bert(args, explanation_strs):
    # get groundtruth explanation
    task_str = mprompt.data.data.get_task_str(args.module_name, args.module_num)
    keyword_groundtruth = mprompt.data.data.get_groundtruth_keyword(task_str)

    # compute bert score with groundtruth explanation
    scores_tup_PRF = bert_score.score(
        explanation_strs, [keyword_groundtruth] * len(explanation_strs),
        model_type='microsoft/deberta-xlarge-mnli')
    scores_bert = scores_tup_PRF[2].detach().cpu().numpy().tolist()

    return scores_bert


if __name__ == '__main__':
    scores_tup_PRF = bert_score.score(['hello world', 'hi world', 'hey world', 'hello worlds'], ['hello worlds'] * 4, model_type='microsoft/deberta-xlarge-mnli')
    scores = scores_tup_PRF[2].detach().cpu().numpy()
    print(scores)
    # print([score[2].item()
        #   for score in bert_score.score(['hello world', 'hi world'], ['hello worlds'] * 2,
                        #    model_type='microsoft/deberta-xlarge-mnli')
                        #    ])