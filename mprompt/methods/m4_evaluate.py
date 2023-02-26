import mprompt.data.data

def compute_recovery_metrics(args, explanation_strs):
    score_contains_keywords = []
    # get groundtruth explanation
    task_str = mprompt.data.data.get_task_str(args.module_name, args.module_num)
    explanation_groundtruth = mprompt.data.data.get_groundtruth_explanation(task_str)
    check_func = mprompt.data.data.get_groundtruth_keywords_check_func(task_str)

    for explanation_str in explanation_strs:
        # compute bleu score with groundtruth explanation
        # r['score_bleu'].append(
        # calc_bleu_score(explanation_groundtruth, explanation_str))

        # compute whether explanation contains any of the synthetic keywords
        score_contains_keywords.append(check_func(explanation_str))
    return score_contains_keywords