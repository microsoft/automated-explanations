import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join, dirname
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle as pkl
import imodelsx
import ridge_utils as huth
import torch
from tqdm import tqdm
import sasc.modules.old_fmri_module
import sasc.modules.fmri_module
import sasc.modules.prompted_module
import sasc.modules.emb_diff_module
import sasc.modules.dictionary_module
import imodelsx.sasc.m1_ngrams
import imodelsx.sasc.m2_summarize
import imodelsx.sasc.m3_generate
import sasc.evaluate
import sasc.data.data
from sasc.config import CACHE_DIR
from sasc.data.data import TASKS_D3, TASKS_TOY
from imodelsx import cache_save_utils
import imodelsx.llm


# load module to interpret (do this outside the loop to speed up loading the module)
def load_module(args, module_num):
    if args.module_name == "fmri":
        mod = sasc.modules.fmri_module.fMRIModule(
            voxel_num_best=module_num,
            subject=args.subject,
            checkpoint=args.checkpoint_module,
        )
    elif args.module_name == "old_fmri":
        mod = sasc.modules.old_fmri_module.OldFMRIModule(
            voxel_num_best=module_num)
    elif args.module_name == "dict_learn_factor":
        mod = sasc.modules.dictionary_module.DictionaryModule(
            layer_idx=args.factor_layer, factor_idx=args.factor_idx, task=args.dl_task
        )
    else:
        task_str = sasc.data.data.get_task_str(args.module_name, module_num)
        logging.info("running " + task_str)
        if args.module_name.startswith("prompted"):
            mod = sasc.modules.prompted_module.PromptedModule(
                task_str=task_str,
                checkpoint=args.checkpoint_module,
            )
        elif args.module_name.startswith("emb_diff"):
            mod = sasc.modules.emb_diff_module.EmbDiffModule(
                task_str=task_str,
                # checkpoint=args.checkpoint_module,
            )
    return mod


def _get_cache_filename(args, CACHE_DIR) -> str:
    if args.module_name == "fmri":
        suffix = ""
        if args.checkpoint_module == "decapoda-research/llama-30b-hf":
            suffix = "_llama"
        return join(
            CACHE_DIR, "cache_ngrams", f"{args.module_name}_{args.subject}{suffix}.pkl"
        )
    elif args.module_name == "old_fmri":
        return join(CACHE_DIR, "cache_ngrams", f"{args.module_name}.pkl")
    elif args.module_name == "dict_learn_factor":
        return join(
            CACHE_DIR,
            "cache_ngrams",
            f"{args.module_name}_{args.dl_task}_l{args.factor_layer}_i{args.factor_idx}.pkl",
        )
    else:
        return join(
            CACHE_DIR, "cache_ngrams", f"{args.module_name}_{args.module_num}.pkl"
        )


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--subsample_frac", type=float, default=1, help="fraction of samples to use"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        # default='google/flan-t5-xxl',
        default="text-davinci-003",
        help="which llm to use for both summarization and generation",
    )
    parser.add_argument(
        "--checkpoint_module",
        type=str,
        default="gpt2-xl",
        help="which llm to use for the module (if synthetic or fmri)",
    )

    # data ablations
    parser.add_argument(
        "--noise_ngram_scores",
        type=float,
        default=0,
        help="""ablation: how much noise to add to ngram scores
                        (noise stddev = noise_ngram_scores * stddev(top-100 ngram responses)""",
    )
    parser.add_argument(
        "--module_num_restrict",
        type=int,
        default=-1,
        help="""ablation: alternative module_num to specify a corpus to restrict the ngram responses""",
    )

    # training misc args
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir", type=str, default="results/tmp", help="directory for saving"
    )

    # module args
    parser.add_argument(
        "--module_name",
        type=str,
        default="emb_diff_toy",
        help="name of module",
        choices=[
            "fmri",
            "old_fmri",
            "emb_diff_d3",
            "emb_diff_toy",
            "prompted_d3",
            "prompted_toy",
            "dict_learn_factor",
        ],
    )
    parser.add_argument(
        "--module_num",
        type=str,  # good task is d3_13_water or d3_16_hillary
        default="all",
        choices=["all"] + [str(i) for i in range(500)],
        help="number of module to select, 'all' for all modules ('all' only currently supported for fmri)",
    )
    parser.add_argument(
        "--subject", type=str, default="UTS03", help="for fMRI, which subject to use"
    )
    parser.add_argument(
        "--factor_layer",
        type=int,
        default=4,
        help="for dictionary learning module, which layer of factor to use",
    )
    parser.add_argument(
        "--factor_idx",
        type=int,
        default=2,
        help="for dictionary learning module, which index of factor in the layer to use",
    )
    parser.add_argument(
        "--dl_task",
        type=str,
        default="wiki",
        help="for dictionary learning module, which task dataset to use",
    )

    # algo args
    parser.add_argument(
        "--num_top_ngrams_to_use",
        type=int,
        default=3,
        help="number of ngrams to use to start the explanation",
    )
    parser.add_argument(
        "--num_top_ngrams_to_consider",
        type=int,
        default=5,
        help="select top ngrams from this many",
    )
    parser.add_argument(
        "--num_summaries", type=int, default=2, help="number of summaries to start with"
    )
    parser.add_argument(
        "--num_synthetic_strs",
        type=int,
        default=3,
        help="number of synthetic strings to generate",
    )
    parser.add_argument(
        "--generate_template_num",
        type=int,
        default=1,
        help="which template to use for generation",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        choices=["ngrams", "gradient"],
        default="ngrams",
        help="(Deprecated) name of algo for explanation",
    )
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldn't use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to check for cache",
    )
    return parser


if __name__ == "__main__":
    # get args
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()
    if not args.module_num == "all":
        assert (
            not int(args.module_num) == args.module_num_restrict
        ), "module_num and module_num_restrict should be different"

    # set up module_nums
    if args.module_num == "all":
        if args.module_name == "fmri":
            module_nums = range(500)
            mod = load_module(
                args, module_nums[0]
            )  # load module before checking cache in this case
        else:
            raise ValueError("only fmri currently supports 'all' module_num")
    else:
        module_nums = [int(args.module_num)]

    # loop over module_nums
    for i in tqdm(range(len(module_nums))):
        if len(module_nums) > 1:
            logging.info(f"\n\n{i + 1} / {len(module_nums)}")
        args.module_num = module_nums[i]  # an integer

        # set up saving directory + check for cache
        already_cached, save_dir_unique = cache_save_utils.get_save_dir_unique(
            parser, parser_without_computational_args, args, args.save_dir
        )

        if args.use_cache and already_cached:
            logging.info(
                f"cached version exists! Successfully skipping :)\n\n\n")
            continue
        for k in sorted(vars(args)):
            logging.info("\t" + k + " " + str(vars(args)[k]))
        logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

        # set seed
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # set up saving dictionary + save params file
        r = defaultdict(list)
        r.update(vars(args))
        r["save_dir_unique"] = save_dir_unique
        cache_save_utils.save_json(
            args=args, save_dir=save_dir_unique, fname="params.json", r=r
        )

        # deal with module
        if i == 0 and not args.module_name == "all":
            mod = load_module(args, args.module_num)
        if args.module_name in ["fmri", "old_fmri"]:
            mod._init_fmri_voxel(args.module_num, args.subject)
            r["fmri_test_corr"] = mod.corr

        # load text data
        text_str_list = sasc.data.data.get_relevant_data(
            args.module_name, args.module_num, args.subject, args.dl_task
        )

        # subsample data
        if args.subsample_frac < 1:  # and not args.module_name == 'fmri':
            assert False, "dont subsample data, since explain_ngrams is using caching"
            # n_subsample = int(len(text_str_list) * args.subsample_frac)

            # randomly subsample list
            # text_str_list, size=n_subsample, replace=False).tolist()

        # explain with method
        cache_filename = _get_cache_filename(args, CACHE_DIR)
        if args.module_num_restrict >= 0:
            text_str_list_restrict = sasc.data.data.get_relevant_data(
                args.module_name, args.module_num_restrict
            )
        else:
            text_str_list_restrict = None
        (
            explanation_init_ngrams,
            explanation_init_scores,
        ) = imodelsx.sasc.m1_ngrams.explain_ngrams(
            text_str_list=text_str_list,
            mod=mod,
            num_top_ngrams=75,
            cache_filename=cache_filename,
            noise_ngram_scores=args.noise_ngram_scores,
            noise_seed=args.seed,
            text_str_list_restrict=text_str_list_restrict,
        )
        r["explanation_init_ngrams"] = explanation_init_ngrams
        r["explanation_init_outputs"] = explanation_init_scores
        logging.info(
            f"{explanation_init_ngrams[:3]=} {len(explanation_init_ngrams)}")

        pkl.dump(r, open(join(save_dir_unique, "results.pkl"), "wb"))

        # # summarize the ngrams into some candidate strings
        # imodelsx.llm.LLM_CONFIG["LLM_REPEAT_DELAY"] = 2
        # llm = imodelsx.llm.get_llm(
        #     args.checkpoint, CACHE_DIR=join(CACHE_DIR, args.checkpoint)
        # )
        # (
        #     explanation_strs,
        #     explanation_rationales,
        # ) = imodelsx.sasc.m2_summarize.summarize_ngrams(
        #     llm,
        #     explanation_init_ngrams,
        #     num_summaries=args.num_summaries,
        #     num_top_ngrams_to_use=args.num_top_ngrams_to_use,
        #     num_top_ngrams_to_consider=args.num_top_ngrams_to_consider,
        #     seed=args.seed,
        # )
        # r["explanation_init_strs"] = explanation_strs
        # r["explanation_init_rationales"] = explanation_rationales
        # logging.info("explanation_init_strs\n\t" + "\n\t".join(explanation_strs))

        # # generate synthetic data
        # logging.info("\n\nGenerating synthetic data....")
        # for explanation_str in explanation_strs:
        #     (
        #         strs_added,
        #         strs_removed,
        #     ) = imodelsx.sasc.m3_generate.generate_synthetic_strs(
        #         llm,
        #         explanation_str=explanation_str,
        #         num_synthetic_strs=args.num_synthetic_strs,
        #         template_num=args.generate_template_num,
        #     )
        #     r["strs_added"].append(strs_added)
        #     r["strs_removed"].append(strs_removed)

        #     # evaluate synthetic data (higher score is better)
        #     r["score_synthetic"].append(
        #         np.mean(mod(strs_added)) - np.mean(mod(strs_removed))
        #     )

        # logging.info(
        #     f"{explanation_strs[0]}\n+++++++++\n\t"
        #     + "\n\t".join(r["strs_added"][0][:3])
        #     + "\n--------\n\t"
        #     + "\n\t".join(r["strs_removed"][0][:3])
        # )

        # # sort everything by score
        # sort_inds = np.argsort(r["score_synthetic"])[::-1]
        # for k in [
        #     "explanation_init_strs",
        #     "strs_added",
        #     "strs_removed",
        #     "score_synthetic",
        # ]:
        #     r[k] = [r[k][i] for i in sort_inds]
        #     r["top_" + k] = r[k][0]

        # # evaluate how well explanation matches a "groundtruth"
        # if not (args.module_name == "fmri" or args.module_name == "dict_learn_factor"):
        #     logging.info("Scoring explanation....")
        #     r[
        #         "score_contains_keywords"
        #     ] = sasc.evaluate.compute_score_contains_keywords(
        #         args, r["explanation_init_strs"]
        #     )

        # # save results
        # pkl.dump(r, open(join(save_dir_unique, "results.pkl"), "wb"))
        # logging.info("Succesfully completed :)\n\n")
