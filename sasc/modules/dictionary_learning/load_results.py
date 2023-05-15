import pickle
import csv
from os.path import dirname, join
import os.path

repo_dir = dirname(dirname(os.path.abspath(__file__)))


low_level = [(4, 2), (4, 16), (4, 33), (0, 30), (2, 30), (4, 30)]
mid_level = [(10, 42), (10, 50), (6, 86), (10, 102), (8, 125), (10, 184), (10, 195), (4, 13), (6, 13), (10, 13)]
high_level = [(10, 297), (10, 322), (10, 386)]

low_exp = ['Word “mind”. Noun. the element of a person that enables them to be aware of the world and their experiences.',
           'Word “park”. Noun. a common first and last name.',
           'Word “light”. Noun. the natural agent that stimulates sight and makes things visible.',
           'Word “left". Adjective or Verb. Mixed senses.',
           'Word “left". Verb. leaving, exiting',
           'Word “left". Verb. leaving, exiting']

mid_exp = ['something unfortunate happened.',
           'Doing something again, or making something new again.',
           'Consecutive years, used in foodball season naming.',
           'African names.',
           'Describing someone in a paraphrasing style. Name, Career.',
           'Institution with abbreviation.',
           'Consecutive of noun (Enumerating).',
           'Numerical values.',
           'Parentheses.',
           'Unit exchange with parentheses']

high_exp = ['repetitive structure detector.',
           'biography, someone born in some year...',
           'war']

low_p = ['mar19_dl_l4_i2/268f564b66e8a186cd2f9c2876925b1b753c266429fadd81be54e86b997bfe18',
         'mar20_dl_l4_i16/3a4b18cdb341ebf9bf2e91228aef3a2f4b21e39032eac673e4929e1fd33d1a1c',
         'mar20_dl_l4_i33/694c301c43123ca4788f6ca8b494f82f0d718dbfea71ad50ea0862e0fe5b60ed',
         'mar20_dl_l0_i30/69ada07b0fa60795b132b2e0d73c26fdf6e94c3cf4dbd6c13cdae8eb75251827',
         'mar20_dl_l2_i30/a5d4e81e190b4c7c5249ae985642129e7861f9b80ff4600e054a1b6f8c279215',
         'mar20_dl_l4_i30/814af8345dc0e3d0b8a5a3756c92c023bd1510c2e853924307eaf6cf94f210f6']

mid_p = ['mar20_dl_l10_i42/4a3aee14a9a8a0464fc5e89d16657ddb8a07cb366d848bcc34d969ab40410d2f',
         'mar20_dl_l10_i50/34e4a15483dec6c9e87f087a48d6293c2c7f9ba978e0145c35929fc5e08daa01',
         'mar20_dl_l6_i86/c4cacbee9894aefe4f0fbc099a8ad181f508ece7ca0d70a288b4e4053e3b50ba',
         'mar20_dl_l10_i102/7a4c700acfd72a29c6278a8a401754a0bf2f970b41b0e119715dff82489dcd0e',
         'mar20_dl_l8_i125/e933cdd746cc8776b3a4268b5a700ec0d2601652da7d1231d8f5b199b10aad7b',
         'mar20_dl_l10_i184/606adb5ae6d97287263e32875249cc45b8cee54ee21af949ea04560a0f0d0532',
         'mar20_dl_l10_i195/8d0bf634dba071a6fcc274096a842bb2728fb0fdc12ae76d7e1de9625a67cea2',
         'mar20_dl_l4_i13/c21294ec3f2b6bccb37a1e8585e8009f1522421784a0bc1e481474b0d3e009a2',
         'mar20_dl_l6_i13/01cb8bfbc06d531eabd9e11f67a39c100cfb547ba054a98161435fe474b49818',
         'mar20_dl_l10_i13/16ac8b76300e0b45bef28d20bc6d83e88514839d5ec1f1e58853afd517087291']

high_p = ['mar20_dl_l10_i297/089863daca700285fb97b75fb4f6380a8332a11a1aa5772804dc982dfad9b015',
         'mar20_dl_l10_i322/e02f845df54536abf3499bcc319f59bf693c8b6501136b79f8f742d81a3517e5',
         'mar20_dl_l10_i386/330771ba7e87a1bfd0278644d90e5aa27066fad1f90b2ecdaaa21a4cf36661a1']

def flatten_list(l):
    res = ''
    for i, k in enumerate(l):
        if i != len(l) - 1:
            res += k + ' __ '
        else:
            res += k
    return res

all_idx = low_level + mid_level + high_level
all_ps = low_p + mid_p + high_p
paper_exps = low_exp + mid_exp + high_exp

data = []
for p in all_ps:
    with (open(join(repo_dir, 'results', p, 'results.pkl'), "rb")) as openfile:
        f = pickle.load(openfile)
    data.append(f)

# fill in metainfo:
for i, f in enumerate(data):
    layer_idx, factor_idx = all_idx[i]
    if all_idx[i] in low_level:
        f['level'] = 'low'
    elif all_idx[i] in mid_level:
        f['level'] = 'mid'
    else:
        f['level'] = 'high'

    f['layer'] = layer_idx 
    f['factor_idx'] = factor_idx
    
    f['baseline_explanation'] = paper_exps[i]

# flatten lists:
for f in data:
    f['explanation_init_ngrams'] = flatten_list(f['explanation_init_ngrams'][:25])
    f['explanation_init_rationales'] = flatten_list(f['explanation_init_rationales'])
    f['explanation_init_strs'] = flatten_list(f['explanation_init_strs'])
    f['top_strs_added'] = flatten_list(f['top_strs_added'])
    f['top_strs_removed'] = flatten_list(f['top_strs_removed'])

csv_file = open(join(repo_dir, 'results', 'dict_learn_results.csv'), 'w')
csv_writer = csv.writer(csv_file)
header = ['layer', 'factor_idx', 'level', 'top_score_synthetic', 'baseline_explanation', 'top_explanation_init_strs',
         'explanation_init_ngrams', 'explanation_init_rationales', 'explanation_init_strs']

count = 0
for f in data:
    if count == 0:
        # Writing headers of CSV file
        csv_writer.writerow(header)
        count += 1
 
    # Writing data of CSV file
    filtered = {}
    for k in header:
        filtered[k] = f[k]
    csv_writer.writerow(filtered.values())
 
csv_file.close()

'''
dict_keys(['subsample_frac', 'checkpoint', 'checkpoint_module', 'noise_ngram_scores',
 'module_num_restrict', 'seed', 'save_dir', 'module_name', 'module_num', 'subject',
  'factor_layer', 'factor_idx', 'method_name', 'num_top_ngrams_to_use', 'num_top_ngrams_to_consider',
   'num_summaries', 'num_synthetic_strs', 'generate_template_num', 'use_cache', 'save_dir_unique', 'explanation_init_ngrams',
    'explanation_init_strs', 'explanation_init_rationales', 'strs_added', 'strs_removed', 'score_synthetic', 'top_explanation_init_strs',
     'top_strs_added', 'top_strs_removed', 'top_score_synthetic'])
'''
'''
'explanation_init_ngrams', 'explanation_init_strs', 'explanation_init_rationales', 'top_explanation_init_strs', 'top_strs_added', 'top_strs_removed', 'top_score_synthetic'])
'''