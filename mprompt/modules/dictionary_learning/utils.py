import pickle as pkl
from os.path import dirname, join
import os.path
from typing import List
import json

low_p = ['dl_l4_i2/268f564b66e8a186cd2f9c2876925b1b753c266429fadd81be54e86b997bfe18',
        'dl_l4_i16/3a4b18cdb341ebf9bf2e91228aef3a2f4b21e39032eac673e4929e1fd33d1a1c',
        'dl_l4_i33/694c301c43123ca4788f6ca8b494f82f0d718dbfea71ad50ea0862e0fe5b60ed',
        #'dl_l0_i30/69ada07b0fa60795b132b2e0d73c26fdf6e94c3cf4dbd6c13cdae8eb75251827',
        #'dl_l2_i30/a5d4e81e190b4c7c5249ae985642129e7861f9b80ff4600e054a1b6f8c279215',
        'dl_l4_i30/814af8345dc0e3d0b8a5a3756c92c023bd1510c2e853924307eaf6cf94f210f6']

mid_p = ['dl_l10_i42/4a3aee14a9a8a0464fc5e89d16657ddb8a07cb366d848bcc34d969ab40410d2f',
        'dl_l10_i50/34e4a15483dec6c9e87f087a48d6293c2c7f9ba978e0145c35929fc5e08daa01',
        'dl_l6_i86/c4cacbee9894aefe4f0fbc099a8ad181f508ece7ca0d70a288b4e4053e3b50ba',
        'dl_l10_i102/7a4c700acfd72a29c6278a8a401754a0bf2f970b41b0e119715dff82489dcd0e',
        'dl_l8_i125/e933cdd746cc8776b3a4268b5a700ec0d2601652da7d1231d8f5b199b10aad7b',
        'dl_l10_i184/606adb5ae6d97287263e32875249cc45b8cee54ee21af949ea04560a0f0d0532',
        'dl_l10_i195/8d0bf634dba071a6fcc274096a842bb2728fb0fdc12ae76d7e1de9625a67cea2',
        #'dl_l4_i13/c21294ec3f2b6bccb37a1e8585e8009f1522421784a0bc1e481474b0d3e009a2',
        #'dl_l6_i13/01cb8bfbc06d531eabd9e11f67a39c100cfb547ba054a98161435fe474b49818',
        'dl_l10_i13/16ac8b76300e0b45bef28d20bc6d83e88514839d5ec1f1e58853afd517087291']

high_p = ['dl_l10_i297/089863daca700285fb97b75fb4f6380a8332a11a1aa5772804dc982dfad9b015',
        'dl_l10_i322/e02f845df54536abf3499bcc319f59bf693c8b6501136b79f8f742d81a3517e5',
        'dl_l10_i386/330771ba7e87a1bfd0278644d90e5aa27066fad1f90b2ecdaaa21a4cf36661a1']

low_level = [(4, 2), (4, 16), (4, 33), (0, 30), (2, 30),
             (4, 30), (4, 47)]
mid_level = [(10, 42), (10, 50), (6, 86), (10, 102), (8, 125),
             (10, 184), (10, 195), (4, 13), (6, 13), (10, 13),
             (10, 24), (10, 25), (10, 51), (10, 86), (10, 99),
             (10, 125), (10, 134), (10, 152), (4, 193), (6, 225)]
high_level = [(10, 297), (10, 322), (10, 386), (10, 179)]

low_exp = ['mind. Noun. the element of a person that enables them to be aware of the world and their experiences.',
           'park. Noun. a common first and last name.',
           'light. Noun. the natural agent that stimulates sight and makes things visible.',
           'left. Adjective or Verb. Mixed senses.',
           'left. Verb. leaving, exiting.',
           'left. Verb. leaving, exiting.',
           'plants. Noun. vegetation.']

mid_exp = ['Something unfortunate happened.',
           'Doing something again, or making something new again.',
           'Consecutive years, used in foodball season naming.',
           'African names.',
           'Describing someone in a paraphrasing style. Name, Career.',
           'Institution with abbreviation.',
           'Consecutive of noun (Enumerating).',
           'Numerical values.',
           'Close Parentheses.',
           'Unit exchange with parentheses.',
           'Male name.',
           'Attributive Clauses.',
           'Apostrophe s, possesive.',
           'Consecutive years, this is convention to name foodball/rugby game season.',
           'Past tense.',
           'Describing someone in a para- phrasing style. Name, Career.',
           'Transition sentence.',
           'In some locations.',
           'Time span in years.',
           'Places in US, followings the convention "city, state".']

high_exp = ['Repetitive structure detector.',
            'Biography, someone born in some year...',
            'War.',
            'Topic: music production.']

cur_dir = dirname(os.path.abspath(__file__))
SAVE_DIR_DICT = cur_dir
    
    
def batch_up(iterable, batch_size=1):
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx:min(ndx + batch_size, l)]

def get_exp_data(factor_idx: int, factor_layer: int) -> List[str]:

    all_ps = low_p + mid_p + high_p
    RESULT_DIR = join('/'.join(os.path.dirname(cur_dir).split('/')[:-2]), 'results')

    exps = None
    for p in all_ps:
        seg = p.split('/')
        key = f'dl_l{factor_layer}_i{factor_idx}'
        if seg[0] == key:
            with (open(join(RESULT_DIR, p, 'results.pkl'), "rb")) as openfile:
                f = pkl.load(openfile)
            exps = f['explanation_init_strs']
            break

    control_strs = {}
    control_strs['strs_added'] = []
    control_strs['strs_removed'] = []
    # load control data
    for p in all_ps:
        seg = p.split('/')
        sub_seg = seg[0].split('_')
        key = f'i{factor_idx}'
        # collect data from all the other factor
        if sub_seg[-1] != key:
            with (open(join(RESULT_DIR, p, 'results.pkl'), "rb")) as openfile:
                f = pkl.load(openfile)
            control_strs['strs_added'].extend(f['top_strs_added'])
            control_strs['strs_removed'].extend(f['top_strs_removed'])

    # load baseline explanations
    with open(join(cur_dir, 'baseline_exp.json'), 'r') as fp:
        baseline_exp = json.load(fp)
    for k, d in baseline_exp.items():
        if d['layer'] == factor_layer and d['factor_idx'] == factor_idx:
            exps.extend([d['exp']])
            break
    
    assert(exps != None)

    return exps, control_strs

def get_baseline_data(factor_idx, factor_layer) -> List[str]:
    exps = []
    # load baseline explanations
    with open(join(cur_dir, 'baseline_exp.json'), 'r') as fp:
        baseline_exp = json.load(fp)
    for k, d in baseline_exp.items():
        if d['factor_idx'] == factor_idx and d['layer'] == factor_layer:
            exps.append(d['exp'])
            break

    return exps, {}

def write_baseline_exp():
    r = {}
    exps = low_exp + mid_exp + high_exp
    levels = low_level + mid_level + high_level

    for i, (l, pos) in enumerate(levels):
        r[i] = {}
        r[i]['layer'] = l
        r[i]['factor_idx'] = pos
        r[i]['exp'] = exps[i]
    
    with open(join(cur_dir, 'baseline_exp.json'), 'w') as fp:
        json.dump(r, fp)
        
        
        
if __name__ == '__main__':
    
    # save baseline explanation file to dir
    write_baseline_exp()
    #exp, data = get_exp_data(2, 4)
    #save_sst2_unique_ngram_list()

