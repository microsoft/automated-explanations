import json

import os
import pandas as pd
from os.path import dirname
from os.path import join as oj

D3_PROCESSED_DIR = oj(
    dirname(os.path.abspath(__file__)), 'd3_processed')
DESCRIPTIONS_DICT = json.load(open(
    oj(D3_PROCESSED_DIR, 'task_defs.json'), 'r')
)


def fetch_data(task_name_induction):
    df = pd.read_csv(oj(D3_PROCESSED_DIR, task_name_induction + '.csv'))
    # Fix input: Encourage model to answer output as next token.
    df['input'] = df['input'].map(lambda s: f'Input: {s} Answer:')
    # Fix output: Prepend a space and add newlines to match output format of number tasks
    df['output'] = df['output'].map(lambda s: f' {s}\n\n')
    return df


TASKS_D3 = {
    'd3_0': {
        'check_func': r'irony|sarcas'
    },

    'd3_1': {
        'check_func': r'objective|factual|nonpersonal|neutral'
    },

    'd3_2': {
        'check_func': r'subjective|opinion|personal|bias'
    },

    'd3_3': {
        'check_func': r'god|religious|religion'
    },

    'd3_4': {
        'check_func': r'atheism|atheist|anti-religion|against religion'
    },

    'd3_5': {
        'check_func': r'evacuate|flee|escape'
    },

    'd3_6': {
        'check_func': r'terorrism|terror'
    },

    'd3_7': {
        'check_func': r'crime|criminal|criminality'
    },

    'd3_8': {
        'check_func': r'shelter|home|house'
    },

    'd3_9': {
        'check_func': r'food|hunger|needs'
    },

    'd3_10': {
        'check_func': r'infrastructure|buildings|roads|bridges|build'
    },

    'd3_11': {
        'check_func': r'regime change|coup|revolution|revolt|political action|political event|upheaval'
    },

    'd3_12': {
        'check_func': r'medical|health'
    },

    'd3_13': {
        'check_func': r'water'
    },

    'd3_14': {
        'check_func': r'search|rescue|help'
    },

    'd3_15': {
        'check_func': r'utility|energy|sanitation|electricity|power'
    },

    'd3_16': {
        'check_func': r'hillary|clinton|against Hillary|opposed to Hillary|republican|against Clinton|opposed to Clinton'
    },

    'd3_17': {
        'check_func': r'hillary|clinton|support Hillary|support Clinton|democrat'
    },

    'd3_18': {
        'check_func': r'offensive|toxic|abusive|insulting|insult|abuse|offend|offend'
    },

    'd3_19': {
        'check_func': r'offensive|toxic|abusive|insulting|insult|abuse|offend|offend|women|immigrants'
    },

    'd3_20': {
        'check_func': r'pro-life|abortion|pro life'
    },

    'd3_21': {
        'check_func': r'pro-choice|abortion|pro choice'
    },

    'd3_22': {
        'check_func': r'physics'
    },

    'd3_23': {
        'check_func': r'computer science|computer|artificial intelligence|ai'
    },

    'd3_24': {
        'check_func': r'statistics|stat|probability'
    },

    'd3_25': {
        'check_func': r'math|arithmetic|algebra|geometry'
    },

    'd3_26': {
        'check_func': r'grammar|syntax|punctuation|grammat'
    },

    'd3_27': {
        'check_func': r'grammar|syntax|punctuation|grammat'
    },

    'd3_28': {
        'check_func': r'sexis|women|femini'
    },

    'd3_29': {
        'check_func': r'sexis|women|femini'
    },

    'd3_30': {
        'check_func': r'news|international|current events'
    },

    'd3_31': {
        'check_func': r'sports'
    },

    'd3_32': {
        'check_func': r'business|economics|finance'
    },

    'd3_33': {
        'check_func': r'tech'
    },

    'd3_34': {
        'check_func': r'bad|negative|awful|terrible|horrible|poor|boring|dislike'
    },

    'd3_35': {
        'check_func': r'good|great|like|love|positive|awesome|amazing|excellent'
    },

    'd3_36': {
        'check_func': r'quantity|number|numeric'
    },

    'd3_37': {
        'check_func': r'location|place'
    },

    'd3_38': {
        'check_func': r'person|group|individual|people'
    },

    'd3_39': {
        'check_func': r'entity|thing|object'
    },

    'd3_40': {
        'check_func': r'abbrevation|abbr|acronym'
    },

    'd3_41': {
        'check_func': r'defin|meaning|explain'
    },

    'd3_42': {
        'check_func': r'environment|climate change|global warming'
    },

    'd3_43': {
        'check_func': r'environment|climate change|global warming'
    },

    'd3_44': {
        'check_func': r'spam|annoying|unwanted'
    },

    'd3_45': {
        'check_func': r'fact|info|knowledge'
    },

    'd3_46': {
        'check_func': r'opinion|personal|bias'
    },

    'd3_47': {
        'check_func': r'math|science'
    },

    'd3_48': {
        'check_func': r'health|medical|disease'
    },

    'd3_49': {
        'check_func': r'computer|internet|web'
    },

    'd3_50': {
        'check_func': r'sport'
    },

    'd3_51': {
        'check_func': r'entertainment|music|movie|tv'
    },

    'd3_52': {
        'check_func': r'family|relationships'
    },

    'd3_53': {
        'check_func': r'politic|government|law'
    },
}
ks = list(TASKS_D3.keys())
for k in ks:
    if not k == 'SUFFIXES':
        TASKS_D3[k]['description'] = DESCRIPTIONS_DICT[k]
        TASKS_D3[k]['gen_func'] = fetch_data

if __name__ == '__main__':
    print(TASKS_D3)
