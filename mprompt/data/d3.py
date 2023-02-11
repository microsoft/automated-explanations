from functools import partial
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
    df = pd.read_csv(oj(D3_PROCESSED_DIR,
                        task_name_induction[:task_name_induction.rindex('_')] + '.csv'))
    # Fix input: Encourage model to answer output as next token.
    # df['input'] = df['input'].map(lambda s: f'Input: {s} Answer:')
    # Fix output: Prepend a space and add newlines to match output format of number tasks
    # df['output'] = df['output'].map(lambda s: f' {s}\n\n')
    return df


TASKS_D3 = {
    'd3_0_irony': {
        'check_func': 'irony|sarcas',
        'groundtruth_explanation': 'contains irony',
        'template': 'The phrase "{input}" contains',
        'target_token': ' sarcasm',
        'examples': ["that's just great.", "thanks a lot.", "how original", "how convenient"],
    },
    'd3_1_objective': {
        'check_func': 'objective|factual|nonpersonal|neutral',
        'groundtruth_explanation': 'is a more objective description of what happened',
        'template': 'The phrase "{input}" is',
        'target_token': ' objective',
        'examples': ["The temperature is 70 degrees.", "The movie is 2 hours long.", "The book is 300 pages long.", "The flight is at 7 PM.", "The concert is at 8 PM.", "The hike is 4 hours long.", "The team scored 3 goals.", "The building is 10 stories tall.", "The package weighs 5 pounds.", "The meeting is at 10 AM."],
    },
    'd3_2_subjective': {
        'check_func': 'subjective|opinion|personal|bias',
        'groundtruth_explanation': 'contains subjective opinion',
        'template': 'The phrase "{input}" is',
        'target_token': ' subjective',
        'examples': ["The temperature is hot.", "The movie is great.", "The book is interesting.", "The music is annoying."],
    },
    'd3_3_god': {
        'check_func': 'god|religious|religion',
        'groundtruth_explanation': 'believes in god',
        'template': 'The phrase "{input}" is',
        'target_token': ' religious',
        'examples': ["Praise the Lord.", "Hallelujah.", "Amen.", "Glory to God."],
    },
    'd3_4_atheism': {
        'check_func': 'atheism|atheist|anti-religion|against religion',
        'groundtruth_explanation': 'is against religion',
        'template': 'The phrase "{input}" is',
        'target_token': ' atheistic',
        'examples': ["secular humanist", "atheist", "agnostic", "anti-religion"],
    },
    'd3_5_evacuate': {
        'check_func': 'evacuate|flee|escape',
        'groundtruth_explanation': 'involves a need for people to evacuate',
        'template': 'The phrase "{input}" involves',
        'target_token': ' evacuation',
        'examples': ["evacuate", "flee", "escape", "leave", "humanitarian crisis"],
    },
    'd3_6_terorrism': {
        'check_func': 'terorrism|terror',
        'groundtruth_explanation': 'describes a situation that involves terrorism',
        'template': 'The phrase "{input}" involves',
        'target_token': ' terrorism',

    },
    'd3_7_crime': {
        'check_func': 'crime|criminal|criminality',
        'groundtruth_explanation': 'involves crime'
    },
    'd3_8_shelter': {
        'check_func': 'shelter|home|house',
        'groundtruth_explanation': 'describes a situation where people need shelter'
    },
    'd3_9_food': {
        'check_func': 'food|hunger|needs',
        'groundtruth_explanation': 'is related to food security'
    },
    'd3_10_infrastructure': {
        'check_func': 'infrastructure|buildings|roads|bridges|build',
        'groundtruth_explanation': 'is related to infrastructure'
    },
    'd3_11_regime change': {
        'check_func': 'regime change|coup|revolution|revolt|political action|political event|upheaval',
        'groundtruth_explanation': 'describes a regime change'
    },
    'd3_12_medical': {
        'check_func': 'medical|health',
        'groundtruth_explanation': 'is related to a medical situation'
    },
    'd3_13_water': {
        'check_func': 'water',
        'groundtruth_explanation': 'involves a situation where people need clean water'
    },
    'd3_14_search': {
        'check_func': 'search|rescue|help',
        'groundtruth_explanation': 'involves a search/rescue situation'
    },
    'd3_15_utility': {
        'check_func': 'utility|energy|sanitation|electricity|power',
        'groundtruth_explanation': 'expresses need for utility, energy or sanitation'
    },
    'd3_16_hillary': {
        'check_func': 'hillary|clinton|against Hillary|opposed to Hillary|republican|against Clinton|opposed to Clinton',
        'groundtruth_explanation': 'is against Hillary'
    },
    'd3_17_hillary': {
        'check_func': 'hillary|clinton|support Hillary|support Clinton|democrat',
        'groundtruth_explanation': 'supports hillary'
    },
    'd3_18_offensive': {
        'check_func': 'offensive|toxic|abusive|insulting|insult|abuse|offend|offend',
        'groundtruth_explanation': 'contains offensive content'
    },
    'd3_19_offensive': {
        'check_func': 'offensive|toxic|abusive|insulting|insult|abuse|offend|offend|women|immigrants',
        'groundtruth_explanation': 'insult women or immigrants'
    },
    'd3_20_pro-life': {
        'check_func': 'pro-life|abortion|pro life',
        'groundtruth_explanation': 'is pro-life'
    },
    'd3_21_pro-choice': {
        'check_func': 'pro-choice|abortion|pro choice',
        'groundtruth_explanation': 'supports abortion'
    },
    'd3_22_physics': {
        'check_func': 'physics',
        'groundtruth_explanation': 'is about physics'
    },
    'd3_23_computer science': {
        'check_func': 'computer science|computer|artificial intelligence|ai',
        'groundtruth_explanation': 'is related to computer science'
    },
    'd3_24_statistics': {
        'check_func': 'statistics|stat|probability',
        'groundtruth_explanation': 'is about statistics'
    },
    'd3_25_math': {
        'check_func': 'math|arithmetic|algebra|geometry',
        'groundtruth_explanation': 'is about math research'
    },
    'd3_26_grammar': {
        'check_func': 'grammar|syntax|punctuation|grammat',
                      'groundtruth_explanation': 'is ungrammatical'
    },
    'd3_27_grammar': {
        'check_func': 'grammar|syntax|punctuation|grammat',
                      'groundtruth_explanation': 'is grammatical'
    },
    'd3_28_sexis': {
        'check_func': 'sexis|women|femini',
        'groundtruth_explanation': 'is offensive to women'
    },
    'd3_29_sexis': {
        'check_func': 'sexis|women|femini',
        'groundtruth_explanation': 'supports feminism'
    },
    'd3_30_news': {
        'check_func': 'news|international|current events',
        'groundtruth_explanation': 'is about world news'
    },
    'd3_31_sports': {
        'check_func': 'sports',
        'groundtruth_explanation': 'is about sports news'
    },
    'd3_32_business': {
        'check_func': 'business|economics|finance',
        'groundtruth_explanation': 'is related to business'
    },
    'd3_33_tech': {
        'check_func': 'tech',
        'groundtruth_explanation': 'is related to technology'
    },
    'd3_34_bad': {
        'check_func': 'bad|negative|awful|terrible|horrible|poor|boring|dislike',
        'groundtruth_explanation': 'contains a bad movie review'
    },
    'd3_35_good': {
        'check_func': 'good|great|like|love|positive|awesome|amazing|excellent',
        'groundtruth_explanation': 'thinks the movie is good'
    },
    'd3_36_quantity': {
        'check_func': 'quantity|number|numeric',
        'groundtruth_explanation': 'asks for a quantity'
    },
    'd3_37_location': {
        'check_func': 'location|place',
        'groundtruth_explanation': 'asks about a location'
    },
    'd3_38_person': {
        'check_func': 'person|group|individual|people',
        'groundtruth_explanation': 'asks about a person'
    },
    'd3_39_entity': {
        'check_func': 'entity|thing|object',
        'groundtruth_explanation': 'asks about an entity'
    },
    'd3_40_abbrevation': {
        'check_func': 'abbrevation|abbr|acronym',
        'groundtruth_explanation': 'asks about an abbreviation'
    },
    'd3_41_defin': {
        'check_func': 'defin|meaning|explain',
        'groundtruth_explanation': 'contains a definition'
    },
    'd3_42_environment': {
        'check_func': 'environment|climate change|global warming',
        'groundtruth_explanation': 'is against environmentalist'
    },
    'd3_43_environment': {
        'check_func': 'environment|climate change|global warming',
        'groundtruth_explanation': 'is environmentalist'
    },
    'd3_44_spam': {
        'check_func': 'spam|annoying|unwanted',
        'groundtruth_explanation': 'is a spam'
    },
    'd3_45_fact': {
        'check_func': 'fact|info|knowledge',
        'groundtruth_explanation': 'asks for factual information'
    },
    'd3_46_opinion': {
        'check_func': 'opinion|personal|bias',
        'groundtruth_explanation': 'asks for an opinion'
    },
    'd3_47_math': {
        'check_func': 'math|science',
        'groundtruth_explanation': 'is related to math and science'
    },
    'd3_48_health': {
        'check_func': 'health|medical|disease',
        'groundtruth_explanation': 'is related to health'
    },
    'd3_49_computer': {
        'check_func': 'computer|internet|web',
        'groundtruth_explanation': 'related to computer or internet'
    },
    'd3_50_sport': {
        'check_func': 'sport',
        'groundtruth_explanation': 'is related to sports'
    },
    'd3_51_entertainment': {
        'check_func': 'entertainment|music|movie|tv',
        'groundtruth_explanation': 'is about entertainment'
    },
    'd3_52_family': {
        'check_func': 'family|relationships',
        'groundtruth_explanation': 'is about family and relationships'
    },
    'd3_53_politic': {
        'check_func': 'politic|government|law',
        'groundtruth_explanation': 'is related to politics or government'
    }
}
ks = list(TASKS_D3.keys())

for k in ks:
    TASKS_D3[k]['gen_func'] = partial(fetch_data, k)

if __name__ == '__main__':
    print(TASKS_D3)
    df = TASKS_D3['d3_5_evacuate']['gen_func']()
    print('df.shape', df.shape)
    print('df.keys()', df.keys())
    print(df['label'].head())
    for i in range(10):
        print(df[df['label'] == 1][['input', 'label']].iloc[i]['input'])
