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
    df['input'] = df['input'].map(lambda s: f'Input: {s} Answer:')
    # Fix output: Prepend a space and add newlines to match output format of number tasks
    df['output'] = df['output'].map(lambda s: f' {s}\n\n')
    return df

TASKS_D3 = {
    'd3_0_irony': {'check_func': 'irony|sarcas', 'description': 'contains irony'},
    'd3_1_objective': {'check_func': 'objective|factual|nonpersonal|neutral',
  'description': 'is a more objective description of what happened'},
 'd3_2_subjective': {'check_func': 'subjective|opinion|personal|bias',
  'description': 'contains subjective opinion'},
 'd3_3_god': {'check_func': 'god|religious|religion',
  'description': 'believes in god'},
 'd3_4_atheism': {'check_func': 'atheism|atheist|anti-religion|against religion',
  'description': 'is against religion'},
 'd3_5_evacuate': {'check_func': 'evacuate|flee|escape',
  'description': 'involves a need for people to evacuate'},
 'd3_6_terorrism': {'check_func': 'terorrism|terror',
  'description': 'describes a situation that involves terrorism'},
 'd3_7_crime': {'check_func': 'crime|criminal|criminality',
  'description': 'involves crime'},
 'd3_8_shelter': {'check_func': 'shelter|home|house',
  'description': 'describes a situation where people need shelter'},
 'd3_9_food': {'check_func': 'food|hunger|needs',
  'description': 'is related to food security'},
 'd3_10_infrastructure': {'check_func': 'infrastructure|buildings|roads|bridges|build',
  'description': 'is related to infrastructure'},
 'd3_11_regime change': {'check_func': 'regime change|coup|revolution|revolt|political action|political event|upheaval',
  'description': 'describes a regime change'},
 'd3_12_medical': {'check_func': 'medical|health',
  'description': 'is related to a medical situation'},
 'd3_13_water': {'check_func': 'water',
  'description': 'involves a situation where people need clean water'},
 'd3_14_search': {'check_func': 'search|rescue|help',
  'description': 'involves a search/rescue situation'},
 'd3_15_utility': {'check_func': 'utility|energy|sanitation|electricity|power',
  'description': 'expresses need for utility, energy or sanitation'},
 'd3_16_hillary': {'check_func': 'hillary|clinton|against Hillary|opposed to Hillary|republican|against Clinton|opposed to Clinton',
  'description': 'is against Hillary'},
 'd3_17_hillary': {'check_func': 'hillary|clinton|support Hillary|support Clinton|democrat',
  'description': 'supports hillary'},
 'd3_18_offensive': {'check_func': 'offensive|toxic|abusive|insulting|insult|abuse|offend|offend',
  'description': 'contains offensive content'},
 'd3_19_offensive': {'check_func': 'offensive|toxic|abusive|insulting|insult|abuse|offend|offend|women|immigrants',
  'description': 'insult women or immigrants'},
 'd3_20_pro-life': {'check_func': 'pro-life|abortion|pro life',
  'description': 'is pro-life'},
 'd3_21_pro-choice': {'check_func': 'pro-choice|abortion|pro choice',
  'description': 'supports abortion'},
 'd3_22_physics': {'check_func': 'physics', 'description': 'is about physics'},
 'd3_23_computer science': {'check_func': 'computer science|computer|artificial intelligence|ai',
  'description': 'is related to computer science'},
 'd3_24_statistics': {'check_func': 'statistics|stat|probability',
  'description': 'is about statistics'},
 'd3_25_math': {'check_func': 'math|arithmetic|algebra|geometry',
  'description': 'is about math research'},
 'd3_26_grammar': {'check_func': 'grammar|syntax|punctuation|grammat',
  'description': 'is ungrammatical'},
 'd3_27_grammar': {'check_func': 'grammar|syntax|punctuation|grammat',
  'description': 'is grammatical'},
 'd3_28_sexis': {'check_func': 'sexis|women|femini',
  'description': 'is offensive to women'},
 'd3_29_sexis': {'check_func': 'sexis|women|femini',
  'description': 'supports feminism'},
 'd3_30_news': {'check_func': 'news|international|current events',
  'description': 'is about world news'},
 'd3_31_sports': {'check_func': 'sports',
  'description': 'is about sports news'},
 'd3_32_business': {'check_func': 'business|economics|finance',
  'description': 'is related to business'},
 'd3_33_tech': {'check_func': 'tech',
  'description': 'is related to technology'},
 'd3_34_bad': {'check_func': 'bad|negative|awful|terrible|horrible|poor|boring|dislike',
  'description': 'contains a bad movie review'},
 'd3_35_good': {'check_func': 'good|great|like|love|positive|awesome|amazing|excellent',
  'description': 'thinks the movie is good'},
 'd3_36_quantity': {'check_func': 'quantity|number|numeric',
  'description': 'asks for a quantity'},
 'd3_37_location': {'check_func': 'location|place',
  'description': 'asks about a location'},
 'd3_38_person': {'check_func': 'person|group|individual|people',
  'description': 'asks about a person'},
 'd3_39_entity': {'check_func': 'entity|thing|object',
  'description': 'asks about an entity'},
 'd3_40_abbrevation': {'check_func': 'abbrevation|abbr|acronym',
  'description': 'asks about an abbreviation'},
 'd3_41_defin': {'check_func': 'defin|meaning|explain',
  'description': 'contains a definition'},
 'd3_42_environment': {'check_func': 'environment|climate change|global warming',
  'description': 'is against environmentalist'},
 'd3_43_environment': {'check_func': 'environment|climate change|global warming',
  'description': 'is environmentalist'},
 'd3_44_spam': {'check_func': 'spam|annoying|unwanted',
  'description': 'is a spam'},
 'd3_45_fact': {'check_func': 'fact|info|knowledge',
  'description': 'asks for factual information'},
 'd3_46_opinion': {'check_func': 'opinion|personal|bias',
  'description': 'asks for an opinion'},
 'd3_47_math': {'check_func': 'math|science',
  'description': 'is related to math and science'},
 'd3_48_health': {'check_func': 'health|medical|disease',
  'description': 'is related to health'},
 'd3_49_computer': {'check_func': 'computer|internet|web',
  'description': 'related to computer or internet'},
 'd3_50_sport': {'check_func': 'sport', 'description': 'is related to sports'},
 'd3_51_entertainment': {'check_func': 'entertainment|music|movie|tv',
  'description': 'is about entertainment'},
 'd3_52_family': {'check_func': 'family|relationships',
  'description': 'is about family and relationships'},
 'd3_53_politic': {'check_func': 'politic|government|law',
  'description': 'is related to politics or government'}}
ks = list(TASKS_D3.keys())

for k in ks:
    TASKS_D3[k]['gen_func'] = partial(fetch_data, k)

if __name__ == '__main__':
    print(TASKS_D3)
    df = TASKS_D3['d3_0_irony']['gen_func']()
    print('df.shape', df.shape)
    print('df.keys()', df.keys())
