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


def fetch_data(task_name, return_df=False):
    df = pd.read_csv(oj(D3_PROCESSED_DIR,
                        task_name[:task_name.rindex('_')] + '.csv'))
    # print('df', df.head())
    # Fix input: Encourage model to answer output as next token.
    # df['input'] = df['input'].map(lambda s: f'Input: {s} Answer:')
    # Fix output: Prepend a space and add newlines to match output format of number tasks
    # df['output'] = df['output'].map(lambda s: f' {s}\n\n')
    if return_df:
        return df
    else:
        return df['input'].values.tolist()





# Note: I have only manually checked the first 20 datasets here
# There might be more issues....
TASKS_D3 = {
    'd3_0_irony': {
        'check_func': 'irony|sarcas',
        'groundtruth_explanation': 'contains irony',
        'template': 'The phrase "{input}" contains',
        'target_token': ' sarcasm',
        'examples': ["that's just great.", "thanks a lot.", "how original", "how convenient"],
    },
    'd3_1_objective': {
        'check_func': 'objective|factual|nonpersonal|neutral|unbias',
        'groundtruth_explanation': 'is a more objective description of what happened',
        'template': 'The phrase "{input}" is',
        'target_token': ' unbiased',
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
        'check_func': 'evacuat|flee|escape',
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
        'examples': ['bomb threat', 'hijacking', 'radicalization', 'terrorism', 'hostage-taking'],
    },
    'd3_7_crime': {
        'check_func': 'crime|criminal|criminality',
        'groundtruth_explanation': 'involves crime',
        'template': 'The phrase "{input}" involves',
        'target_token': ' crime',
        'examples': ['robbery', 'theft', 'murder', 'fraud', 'embezzlement'],
    },
    'd3_8_shelter': {
        'check_func': 'shelter|home|house',
        'groundtruth_explanation': 'describes a situation where people need shelter',
        'template': 'The phrase "{input}" describes a situation where people need',
        'target_token': ' shelter',
        'examples': ['homelessness', 'natural disaster', 'refugee crisis', 'emergency housing', 'eviction'],
    },
    'd3_9_food': {
        'check_func': 'food|hunger|needs',
        'groundtruth_explanation': 'is related to food security',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' hunger',
        'examples': ['food insecurity', 'malnutrition', 'food bank', 'hunger strike', 'food assistance'],
    },
    'd3_10_infrastructure': {
        'check_func': 'infrastructure|buildings|roads|bridges|build',
        'groundtruth_explanation': 'is related to infrastructure',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' infrastructure',
        'examples': ['transportation', 'construction', 'public works', 'utilities', 'urban planning'],
    },
    'd3_11_regime change': {
        'check_func': 'regime change|coup|revolution|revolt|political action|political event|upheaval',
        'groundtruth_explanation': 'describes a regime change',
        'template': 'The phrase "{input}" describes a',
        'target_token': ' regime change',
        'examples': ['overthrow', 'uprising', 'transition', 'elections', "coup"],
    },
    'd3_12_medical': {
        'check_func': 'medical|health',
        'groundtruth_explanation': 'is related to a medical situation',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' health',
        'examples': ['disease', 'vaccination', 'emergency', 'health', 'healthcare'],
    },
    'd3_13_water': {
        'check_func': 'water',
        'groundtruth_explanation': 'involves a situation where people need clean water',
        'template': 'The phrase "{input}" involves a situation involving scarcity of',
        'target_token': ' water',
        'examples': ['scarcity', 'waterborne', 'drought'],
    },
    'd3_14_search': {
        'check_func': 'search|rescue|help',
        'groundtruth_explanation': 'involves a search/rescue situation',
        'template': 'The phrase "{input}" describes a situation involving',
        'target_token': ' rescue',
        'examples': ['missing', 'disaster', 'emergency', 'rescue', 'relief'],
    },
    'd3_15_utility': {
        'check_func': 'utility|energy|sanitation|electricity|power',
        'groundtruth_explanation': 'expresses need for utility, energy or sanitation',
        'template': 'The phrase "{input}" expressed need for',
        'target_token': ' utility',
        'examples': ['electricity', 'electricity', 'power', 'sewage', 'garbage'],
    },
    'd3_16_hillary': {
        'check_func': 'hillary|clinton|against Hillary|opposed to Hillary|republican|against Clinton|opposed to Clinton',
        'groundtruth_explanation': 'is against Hillary',
        'template': 'The phrase "{input}" involves opposing',
        'target_token': ' Hillary',
        'examples': ['Never Hillary', 'Stop Hillary', 'Hillary for Prison', 'Anti-Clinton', 'Never Clinton'],
    },
    'd3_17_hillary': {
        'check_func': 'hillary|clinton|support Hillary|support Clinton|democrat',
        'groundtruth_explanation': 'supports hillary',
        'template': 'The phrase "{input}" involves supporting',
        'target_token': ' Hillary',
        'examples': ["I'm With Her", 'Hillary 2020', 'Clinton Foundation', 'Democrats for Hillary', 'Hillary supporter'],
    },
    'd3_18_offensive': {
        'check_func': 'offensive|toxic|abusive|insulting|insult|abuse|offend|offend|derogatory',
        'groundtruth_explanation': 'contains offensive content',
        'template': 'The phrase "{input}" contains content that is',
        'target_token': ' derogatory',
        'examples': ['racial slurs', 'sexist comments', 'homophobic remarks', 'hate speech', 'harassment'],
    },
    'd3_19_offensive': {
        'check_func': 'offensive|toxic|abusive|insulting|insult|abuse|offend|offend|women|immigrants',
        'groundtruth_explanation': 'insult women or immigrants',
        'template': 'The phrase "{input}" involves behavior that is',
        'target_token': ' toxic',
        'examples': ['misogynistic comments', 'xenophobic remarks', 'sexist insults', 'hate speech against women', 'racist attacks on immigrants'],
    },
    'd3_20_pro-life': {
        'check_func': 'pro-life|abortion|pro life',
        'groundtruth_explanation': 'is pro-life',
        'template': 'The phrase "{input}" involves being',
        'target_token': ' pro-life',
        'examples': ['anti-abortion', 'defund Planned Parenthood', 'sanctity of life', 'pro-birth', 'abortion is murder'],
    },
    'd3_21_pro-choice': {
        'check_func': 'pro-choice|abortion|pro choice',
        'groundtruth_explanation': 'supports abortion',
        'template': 'The phrase "{input}" involves supporting',
        'target_token': ' abortion',
        'examples': ['reproductive rights', 'abortion access', "women's choice", 'my body my choice', 'trust women'],
    },
    'd3_22_physics': {
        'check_func': 'physics',
        'groundtruth_explanation': 'is about physics',
        'template': 'The phrase "{input}" is about',
        'target_token': ' physics',
        'examples': ["quantum mechanics", "thermodynamics", "astrophysics"],
    },
    'd3_23_computer science': {
        'check_func': 'computer science|computer|artificial intelligence|ai',
        'groundtruth_explanation': 'is related to computer science',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' computers',
        'examples': ["programming", "machine learning", "data structures"],
    },
    'd3_24_statistics': {
        'check_func': 'statistics|stat|probability',
        'groundtruth_explanation': 'is about statistics',
        'template': 'The phrase "{input}" is about',
        'target_token': ' statistics',
        'examples': ["hypothesis testing", "regression analysis", "sampling distributions"],
    },
    'd3_25_math': {
        'check_func': 'math|arithmetic|algebra|geometry',
        'groundtruth_explanation': 'is about math research',
        'template': 'The phrase "{input}" is about',
        'target_token': ' math',
        'examples': ["number theory", "topology", "combinatorics"],
    },
    'd3_26_grammar': {
        'check_func': 'grammar|syntax|punctuation|grammat|linguistic',
        'groundtruth_explanation': 'is ungrammatical',
        'template': 'The phrase "{input}" is',
        'target_token': ' ungrammatical',
        'examples': ["Me no speak English good.", "Her don't like coffee.", "I is very happy."],
    },
    'd3_27_grammar': {
        'check_func': 'grammar|syntax|punctuation|grammat|linguistic',
        'groundtruth_explanation': 'is grammatical',
        'template': 'The phrase "{input}" is',
        'target_token': ' grammatical',
        'examples': ["I am happy.", "He likes coffee.", "She speaks English well."],
    },
    'd3_28_sexis': {
        'check_func': 'sexis|women|femini',
        'groundtruth_explanation': 'is offensive to women',
        'template': 'The phrase "{input}" is',
        'target_token': ' sexist',
        'examples': ["She's not good at math because she's a woman.", "Women belong in the kitchen.", "Girls can't play sports."],
    },
    'd3_29_sexis': {
        'check_func': 'sexis|women|femini',
        'groundtruth_explanation': 'supports feminism',
        'template': 'The phrase "{input}" supports',
        'target_token': ' feminism',
        'examples': ["Equal pay for equal work.", "Reproductive rights for women.", "Girls can do anything boys can do."],
    },
    'd3_30_news': {
        'check_func': 'world|cosmopolitan|international|global',
        'groundtruth_explanation': 'is about world news',
        'template': 'The phrase "{input}" is about',
        'target_token': ' world',
        'examples': ["Politics in Europe", "Economic developments in Asia", "Natural disasters in South America"],
    },
    'd3_31_sports': {
        'check_func': 'sports',
        'groundtruth_explanation': 'is about sports news',
        'template': 'The phrase "{input}" is about',
        'target_token': ' sports news',
        'examples': ["football game", "basketball player", "tennis tournament"],
    },
    'd3_32_business': {
        'check_func': 'business|economics|finance',
        'groundtruth_explanation': 'is related to business',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' business',
        'examples': ["stock market", "corporate merger", "entrepreneurship"],
    },
    'd3_33_tech': {
        'check_func': 'tech',
        'groundtruth_explanation': 'is related to technology',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' technology',
        'examples': ["computer hardware", "artificial intelligence", "digital privacy"],
    },
    'd3_34_bad': {
        'check_func': 'bad|negative|awful|terrible|horrible|poor|boring|dislike',
        'groundtruth_explanation': 'contains a bad movie review',
        'template': 'The phrase "{input}" contains a sentiment that is',
        'target_token': ' negative',
        'examples': ["worst movie ever", "boring plot", "terrible acting"],
    },
    'd3_35_good': {
        'check_func': 'good|great|like|love|positive|awesome|amazing|excellent',
        'groundtruth_explanation': 'thinks the movie is good',
        'template': 'The phrase "{input}" thinks the movie is',
        'target_token': ' good',
        'examples': ["excellent performance", "amazing special effects", "loved the movie"],
    },
    'd3_36_quantity': {
        'check_func': 'quantity|number|numeric',
        'groundtruth_explanation': 'asks for a quantity',
        'template': 'The phrase "{input}" asks for a',
        'target_token': ' quantity',
        'examples': ["How many are there?", "What is the total?", "How much does it cost?"],
    },
    'd3_37_location': {
        'check_func': 'location|place',
        'groundtruth_explanation': 'asks about a location',
        'template': 'The phrase "{input}" asks about a',
        'target_token': ' location',
        'examples': ["Where is the nearest hotel?", "What's the address?", "Can you give me directions?"],
    },
    'd3_38_person': {
        'check_func': 'person|individual|people',
        'groundtruth_explanation': 'asks about a person',
        'template': 'The phrase "{input}" asks about a',
        'target_token': ' person',
        'examples': ["Who is the CEO?", "What's your name?", "Can you introduce me to him?"],
    },
    'd3_39_entity': {
        'check_func': 'entity|thing|object',
        'groundtruth_explanation': 'asks about an entity',
        'template': 'The phrase "{input}" asks about an',
        'target_token': ' entity',
        'examples': ["What is that?", "Can you show me?", "What's the name of this?"],
    },
    'd3_40_abbrevation': {
        'check_func': 'abbrevation|abbr|acronym',
        'groundtruth_explanation': 'asks about an abbreviation',
        'template': 'The phrase "{input}" is an',
        'target_token': ' abbreviation',
        'examples': ["NASA", "UNESCO", "FBI"],
    },
    'd3_41_defin': {
        'check_func': 'defin|meaning|explain',
        'groundtruth_explanation': 'contains a definition',
        'template': 'The phrase "{input}" contains a',
        'target_token': ' definition',
        'examples': ["What is the meaning of love?", "Define 'algorithm'", "Explain 'cognitive dissonance'"],
    },
    'd3_42_environment': {
        'check_func': 'environment|climate change|global warming',
        'groundtruth_explanation': 'is against environmentalist',
        'template': 'The phrase "{input}" is against',
        'target_token': ' environmentalism',
        'examples': ["climate change is a hoax", "stop the eco-terrorists", "save the whales... and eat them"],
    },
    'd3_43_environment': {
        'check_func': 'environment|climate change|global warming',
        'groundtruth_explanation': 'is environmentalist',
        'template': 'The phrase "{input}" supports',
        'target_token': ' environmentalism',
        'examples': ["reduce, reuse, recycle", "climate change is real", "protect the rainforest"],
    },
    'd3_44_spam': {
        'check_func': 'spam|annoying|unwanted',
        'groundtruth_explanation': 'is a spam',
        'template': 'The phrase "{input}" is',
        'target_token': ' spam',
        'examples': ["Make money fast", "Lose weight quickly", "You've won a prize"],
    },
    'd3_45_fact': {
        'check_func': 'fact|info|knowledge',
        'groundtruth_explanation': 'asks for factual information',
        'template': 'The phrase "{input}" asks for',
        'target_token': ' facts',
        'examples': ["What is the capital of France?", "When was World War II?", "How many planets are in the solar system?"],
    },
    'd3_46_opinion': {
        'check_func': 'opinion|personal|bias',
        'groundtruth_explanation': 'asks for an opinion',
        'template': 'The phrase "{input}" contains an',
        'target_token': ' opinion',
        'examples': ["favorite", "preferences", "personal", "opinion"],
    },
    'd3_47_math': {
        'check_func': 'math|science',
        'groundtruth_explanation': 'is related to math and science',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' science',
        'examples': ["Pythagorean theorem", "biology", "nature", "evolution"],
    },
    'd3_48_health': {
        'check_func': 'health|medical|disease',
        'groundtruth_explanation': 'is related to health',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' health',
        'examples': ["heart disease", "diabetes", "cancer", "depression"],
    },
    'd3_49_computer': {
        'check_func': 'computer|internet|web',
        'groundtruth_explanation': 'related to computer or internet',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' computers',
        'examples': ["website", "software", "network", "online", "programming"],
    },
    'd3_50_sport': {
        'check_func': 'sport',
        'groundtruth_explanation': 'is related to sports',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' sports',
        'examples': ["football", "basketball", "tennis", "soccer", "baseball"],
    },
    'd3_51_entertainment': {
        'check_func': 'entertainment|music|movie|tv',
        'groundtruth_explanation': 'is about entertainment',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' entertainment',
        'examples': ["concert", "film", "show", "streaming", "cinema"],
    },
    'd3_52_family': {
        'check_func': 'family|relationships',
        'groundtruth_explanation': 'is about family and relationships',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' relationships',
        'examples': ["parenting", "marriage", "divorce", "childhood", "siblings"],
    },
    'd3_53_politic': {
        'check_func': 'politic|government|law',
        'groundtruth_explanation': 'is related to politics or government',
        'template': 'The phrase "{input}" is related to',
        'target_token': ' politics',
        'examples': ["election", "legislation", "democracy", "voting", "diplomacy"],
    }
}
ks = list(TASKS_D3.keys())

for k in ks:
    TASKS_D3[k]['gen_func'] = partial(fetch_data, task_name=k)

if __name__ == '__main__':
    print(TASKS_D3)
    df = TASKS_D3['d3_5_evacuate']['gen_func']()
    print('df.shape', df.shape)
    print('df.keys()', df.keys())
    print(df['label'].head())
    for i in range(10):
        print(df[df['label'] == 1][['input', 'label']].iloc[i]['input'])
    