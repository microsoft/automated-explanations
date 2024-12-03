from tqdm import tqdm
import imodelsx.llm
import pandas as pd
import joblib
import json


prompt_template = '''Here is a list of phrases:
{s}

What is a common theme among these phrases (especially the top ones)? Return only a concise phrase.'''

# subject = 'S02'
# suffix_setting = '_filt=0.15'
# suffix_setting = '_rj_custom'


# d_selected = pd.read_pickle(
# f'top_clusters_by_pfc_cluster_{subject}{suffix_setting}.pkl')
# gpt4 = imodelsx.llm.get_llm('gpt-4')
# explanations = []
# print(d_selected.shape)

# for i, row in d_selected.iterrows():
#     top_ngrams = row['top_ngrams']
#     s = '- ' + '\n- '.join(top_ngrams[:60])
#     prompt = prompt_template.format(s=s)
#     # print(prompt)
#     explanation = gpt4(prompt, use_cache=True)
#     if explanation is None:
#         explanation = '<FAILED FOR CONTENT MODERATION>'
#         # explanation = gpt4(prompt, use_cache=False)
#     explanations.append(explanation)
#     print(explanations)
# joblib.dump(
#     explanations, f'explanations_by_pfc_cluster_{subject}{suffix_setting}.jbl')


# subject = 'S01'
# # subject = 'S02'
# # subject = 'S03'
# # suffix_setting = '_fedorenko'
# suffix_setting = '_spotlights'

# explanations = {}
# if suffix_setting == '_spotlights':
#     top_ngrams_df = pd.read_pickle(
#         f'top_ngrams_custom_communication_{subject}{suffix_setting}_filtered.pkl')
# else:
#     top_ngrams_df = pd.read_pickle(
#         f'top_ngrams_custom_communication_{subject}{suffix_setting}.pkl')
# gpt4 = imodelsx.llm.get_llm('gpt-4')
# for k in tqdm(top_ngrams_df.columns):

#     s = '- ' + '\n- '.join(top_ngrams_df[k].iloc[:100])
#     prompt = prompt_template.format(s=s)
#     if not k in explanations:
#         # print(prompt)
#         explanations[k] = gpt4(prompt)
#     print(explanations)

# out_name = f'explanations_by_roi_communication_{subject}{suffix_setting}.json'
# print('saved to ', out_name)
# json.dump(explanations, open(out_name, 'w'), indent=4)


# normal rois
subject = 'S03'

explanations = {}
top_ngrams_df = pd.read_csv(f'top_ngrams_by_roi_{subject}.csv', index_col=0)

gpt4 = imodelsx.llm.get_llm('gpt-4')
for k in tqdm(top_ngrams_df.columns):
    print('vals', top_ngrams_df[k].iloc[:100])
    s = '- ' + '\n- '.join(top_ngrams_df[k].iloc[:100])
    prompt = prompt_template.format(s=s)
    if not k in explanations:
        # print(prompt)
        explanations[k] = gpt4(prompt)
    print(explanations)

out_name = f'explanations_known_roi_{subject}.json'
print('saved to ', out_name)
json.dump(explanations, open(out_name, 'w'), indent=4)
