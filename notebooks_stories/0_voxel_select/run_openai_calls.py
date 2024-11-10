import imodelsx.llm
import pandas as pd
import joblib

subject = 'S02'
suffix_setting = '_filt=0.15'
# suffix_setting = '_rj_custom'


d_selected = pd.read_pickle(
    f'top_clusters_by_pfc_cluster_{subject}{suffix_setting}.pkl')
gpt4 = imodelsx.llm.get_llm('gpt-4')
explanations = []
print(d_selected.shape)
prompt_template = '''Here is a list of phrases:
{s}

What is a common theme among these phrases (especially the top ones)? Return only a concise phrase.'''
for i, row in d_selected.iterrows():
    top_ngrams = row['top_ngrams']
    s = '- ' + '\n- '.join(top_ngrams[:60])
    prompt = prompt_template.format(s=s)
    # print(prompt)
    explanation = gpt4(prompt, use_cache=True)
    if explanation is None:
        explanation = '<FAILED FOR CONTENT MODERATION>'
        # explanation = gpt4(prompt, use_cache=False)
    explanations.append(explanation)
    print(explanations)
joblib.dump(
    explanations, f'explanations_by_pfc_cluster_{subject}{suffix_setting}.jbl')
