import imodelsx.llm
import pandas as pd
import joblib

subject = 'S02'
d_selected = pd.read_pickle(f'top_clusters_by_pfc_cluster_{subject}.pkl')
gpt4 = imodelsx.llm.get_llm('gpt-4')
explanations = []


prompt_template = '''Here is a list of phrases:
{s}

What is a common theme among these phrases? Return only a concise phrase.'''
for i, row in d_selected.iterrows():
    top_ngrams = row['top_ngrams']
    s = '- ' + '\n- '.join(top_ngrams[:60])
    prompt = prompt_template.format(s=s)
    # print(prompt)
    explanations.append(gpt4(prompt, use_cache=True))
    print(explanations)
joblib.dump(explanations, f'explanations_by_pfc_cluster_{subject}.jbl')
