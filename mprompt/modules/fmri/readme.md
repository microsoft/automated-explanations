# Contents



**Stories -** Contains the TextGrids and TRfiles for every story in the training and test sets. Requires the ridge_utils python module to unpickle successfully. This module has been included in this folder as a zipped directory. 



**Stimulus Features** - Contains the story stimuli, transformed into LM-based contextual embeddings. Uses the 33rd layer of OPT-30B for embeddding generation. 4 time delays (2s, 4s, 6s, 8s). Downsampled using Lanczos downsampling with a window size of 3. **Not trimmed like the responses.**



**Responses Folder** - Contains BOLD responses for the 3 subjects. **Responses are pretrimmed compared to the stimulus.** In order to align the dimensions of the two, just trim 10 TRs from the front of each set of features and 5 TRs from the back. 



**Model Weights** - Contains computed model weights for each of the 3 subjects. Generated via bootstrapped ridge regression from data contained in stimulus features folder to data contained in responses folder.



**Voxel Performances** - Contains performances of model weights from models in the model weights folder, in both a loadable form and as a cortical flatmap visualization.



**Voxel Lists -** Contains a list of voxels with varied semantic selectivity, chosen by looking at the convex hull of encoding model weights and test set performance threshold.



**Voxel ROIs** - 

- Voxel anat ROIs - A dictionary of the anatomical ROIs for every voxel (if known).
- Voxel func ROIs - A dictionary of the functional ROIs for every voxel (if known).
- Anat ROI voxels - A dictionary of the known voxels in each anatomical ROI.
- Func ROI voxels - A dictionary of the known voxels in each functional ROI.

Also contains a PDF of a flattened cortical surface with all anatomical and functional ROIs marked.


# Code snippets

***Load all .jbl files with joblib.load***



**Example story data loading code:**

(Requires `pip install -e setup.py` in the `csinva/fmri` directory)

```python
# Load stories
grids = joblib.load("grids_all.jbl")
trfiles = joblib.load("trfiles_all.jbl")

# Make datasequences
from ridge_utils.dsutils import make_word_ds, make_phoneme_ds
wordseqs = make_word_ds(grids, trfiles)

# For story words
print(wordseqs['wheretheressmoke'].data)

# For word times 
print(wordseqs['wheretheressmoke'].data_times)
```



**Example model feature loading code with trimming and feature normalization:**

```python
zscore = lambda v: (v - v.mean(0)) / v.std(0)
raw_features = joblib.load("OPT_features.jbl")
stim = np.nan_to_num(np.vstack([zscore(raw_features[story][10:-5]) for story in stories]))
```


**Example model prediction code:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-30b", device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")

text = tokenizer.encode("This is a demo.")
weights = joblib.load("wt_UTS01.jbl")

def run_opt_model(text, weights)
    '''Takes pretokenized text and ridge weights as input. Returns OPT ridge model predictions.
    Ideally, you would use downsampled features instead of copying features across time delays.'''
    inputs = {}
    inputs['input_ids'] = torch.tensor([text]).int()
    inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape)
    out = list(model(**inputs, output_hidden_states=True)[2])[33][0][-1].cpu().detach().numpy()
    pred = np.dot(np.hstack([out,out,out,out]), weights)
    return pred

model_predictions = run_opt_model(text, weights)
```



**Train and Test Split for provided models:**

**For UTS01:**

```python
train_stories = ['itsabox', 'odetostepfather', 'inamoment',  'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
test_stories = ['wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood']
```



**For UTS02 and UTS03:**

```python
train_stories = ['itsabox', 'odetostepfather', 'inamoment', 'afearstrippedbare', 'findingmyownrescuer', 'hangtime', 'ifthishaircouldtalk', 'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection', 'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian', 'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts', 'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food', 'theclosetthatateeverything', 'escapingfromadirediagnosis', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes', 'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls', 'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs', 'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow', 'breakingupintheageofgoogle', 'gpsformylostidentity', 'marryamanwholoveshismother', 'eyespy', 'treasureisland', 'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner', 'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse', 'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic', 'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck', 'thesecrettomarriage', 'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope', 'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove', 'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail', 'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus', 'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar', 'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview', 'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo', 'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance', 'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange', 'leavingbaghdad']
test_stories = ['wheretheressmoke', 'onapproachtopluto', 'fromboyhoodtofatherhood']
```