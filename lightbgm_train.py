# -*- coding: utf-8 -*-

import nltk
nltk.download('stopwords')
!pip install optuna

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import rankdata
import re
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import optuna.integration.lightgbm as lgb 
from sklearn.model_selection import KFold

TRAIN_DATA_PATH = "/content/drive/MyDrive/Jigsaw/jigsaw-toxic-comment-classification-challenge/train.csv"
VALID_DATA_PATH = "/content/drive/MyDrive/Jigsaw/jigsaw-toxic-severity-rating/validation_data.csv"
TEST_DATA_PATH = "/content/drive/MyDrive/Jigsaw/jigsaw-toxic-severity-rating/comments_to_score.csv"

df_train2 = pd.read_csv(TRAIN_DATA_PATH)
df_valid2 = pd.read_csv(VALID_DATA_PATH)
df_test2 = pd.read_csv(TEST_DATA_PATH)
cat_mtpl = {'obscene': 1.6, 'toxic': 1, 'threat': 1.3, 
            'insult': 1, 'severe_toxic': 2, 'identity_hate': 0.9}

for category in cat_mtpl:
    df_train2[category] = df_train2[category] * cat_mtpl[category]
df_train2['score'] = df_train2.loc[:, 'toxic':'identity_hate'].mean(axis=1)
df_train2['y'] = df_train2['score']

min_len = (df_train2['y'] > 0).sum()  # len of toxic comments
df_y0_undersample = df_train2[df_train2['y'] == 0].sample(n=min_len, random_state=41)  # take non toxic comments
df_train_new = pd.concat([df_train2[df_train2['y'] > 0], df_y0_undersample])  # make new df

from nltk.corpus import stopwords
stop = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text): return [lemmatizer.lemmatize(w) for w in text]

def clean(data, col):
        data[col] = data[col].str.replace(r"what's", "what is ")
        data[col] = data[col].str.replace(r"\'ve", " have ")
        data[col] = data[col].str.replace(r"can't", "cannot ")
        data[col] = data[col].str.replace(r"n't", " not ")
        data[col] = data[col].str.replace(r"i'm", "i am ")
        data[col] = data[col].str.replace(r"\'re", " are ")
        data[col] = data[col].str.replace(r"\'d", " would ")
        data[col] = data[col].str.replace(r"\'ll", " will ")
        data[col] = data[col].str.replace(r"\'scuse", " excuse ")
        data[col] = data[col].str.replace(r"\'s", " ")
        data[col] = data[col].str.replace('\n', ' \n ')
        data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)', r'\1 \2 \3')
        data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}', r'\1\1\1')
        data[col] = data[col].str.replace(r'([*!?\']+)', r' \1 ')
        data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b', r'\1\1')
        data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1')
        data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()
        data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()
        data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        return data

)

labels = df_train_new['y']
df_train_new = clean(df_train_new,"comment_text")
comments = df_train_new['comment_text']
vec = TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))
comments_tr = vec.fit_transform(comments)

comments_tr

#X_train, X_valid, y_train, y_valid = train_test_split(comments_tr, labels, shuffle=True, test_size=0.2)
lgb_train = lgb.Dataset(comments_tr, labels)
#lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
params = {
    "objective":"regression",
    "metric":"rmse",
    "random_seed":0
    
}

tuner = lgb.LightGBMTunerCV(params, lgb_train, verbose_eval=100, early_stopping_rounds=100, folds=KFold(n_splits=2))
tuner.run()
best_params = tuner.best_params
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))
"""booster = LGB_optuna.LightGBMTuner(params = params,
                                   train_set = lgb_train,
                                   valid_sets = lgb_eval,
                                   optuna_seed = 123)"""
#booster.run()
#best = LGB_optuna.train(param, lgb_train, valid_sets=lgb_eval)
#best_params, history = {}, []
"""model = LGB_optuna.train(params,
                        lgb_train,
                        num_boost_round=1000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=10,
                        best_params=best_params,
                        tuning_history=history,
                        verbose_eval=50 )"""

#regressor = Ridge(random_state=41, alpha=1.5)
#regressor.fit(comments_tr, labels)
df_valid2 = clean(df_valid2,'less_toxic')
df_valid2 = clean(df_valid2,'more_toxic')
less_toxic_comments = df_valid2['less_toxic']
more_toxic_comments = df_valid2['more_toxic']
less_toxic = vec.transform(less_toxic_comments)
more_toxic = vec.transform(more_toxic_comments)
# make predictions
y_pred_less = model.predict(less_toxic)
y_pred_more = model.predict(more_toxic)

#y_pred_less = regressor.predict(less_toxic)
#y_pred_more = regressor.predict(more_toxic)

print(f'val : {(y_pred_less < y_pred_more).mean()}')

"""texts = df_test2['text']
texts = vec.transform(texts)

df_test2['prediction'] = regressor.predict(texts)
df_test2 = df_test2[['comment_id','prediction']]

df_test2['score'] = df_test2['prediction']
df_test2 = df_test2[['comment_id','score']]


df_test2.to_csv('./submission2.csv', index=False)"""

jr = pd.read_csv("/content/drive/MyDrive/Jigsaw/jigsaw-regression-based-data/train_data_version2.csv")
jr.shape
df = jr[['text', 'y']]
df = clean(df,"text")
vec_1 = TfidfVectorizer(analyzer='char_wb', max_df=0.7, min_df=1, ngram_range=(2, 5) )
X = vec_1.fit_transform(df['text'])
z = df["y"].values
y=np.around ( z ,decimals = 2)

model1=Ridge(alpha=1.5)
model1.fit(X, y)
df_test = pd.read_csv("/content/drive/MyDrive/Jigsaw/jigsaw-toxic-severity-rating/comments_to_score.csv")
test=vec_1.transform(df_test['text'])
jr_preds=model1.predict(test)
df_test['score1']=rankdata( jr_preds, method='ordinal') 
rud_df = pd.read_csv("/content/drive/MyDrive/Jigsaw/ruddit-jigsaw-dataset/Dataset/ruddit_with_text.csv")
#print(f"rud_df:{rud_df.shape}")
rud_df['y'] = rud_df["offensiveness_score"] 
df = rud_df[['txt', 'y']].rename(columns={'txt': 'text'})

df = df[df["text"] != "[deleted]"]
df= clean(df,"text")

vec_2 = TfidfVectorizer(analyzer='char_wb', max_df=0.7, min_df=3, ngram_range=(3, 4) )
X = vec_2.fit_transform(df['text'])
z = df["y"].values
y=np.around ( z ,decimals = 1)
model2=Ridge(alpha=1.5)
model2.fit(X, y)
test=vec_2.transform(df_test['text'])
rud_preds=model2.predict(test)
df_test['score2']=rankdata( rud_preds, method='ordinal')
df_test['score']=df_test['score1']+df_test['score2']
df_test[['comment_id', 'score']].to_csv("submission1.csv", index=False)

less_toxic = vec_1.transform(less_toxic_comments)
more_toxic = vec_1.transform(more_toxic_comments)
y_pred_les = model1.predict(less_toxic)
y_pred_mor = model1.predict(more_toxic)

print(f'val : {(y_pred_les < y_pred_mor).mean()}')

y_pred_les

y_pred_mor

print(f'val : {(y_pred_les < y_pred_mor).mean()}')

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import re 
import scipy
from scipy import sparse

from IPython.display import display
from pprint import pprint
from matplotlib import pyplot as plt 

import time
import scipy.optimize as optimize
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_colwidth=300
pd.options.display.max_columns = 100

from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.svm import SVR

df_train = pd.read_csv("/content/drive/MyDrive/Jigsaw/jigsaw-toxic-comment-classification-challenge/train.csv")
df_sub = pd.read_csv("/content/drive/MyDrive/Jigsaw/jigsaw-toxic-severity-rating/comments_to_score.csv")

cat_mtpl = {'obscene': 0.16, 'toxic': 0.32, 'threat': 1.5, 
            'insult': 0.64, 'severe_toxic': 1.5, 'identity_hate': 1.5}

for category in cat_mtpl:
    df_train[category] = df_train[category] * cat_mtpl[category]

df_train['score'] = df_train.loc[:, 'toxic':'identity_hate'].sum(axis=1)

df_train['y'] = df_train['score']

min_len = (df_train['y'] > 0).sum()  # len of toxic comments
df_y0_undersample = df_train[df_train['y'] == 0].sample(n=min_len, random_state=201)  # take non toxic comments
df_train_new = pd.concat([df_train[df_train['y'] > 0], df_y0_undersample])  # make new df
df_train = df_train.rename(columns={'comment_text':'text'})
"""
def text_cleaning(text):
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis
    
    text - Text piece to be cleaned.
    '''
    template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
    text = template.sub(r'', text)
    
    soup = BeautifulSoup(text, 'lxml') #Removes HTML tags
    only_text = soup.get_text()
    text = only_text
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
    text = re.sub(' +', ' ', text) #Remove Extra Spaces
    text = text.strip() # remove spaces at the beginning and at the end of string

    return text"""

from nltk.corpus import stopwords
stop = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text): return [lemmatizer.lemmatize(w) for w in text]

def clean(data, col):
        data[col] = data[col].str.replace(r"what's", "what is ")
        data[col] = data[col].str.replace(r"\'ve", " have ")
        data[col] = data[col].str.replace(r"can't", "cannot ")
        data[col] = data[col].str.replace(r"n't", " not ")
        data[col] = data[col].str.replace(r"i'm", "i am ")
        data[col] = data[col].str.replace(r"\'re", " are ")
        data[col] = data[col].str.replace(r"\'d", " would ")
        data[col] = data[col].str.replace(r"\'ll", " will ")
        data[col] = data[col].str.replace(r"\'scuse", " excuse ")
        data[col] = data[col].str.replace(r"\'s", " ")
        data[col] = data[col].str.replace('\n', ' \n ')
        data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)', r'\1 \2 \3')
        data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}', r'\1\1\1')
        data[col] = data[col].str.replace(r'([*!?\']+)', r' \1 ')
        data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b', r'\1\1')
        data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1')
        data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()
        data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()
        data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
        return data

tqdm.pandas()
df_train = clean(df_train,"text")
df = df_train.copy()
df['y'].value_counts(normalize=True)
min_len = (df['y'] >= 0.1).sum()
df_y0_undersample = df[df['y'] == 0].sample(n=min_len * 2, random_state=402)
df = pd.concat([df[df['y'] >= 0.1], df_y0_undersample])
vec = TfidfVectorizer(min_df= 3, max_df=0.5, analyzer = 'char_wb', ngram_range = (3,5))
X = vec.fit_transform(df['text'])
model = Ridge(alpha=0.5)
model.fit(X, df['y'])
l_model = Ridge(alpha=1.5)
l_model.fit(X, df['y'])
s_model = Ridge(alpha=2.)
s_model.fit(X, df['y'])
df_val = pd.read_csv("/content/drive/MyDrive/Jigsaw/jigsaw-toxic-severity-rating/validation_data.csv")
tqdm.pandas()
df_val = clean(df_val,'less_toxic')
df_val = clean(df_val,'more_toxic')
X_less_toxic = vec.transform(df_val['less_toxic'])
X_more_toxic = vec.transform(df_val['more_toxic'])
p1 = l_model.predict(X_less_toxic)
p2 = l_model.predict(X_more_toxic)
# Validation Accuracy
print(f'val : {(p1 < p2).mean()}')
df_sub = pd.read_csv("/content/drive/MyDrive/Jigsaw/jigsaw-toxic-severity-rating/comments_to_score.csv")
tqdm.pandas()
df_sub = clean(df_sub,'text')
X_test = vec.transform(df_sub['text'])
p3 = model.predict(X_test)
p4 = l_model.predict(X_test)
p5 = s_model.predict(X_test)
df_sub['score'] = (p3 + p4 + p5) / 3.
df_sub['score'] = df_sub['score']
df_sub[['comment_id', 'score']].to_csv("submission3.csv", index=False)

p1 = s_model.predict(X_less_toxic)
p2 = s_model.predict(X_more_toxic)
# Validation Accuracy
print(f'val : {(p1 < p2).mean()}')

data = pd.read_csv("./submission1.csv",index_col="comment_id")
data["score1"] = data["score"]
data["score1"] = rankdata( data["score1"], method='ordinal')
data["score2"] = pd.read_csv("./submission2.csv",index_col="comment_id")["score"]
data["score2"] = rankdata( data["score2"], method='ordinal')

data["score3"] = pd.read_csv("./submission3.csv",index_col="comment_id")["score"]
data["score3"] = rankdata( data["score3"], method='ordinal')

data["score"] = .82*data["score1"] + .67*data["score2"] + data["score3"]*.31

df_test = data
df_test["score"] = rankdata( df_test["score"], method='ordinal')
df_test["score"].to_csv('./submission.csv')
