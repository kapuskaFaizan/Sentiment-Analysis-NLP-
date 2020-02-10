#!/usr/bin/env python
# coding: utf-8

from sklearn.preprocessing import LabelEncoder
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import pandas as pd
from keras.utils import np_utils
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm



#load_data and EDA
df1= pd.read_csv('C:/Users/faiza/Downloads/4000_comment_data_reset.csv', error_bad_lines=False)
df1.replace('@[\S]+', '', regex =True, inplace = True)
df1.replace('((www\.[\S]+)|(https?://[\S]+))', '', regex =True, inplace = True)
df=df1.apply(lambda x: x.astype(str).str.lower())

punc=np.array(['!','?',':','(',',',')','{','}','[',']','{','}',';'])

df['word_count'] = df['Comment Text'].apply(lambda x : len(x.split()))
df['char_count'] = df['Comment Text'].apply(lambda x : len(x.replace(" ","")))
df['word_density'] = df['word_count'] / (df['char_count'] + 1)
df['punc_count'] = df['Comment Text'].apply(lambda x : len([a for a in x if a in punc]))

df[['word_count', 'char_count', 'word_density','punc_count']].head(10)


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

k= (' '.join(df['Comment Text']))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


l= df.loc[df['sentiment'] =='2']
k= (' '.join(l['Comment Text']))
wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')


df['Comment Text'].replace(r'[^\w\s]',' ',regex=True,inplace=True)#pun
df['Comment Text'].replace(r'\W*\b\w{1,2}\b', "", regex=True,inplace=True)
nltk.download('wordnet')

df= df.loc[df['sentiment']!='4']
df['sentiment'].value_counts()


two= df.loc[df['sentiment']=='2']
x=two.sample(n=200)
df=df.append(x)
df=df.append(two)
#x= two.sample(n=7)
df['sentiment'].value_counts().plot(kind='bar',figsize=(7,4));


df=df.sample(frac=1)


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


df['Comment Text'] = df['Comment Text'].apply(lemmatize_text)


from nltk.corpus import stopwords
stop = stopwords.words('english')
df['Comment Text']=df['Comment Text'].apply(lambda x: [item for item in x if item not in stop])


alll= df['Comment Text']
allwords=list(alll)
k = sum(allwords,[])
np_k = np.array(k)



#stopwords = nltk.corpus.stopwords.words('english')
allWordExceptStopDist = nltk.FreqDist(w for w in np_k if w not in stop) 
freq=dict(allWordExceptStopDist)
freq_df= pd.DataFrame(freq.items())
freq_df=freq_df.sort_values(by=[1],ascending=False)


jkl= freq_df.loc[:10,:]

values = np.array(jkl[0])

indexes = np.array(jkl[1])

bar_width = 5.35

plt.bar(values,indexes)


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['Comment Text'],df['sentiment'],test_size=0.1)


Train_Y.value_counts()

Test_Y.value_counts()

def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None) 

tfidf.fit(df['Comment Text'])

tfidf.transform(df['Comment Text'])

Train_X_Tfidf = tfidf.transform(Train_X)
Test_X_Tfidf = tfidf.transform(Test_X)

# SVM
from sklearn.metrics import accuracy_score
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',class_weight={'1':1,'0':1,'2': 4})
SVM.fit(Train_X_Tfidf,Train_Y)

predictions_SVM = SVM.predict(Test_X_Tfidf)


print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

from sklearn.metrics import classification_report
print(classification_report(Test_Y,predictions_SVM))

y =pd.DataFrame(Test_X).reset_index()
z=pd.DataFrame(predictions_SVM).reset_index()
r=pd.merge(y,z,left_index=True,right_index=True)
r.drop('index_y',axis =1,inplace=True)
t=pd.DataFrame(Test_Y).reset_index()
compar_df=pd.merge(t,r,left_index=True,right_index=True)


compar_df.loc[compar_df['sentiment']=='2']
