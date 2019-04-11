#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import re
import pandas as pd

csv = pd.read_csv('C:/Users/faiza/OneDrive/Desktop/data_sets/tweets.csv')

df = pd.DataFrame(csv)

df1=df[['text','airline_sentiment']]


df1.replace('@[\S]+', '', regex =True, inplace = True)
df1.replace('#(\S+)', ' \1 ', regex =True, inplace = True)#hashtag
df1.replace('((www\.[\S]+)|(https?://[\S]+))', ' URL ', regex =True, inplace = True)
df1.replace('\.{2,}', ' ', regex =True, inplace = True) #2=dots with space

df1.replace('\brt\b', '', regex =True, inplace = True)# retweet removal
df1.replace('((www\.[\S]+)|(https?://[\S]+))', ' URL', regex =True, inplace = True)

df1['airline_sentiment'].replace('neutral', int('0'), regex =True, inplace = True)
df1['airline_sentiment'].replace('positive', int('1'), regex =True, inplace = True)
df1['airline_sentiment'].replace('negative', int('-1'), regex =True, inplace = True)

from nltk.tokenize import word_tokenize
tokenized_tweet = df1['text'].apply(lambda x: x.split() )


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()
.
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

df1['text'] = tokenized_tweet

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vectorizer.fit_transform(df1['text'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]


x_train, x_test, y_train, y_test = train_test_split(train_bow, df1['airline_sentiment'], random_state=42, test_size=0.3)

print( bow)


# In[2]:


classifr = tree.DecisionTreeClassifier()
classifr.fit(x_train,y_train)
acc_train = np.sum(classifr.predict(x_train) == y_train)/ float(y_train.size)
acc_test = np.sum(classifr.predict(x_test) == y_test)/ float(y_test.size)


# In[3]:


print('classification accuracy for train',acc_train)
print('classification accuracy for test',acc_test)


# In[4]:


y_pred=classifr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred)*100)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))

