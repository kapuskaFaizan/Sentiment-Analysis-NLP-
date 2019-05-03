#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

from sklearn.manifold import TSNE
import re
import pandas as pd

csv = pd.read_csv('C:/Users/faiza/OneDrive/Desktop/data_sets/tweets.csv')
df = pd.DataFrame(csv)
df1=df[['text','airline_sentiment']]

df1.replace('@[\S]+', 'USER', regex =True, inplace = True)
df1.replace('#(\S+)', ' \1 ', regex =True, inplace = True)#hashtag
df1.replace('((www\.[\S]+)|(https?://[\S]+))', ' URL ', regex =True, inplace = True)
df1.replace('\.{2,}', ' ', regex =True, inplace = True) #2=dots with space

df1.replace('\brt\b', '', regex =True, inplace = True)# retweet removal
df1.replace('((www\.[\S]+)|(https?://[\S]+))', ' URL', regex =True, inplace = True)

df1['airline_sentiment'].replace('neutral', int('0'), regex =True, inplace = True)
df1['airline_sentiment'].replace('positive', int('1'), regex =True, inplace = True)
df1['airline_sentiment'].replace('negative', int('-1'), regex =True, inplace = True)

labels = df1['airline_sentiment']


# In[3]:


vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)
print(data)


# In[4]:


from keras.layers.recurrent import LSTM, GRU, SimpleRNN

model = Sequential()
model.add(Embedding(vocabulary_size, 10, input_length=50))
model.add(SimpleRNN(10))  

model.add(Dense(50, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(50, activation ='relu'))
model.add(Dense(1, activation ='linear'))


model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.summary()


# In[6]:


model.fit(data, np.array(labels), validation_split=0.1, epochs=5)


# In[7]:


x=model.predict(data)


# In[9]:


x=pd.DataFrame(x)


# In[18]:


x=x.round().reset_index()
y=pd.DataFrame(labels).reset_index()
r=pd.merge(x,y,left_index=True,right_index=True)
r[0] = r[0].astype(int)


# In[33]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(r['airline_sentiment'],r[0]))


# In[34]:


from sklearn.metrics import classification_report
print(classification_report(r['airline_sentiment'],r[0]))


# In[35]:


from sklearn.metrics import accuracy_score
print(accuracy_score(r['airline_sentiment'],r[0])*100)


# In[ ]:




