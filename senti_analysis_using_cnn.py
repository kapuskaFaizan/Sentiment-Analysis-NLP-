#!/usr/bin/env python
# coding: utf-8


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
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
data = [x[::-1] for x in data]
data = pad_sequences(sequences, maxlen=50)

# Network architecture
model = Sequential()
model.add(Embedding(20000, 100, input_length=50))
model.add(Conv1D(32,7,activation='relu'))
model.add(MaxPooling1D(5)
model.add(Conv1D(32,7,activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(data, np.array(labels), validation_split=0.2, epochs=5)

