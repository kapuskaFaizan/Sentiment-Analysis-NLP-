#!/usr/bin/env python
# coding: utf-8


import re
import pandas as pd
from nltk.stem.porter import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

csv = pd.read_csv('C:/Users/faiza/OneDrive/Desktop/data_sets/tweets.csv')

df = pd.DataFrame(csv)

df1=df[['text','airline_sentiment']]


df1.replace('@[\S]+', '', regex =True, inplace = True)
df1.replace('#(\S+)', ' \1 ', regex =True, inplace = True)#hashta
df1.replace('((www\.[\S]+)|(https?://[\S]+))', ' URL ', regex =True, inplace = True)
df1.replace('\.{2,}', ' ', regex =True, inplace = True) #2=dots with space

df1.replace('\brt\b', '', regex =True, inplace = True)# retweet removal
df1.replace('((www\.[\S]+)|(https?://[\S]+))', ' URL', regex =True, inplace = True)

df1['airline_sentiment'].replace('neutral', int('0'), regex =True, inplace = True)
df1['airline_sentiment'].replace('positive', int('1'), regex =True, inplace = True)
df1['airline_sentiment'].replace('negative', int('-1'), regex =True, inplace = True)



tokenized_tweet = df1['text'].apply(lambda x: x.split())
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_tweet.head()



all_words = ' '.join([text for text in df1['text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


negative_words = ' '.join([text for text in df1['text'][df1['airline_sentiment'] == 0]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=100).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



normal_words =' '.join([text for text in df1['text'][df1['airline_sentiment'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
bow = pad_sequences(sequences, maxlen=50)



train_bow = bow[:31962,:]
test_bow = bow[31962:,:]


xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, df1['airline_sentiment'], random_state=42, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain_bow, ytrain)


y_pred= knn.predict(xvalid_bow)

print(accuracy_score(yvalid,y_pred)*100)
print(confusion_matrix(yvalid,y_pred))
print(classification_report(yvalid,y_pred))


