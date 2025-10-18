# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:53:44 2017

@author: Eugene Yankovsky
"""

import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import os
from unidecode import unidecode
import time
import datetime

start_time = time.clock()
os.chdir("C:\\")
df_train = pd.read_pickle('Training Set.pkl')
df_test = pd.read_pickle('Testing Set.pkl')

split=[]
for i in range(10):
    split.append(round((i+1)*df_test.shape[0]/10)-1)    

precision_threshold = 0.9  # setting precision threshold to create low precision code list

corpus_train = df_train['PRODUCT DESCRIPTION'].astype('str')
train_label = df_train['TARIFF CODE'].astype('str')


def get_metrics(true_labels, predicted_labels):
    print('Accuracy', np.round(metrics.accuracy_score(true_labels, predicted_labels), 4))
    print('Precision', np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 4))
    print('Recall', np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted') ,4))
    print('F1 Score', np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 4))
    

def tfidf_extractor(corpus, ngram_range=(1,2)):
    
    vectorizer = TfidfVectorizer(min_df=1, 
                                 norm='l2', 
                                 max_features=300000, 
                                 smooth_idf=True, 
                                 use_idf=True, 
                                 ngram_range=ngram_range)   
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

tfidf_vectorizer, tfidf_features = tfidf_extractor(corpus_train)

alpha = 0.0000006
svm = SGDClassifier(loss='hinge', alpha=alpha, verbose=0, max_iter=1)
svm.fit(tfidf_features, train_label)

test_list=[]
for s in range(10):
    if s==0:
        df = df_test.iloc[:split[s],]
    else:
        df = df_test.iloc[split[s-1]:split[s],]
    corpus_test = df['PRODUCT DESCRIPTION'].astype('str')
    nd_tfidf = tfidf_vectorizer.transform(corpus_test)
    predictions = svm.predict(nd_tfidf)
    df['PREDICTION'] = predictions
    test_list.append(df)

df_prediction = pd.concat(test_list)
get_metrics(true_labels=df_prediction['TARIFF CODE'], predicted_labels=df_prediction['PREDICTION'])
df_prediction.to_excel('Prediction.xlsx', sheet_name='Prediction', index=False)

df_prediction2 = df_prediction.copy()[df_prediction['SOURCE']=='EY']

df_prediction2['Indicator'] = (df_prediction2['PREDICTION']==df_prediction2['TARIFF CODE'])
df_prediction2['Indicator'] =df_prediction2['Indicator'].astype('int')
code_count = df_prediction2.groupby(['PREDICTION']).size().rename('COUNT').reset_index()
num_of_correct = df_prediction2.groupby(['PREDICTION'])['Indicator'].agg('sum').reset_index()

merged = code_count.merge(num_of_correct, how='outer', left_on='PREDICTION', right_on='PREDICTION')
merged['PRECISION'] = merged['Indicator']/merged['COUNT']
code_list = merged[merged['PRECISION']<precision_threshold]
code_list.to_excel('Low Precision Code List.xlsx', sheet_name='Sheet1')

print('Total time cost:', str(datetime.timedelta(seconds=round(time.clock()-start_time,0))))
    
