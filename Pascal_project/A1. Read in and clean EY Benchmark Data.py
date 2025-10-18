# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:09:12 2017

@author: qings
"""

import pandas as pd
import numpy as np
import time
import datetime
import os
from unidecode import unidecode
import nltk

start_time = time.clock()

stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords_noaccent = [unidecode(w) for w in stopwords]

os.chdir("C:\\Users\\qingch\\Desktop\\Foundry") #path for EY Benchmark Data output after data cleansing

file = "C:\\Users\\qingch\\Desktop\\Foundry\\EY Benchmark Data\\EY Benchmark Data.xlsx" #Raw data file

df = pd.read_excel(file, sheet_name='Sheet1')

df['TARIFF CODE'] = df['TARIFF CODE'].astype('str')
df['TARIFF CODE'] = df['TARIFF CODE'].apply(lambda x: '0'+x if len(x)==7 else x)
df['DROP'] = df['TARIFF CODE'].apply(lambda x: 1 if len(x)!=8 else 0)
df = df.copy()[df['DROP']==0]
df['PRODUCT DESCRIPTION'] = df['PRODUCT DESCRIPTION'].astype('str')


PD = df['PRODUCT DESCRIPTION'].str.lower()
PD = PD.apply(unidecode)
PD = PD.str.replace('\'', '', case=False)
PD = PD.str.replace('[^A-Za-z0-9\s]+', ' ', case=False) #remove special characters
PD = PD.str.replace('\d+\.?\d*', '', case=False) # remove numbers
PD = PD.str.replace(r'\b\w{1}\b', '', case=False)          
PD = PD.str.split() # split descriptions into words
PD = PD.apply(lambda x: [w for w in x if not w in stopwords_noaccent]) # remove stopwords
PD = PD.apply(lambda x: " ".join(x)) 

df.drop(['PRODUCT DESCRIPTION','DROP'], axis=1, inplace=True) 
df['PRODUCT DESCRIPTION'] = PD
df2 = df.copy()[df['PRODUCT DESCRIPTION'].notnull()]
df2.to_pickle('EY Benchmark Data.pkl')

print('Total time cost:', str(datetime.timedelta(seconds=round(time.clock()-start_time,0))))