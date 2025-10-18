# -*- coding: utf-8 -*-
"""
@author: Eugene Yankovsky 

"""

import pandas as pd
import numpy as np
import time
import datetime
import os
from unidecode import unidecode
import nltk

start_time = time.clock()

os.chdir("C:\\Users\\")
path = "C:\\Users\\"

code_count = []
record_list = []
file_list = []

stopwords = nltk.corpus.stopwords.words('portuguese')
stopwords_noaccent = [unidecode(w) for w in stopwords]

for i in range(97):
    files = os.listdir(path+"\\" + str(i+1) + "_Capitulo")
    if (i != 76):
        for f in files:
            full_f = path+"\\" + str(i+1) + "_Capitulo" + "\\" + f
            data = pd.read_csv(full_f, delimiter='@', error_bad_lines=False, engine='c', encoding = 'ISO-8859-1', low_memory=False)
            data.columns = ['ORDER NUMBER', 'YEAR MONTH', 'COD.NCM', 'DESCRIPTION OF THE NCM CODE',
                            'ORIGIN COUNTRY CODE', 'COUNTRY OF ORIGIN', 'ACQUISITION COUNTRY CODE', 
                            'COUNTRY OF ACQUISITION', 'UNIT STATISTIC', 'UNIT OF MEASUREMENT', 
                            'COMMERCIAL UNIT', 'PRODUCT DESCRIPTION', 'QUANTITY STATISTICS',
                            'NET WEIGHT', 'VMLE DOLLAR', 'VL FREIGHT DOLLAR', 'VL INSURANCE DOLLAR',
                            'UNIT PRODUCT DOLLAR VALUE', 'COMERCIAL QUANTITY', 'TOTAL UNIT PRODUCT DOLLAR',
                            'LANDING UNIT', 'RELEASE UNIT', 'INTERNATIONAL COMERCIAL TERMS',
                            'NAT INFORMATION', 'SITUATION OF THE OFFICE']
            
            data['TARIFF CODE'] = data['COD.NCM'].astype('str')
            data = data[(-data['TARIFF CODE'].str.contains('[^0-9]')) & data['PRODUCT DESCRIPTION'].notnull()]
            if i<9:
                data['TARIFF CODE'] = data['TARIFF CODE'].apply(lambda x: '0'+x)
            data2 = data.copy()[['TARIFF CODE', 'PRODUCT DESCRIPTION']]
            del data
            
            PD = data2['PRODUCT DESCRIPTION'].str.lower() # change everything to lowercase
            PD = PD.apply(unidecode)
            PD = PD.str.replace('[^A-Za-z0-9\s]+', ' ', case=False) #remove special characters
            PD = PD.str.replace('\d+\.?\d*', '', case=False) # remove numbers          
            PD = PD.str.replace(r'\b\w{1}\b', '', case=False) # remove words with only one character
            PD = PD.str.split() # split descriptions into words
#            PD = PD.apply(lambda x: [stemmer.stem(w) for w in x]) # stemming and lemmatize
            PD = PD.apply(lambda x: [w for w in x if not w in stopwords_noaccent]) # remove stopwords
            PD = PD.apply(lambda x: " ".join(x))           

            data2.drop(['PRODUCT DESCRIPTION'], axis=1, inplace=True)                        
            data2['PRODUCT DESCRIPTION'] = PD
            data3 = data2.copy()[data2['PRODUCT DESCRIPTION'].notnull()]
            del data2
            record_list.append(data3)            
            file_list.append(f)
            print(f)

print('Total time cost:', str(datetime.timedelta(seconds=round(time.clock()-start_time,0))))

           

start_time2 = time.clock() 
df_full_data = pd.concat(record_list)
print('Total time cost:', str(datetime.timedelta(seconds=round(time.clock()-start_time2,0))))
df_full_data.to_pickle('IRS Data.pkl')
