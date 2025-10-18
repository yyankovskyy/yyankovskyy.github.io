# -*- coding: utf-8 -*-
"""

@author: Eugene Yankovsky 

# Diagnostic steps are commented out
"""

import pandas as pd
import numpy as np
import pickle
import os
from unidecode import unidecode

S = 625 # 625 Number of obsrevations per a tariff code to be set upfront by a developer 
S_upper = 100 # Threshold number for a tariff code considered as a separate category and present only in IRS public data
S_lower =  5 # Threshold number for a tariff code considered as a separate category and present in a joint data of EY banchmared and IRS public data 
train_share = 0.8
train_size = np.round(train_share*S)
test_size = S - train_size
print(train_size, test_size)


# Loading necessary data from the public IRS data 

os.chdir("C:\\Brazil Custom Case\\Sample Data\\EY_sampling")

df = pickle.load(open("Full Data.pkl", "rb"))
# df.head(10)
#len(df)
#print(df.columns)
#print(df.dtypes)


df['TARIFF_CODE'] = df['CLEANED PRODUCT DESCRIPTION'].str.replace(r'\D+', '').astype(str)
df = df.rename(columns={'CLEANED PRODUCT DESCRIPTION': 'PRODUCT_DESCRIPTION'})
df['PRODUCT_DESCRIPTION'] = df['PRODUCT_DESCRIPTION'].str.replace('\d+\.?\d*', '', case=False)

df["TARIFF_CODE_count"] = df["TARIFF_CODE"].astype(str).str.len()
df["TARIFF_CODE"] = np.where(df["TARIFF_CODE_count"]==7,'0'+df["TARIFF_CODE"].astype(str),df["TARIFF_CODE"].astype(str)) 


ds = df.sample(frac=1, random_state =111)
#print(df.head(5))


ds["TARIFF_CODE"] = ds.index
dfs = ds[['TARIFF_CODE', 'PRODUCT_DESCRIPTION']].groupby('TARIFF_CODE').head(S).reset_index(drop=True)

dfs = dfs.sort_values(['TARIFF_CODE'],ascending=[1])

# print(dfs)
# print(dfs["TARIFF_CODE"].unique())
# print(len(dfs["TARIFF_CODE"].unique()))

# Loading clean EY banchmarked data
file = 'cleaned client data.xlsx'

df_ey = pd.read_excel(file, sheetname='Sheet1', usecols=['TARIFF CODE','CLEANED PRODUCT DESCRIPTION'],
                      dtypes = {'TARIFF CODE': np.str,'CLEANED PRODUCT DESCRIPTION': np.str})

df_ey =df_ey.rename(columns={'TARIFF CODE' : 'TARIFF_CODE','CLEANED PRODUCT DESCRIPTION': 'PRODUCT_DESCRIPTION'})

df_ey.head()
# print('EY banchmarked data size is', len(df_ey))

df_ey["PRODUCT_DESCRIPTION"] = df_ey["PRODUCT_DESCRIPTION"].astype(str)
df_ey['PRODUCT_DESCRIPTION']  = df_ey['PRODUCT_DESCRIPTION'].apply(unidecode)
df_ey['PRODUCT_DESCRIPTION']  = df_ey['PRODUCT_DESCRIPTION'].str.replace('[^A-Za-z0-9\s]+', ' ', case=False) #remove special characters
df_ey['PRODUCT_DESCRIPTION'] = df_ey['PRODUCT_DESCRIPTION'].str.replace('\d+\.?\d*', '', case=False) # remove numbers          
df_ey['PRODUCT_DESCRIPTION'] = df_ey['PRODUCT_DESCRIPTION'].str.replace(r'\b\w{1}\b', '', case=False) # remove words with only one character
df_ey['PRODUCT_DESCRIPTION'] = df_ey['PRODUCT_DESCRIPTION'].str.strip()
# print(df_ey['PRODUCT_DESCRIPTION'])

###      Removing the poor description data   
file = 'PoorDescription_LR_SD.xlsx'

ctn = pd.read_excel(file, sheetname='Sheet1', usecols=['TARIFF CODE','PRODUCT DESCRIPTION', 'EY BR comments'],
                      dtypes = {'TARIFF CODE': np.str,'PRODUCT DESCRIPTION': np.str, 'EY BR comments': np.str})


#print(ctn.head())

ctn = ctn[ctn["EY BR comments"].str.strip() == 'poor']
# print('Count of poor description obs is', len(ctn))
# print(ctn.head())

ctn["PRODUCT DESCRIPTION"] = ctn["PRODUCT DESCRIPTION"].astype(str)


ctn["PRODUCT DESCRIPTION"] = ctn["PRODUCT DESCRIPTION"].apply(unidecode)
ctn["PRODUCT DESCRIPTION"] = ctn["PRODUCT DESCRIPTION"].str.replace('[^A-Za-z0-9\s]+', ' ', case=False) #remove special characters
ctn["PRODUCT DESCRIPTION"] = ctn["PRODUCT DESCRIPTION"].str.replace('\d+\.?\d*', '', case=False) # remove numbers          
ctn["PRODUCT DESCRIPTION"] = ctn["PRODUCT DESCRIPTION"].str.replace(r'\b\w{1}\b', '', case=False) # remove words with only one character
ctn['PRODUCT DESCRIPTION'] = ctn['PRODUCT DESCRIPTION'].str.strip()


ctn =ctn.rename(columns={'TARIFF CODE' : 'TARIFF_CODE','PRODUCT DESCRIPTION': 'PRODUCT_DESCRIPTION'})


print('EY banchmarked data size', len(df_ey))

print(df_ey[df_ey["PRODUCT_DESCRIPTION"] =="lcool"])
print(ctn[ctn["PRODUCT_DESCRIPTION"] =="lcool"])



# df_ey["PRODUCT_DESCRIPTION"] = df_ey["PRODUCT_DESCRIPTION"].astype(str)
# ctn["PRODUC_ DESCRIPTION"] = ctn["PRODUCT_DESCRIPTION"].astype(str)
#df_eyc = df_ey.merge(right=ctn, how='outer', right_on="PRODUCT_DESCRIPTION", left_on="PRODUCT_DESCRIPTION",right_index=True, sort=False)

df_eyc = pd.merge(right=ctn, left=df_ey, how='left', right_on="PRODUCT_DESCRIPTION", left_on="PRODUCT_DESCRIPTION")

# print(len(df_eyc[df_eyc["EY BR comments"] =="poor"]))
# print(df_eyc[df_eyc["EY BR comments"] =="poor"])

df_eyc = df_eyc[df_eyc["EY BR comments"] !="poor"]

# print(df_eyc.head())
# print(len(df_ey))
# print(len(ctn))
# print(len(df_eyc))

df_eyc =df_eyc.rename(columns={'TARIFF_CODE_x' : 'TARIFF_CODE'})
df_eyc["TARIFF_CODE"] = df_eyc["TARIFF_CODE"].astype(str)

df_eyc["TARIFF_CODE"].str.strip()

df_eyc.head(10)

### Cleaning is over 

### Enriching the data with public data
#################################################################
# Random sampling and S-capping for EY banchmarked data
ds_ey = df_ey[["TARIFF_CODE", "PRODUCT_DESCRIPTION"]].sample(frac=1, random_state = 112)

# print(ds_ey.head(5))

ds_ey["TARIFF_CODE_count"] = ds_ey["TARIFF_CODE"].astype(str).str.len()
ds_ey["TARIFF_CODE"] = np.where(ds_ey["TARIFF_CODE_count"]==7,'0'+ds_ey["TARIFF_CODE"].astype(str),ds_ey["TARIFF_CODE"].astype(str)) 
# print(ds_ey.head(5))

#print(dfs_ey.head(5))
#print(dfs_ey.head(5))
#dfs_ey["TARIFF_CODE_count"] = dfs_ey["TARIFF_CODE"].astype(str).str.len()


ds_ey.groupby("TARIFF_CODE").agg({"TARIFF_CODE":[np.size]})

dfs_ey = ds_ey.groupby('TARIFF_CODE').head(S).reset_index(drop=True)
dfs_ey.groupby("TARIFF_CODE").agg({"TARIFF_CODE":[np.size]})


# print(dfs_ey.head(5))


# Merging EY benchmarked data with public data
dfs_ey['Source'] = 'EY'
dfs['Source'] = 'IRS'
# dfs_ey.groupby("TARIFF_CODE").agg({"TARIFF_CODE":[np.size]})
# dfs_ey.groupby("Source").agg({"Source":[np.size]})

#dfs_ey = dfs_ey['TARIFF_CODE'].astype(str)
dfj = pd.concat([dfs_ey, dfs], join='outer',  ignore_index=True)

### Additional enrichment data peace

freq = pd.DataFrame({'frequency' : dfj.groupby(["TARIFF_CODE"]).size()}).reset_index()
# print(freq.head())
# print(freq.dtypes)


dfjf = pd.merge(right = freq , left=dfj, how='left', right_on="TARIFF_CODE", left_on="TARIFF_CODE")
# print(dfjf['frequency'].describe())


dfjf["CODE"] = np.where((dfjf["frequency"]>=S_lower) & ((dfjf["Source"] == 'EY') | (dfjf["frequency"]>=S_upper)), dfjf["TARIFF_CODE"], "Other")

# print(len(dfjf["TARIFF_CODE"].unique()))
# print(len(dfjf["CODE"].unique()))
# print(len(dfj["CODE_1"].unique()))

# print(dfjf["Source"].unique())
# print(dfjf.head())

dfj_1 = dfjf[['CODE', 'PRODUCT_DESCRIPTION', 'Source']].groupby('CODE').head(S).reset_index(drop=True)
dfj_1 = dfj_1.rename(columns={'CODE' : 'TARIFF_CODE'})

# print(len(dfj_1['TARIFF_CODE'].unique()))

dfj_1  = dfj_1 .sort_values(['TARIFF_CODE'],ascending=[1])

# print(dfj_1.head())
# print(len(dfj_1))


### Boostrapping 
#https://stackoverflow.com/questions/33097167/sampling-a-dataframe-based-on-a-given-distribution
sample = pd.DataFrame()

for i in dfj_1['TARIFF_CODE'].unique(): 
    dfa = dfj_1[dfj_1['TARIFF_CODE']==i]
    dfasample = dfa.sample(n=S, random_state = 113, replace=True)
    sample = pd.concat([sample, dfasample], join='outer',  ignore_index=True)

# print(sample.groupby('TARIFF_CODE').agg({'TARIFF_CODE':[np.size]}))
# print('Number of observations', len(sample))
# print('Number of tariff codes', len(sample['TARIFF_CODE'].unique()))
# print('Number of obs by source', sample.groupby("Source").agg({"Source":[np.size]}))


# Original code
# train_file = sample[['TARIFF_CODE', 'PRODUCT DESCRIPTION','Source']].groupby('TARIFF_CODE').head(train_size).reset_index(drop=True)
# test_file = sample[['TARIFF_CODE', 'PRODUCT DESCRIPTION', 'Source']].groupby('TARIFF_CODE').tail(test_size).reset_index(drop=True)

# Cleaned code to match the input requirement into Charlie's model
sample = sample.rename(columns={'TARIFF_CODE': 'CODE', 'PRODUCT_DESCRIPTION': 'PRODUCT DESCRIPTION' })
train_file = sample[['CODE', 'PRODUCT DESCRIPTION','Source']].groupby('CODE').head(train_size).reset_index(drop=True)
test_file = sample[['CODE', 'PRODUCT DESCRIPTION', 'Source']].groupby('CODE').tail(test_size).reset_index(drop=True)

# print(sample.groupby("TARIFF_CODE").agg({"TARIFF_CODE":[np.size]}))
# print(test_file.groupby('TARIFF_CODE').agg({'TARIFF_CODE':[np.size]}))
# print(train_file.groupby('TARIFF_CODE').agg({'TARIFF_CODE':[np.size]}))
# print(len(sample))
# print(len(test_file))
# print(len(train_file))

os.chdir("C:\\Brazil Custom Case\\Sample Data")

sample.to_pickle('whole_file_' + str(S) + '_obs'+ '.pkl')
train_file.to_pickle('train_file_' + str(S) + '_obs'+ '.pkl')
test_file.to_pickle('test_file_' + str(S) + '_obs' + '.pkl')


# Exclude duplicates
sample_nodup = sample.drop_duplicates()
train_file_nodup = train_file.drop_duplicates()
test_file_nodup = test_file.drop_duplicates()

# print(len(sample_nodup))
# print(len(train_file_nodup))
# print(len(test_file_nodup))

sample_nodup.to_pickle('whole_file_nodup' + str(S) + '_obs'+ '.pkl')
train_file_nodup.to_pickle('train_file_nodup' + str(S) + '_obs'+ '.pkl')
test_file_nodup.to_pickle('test_file_nodup' + str(S) + '_obs' + '.pkl')