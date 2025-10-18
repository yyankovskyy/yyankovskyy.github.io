
# coding: utf-8

# In[1]:


import math
import io
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from datetime import datetime as dt
from itertools import islice
import sys
import statsmodels.formula.api as smf
import numpy as np

import statsmodels.api as sm

if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 100000
pd.options.display.float_format = '{:.3f}'.format
import re 

path = '/data/' 

# Reading from the field without extension
from os import listdir
from os.path import isfile, join



# In[2]:


onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
print 'Number of files in the location is ', len(onlyfiles)
print onlyfiles


# In[9]:


def diff(data,var):
    data[var] =np.where(data[var]==0, np.nan, data[var])
    var_1=var+"_1"
    data[var_1]=data[var].shift()
    return data[var_1]


# In[10]:


def arraytransform(path0,file0):
    file_in = path+file0
    #print file_in
    npy1 = np.load(file_in)
    
    column_list=['frame',"Nose_x", "Neck_x", "RShoulder_x", "RElbow_x","RWrist_x","LShoulder_x","LElbow_x","LWrist_x","RHip_x","RKnee_x","RAnkle_x","LHip_x","LKnee_x","LAnkle_x","REye_x","LEye_x","REar_x","LEar_x","Nose_y","Neck_y","RShoulder_y","RElbow_y","RWrist_y","LShoulder_y","LElbow_y","LWrist_y","RHip_y","RKnee_y","RAnkle_y","LHip_y","LKnee_y","LAnkle_y","REye_y","LEye_y","REar_y","LEar_y","Nose_c","Neck_c","RShoulder_c","RElbow_c","RWrist_c","LShoulder_c","LElbow_c","LWrist_c","RHip_c","RKnee_c","RAnkle_c","LHip_c","LKnee_c","LAnkle_c","REye_c","LEye_c","REar_c","LEar_c"]
    df = pd.DataFrame(npy1, columns=column_list) 
    
    frame_min =np.min(df["frame"].astype(int))
    frame_max = np.max(df["frame"].astype(int))
    #print frame_min, frame_max
    
    data = range(frame_min, frame_max+1)
    stdt = pd.DataFrame(data, columns={'frame'})
    #print stdt

    df_fin= stdt.merge(df, how='outer')
    col_list=list(df_fin.columns)
    
    for idx,joint in enumerate(col_list):
        diff(data=df_fin, var=joint)
    
    df_fin.drop(["frame_1", 'Nose_c', 'Neck_c', 'RShoulder_c', 'RElbow_c', 'RWrist_c', 'LShoulder_c', 'LElbow_c', 'LWrist_c', 'RHip_c', 'RKnee_c', 'RAnkle_c', 'LHip_c', 'LKnee_c', 'LAnkle_c', 'REye_c', 'LEye_c', 'REar_c', 'LEar_c',
            'Nose_c_1', 'Neck_c_1', 'RShoulder_c_1', 'RElbow_c_1', 'RWrist_c_1', 'LShoulder_c_1', 'LElbow_c_1', 'LWrist_c_1', 'RHip_c_1', 
             'RKnee_c_1', 'RAnkle_c_1', 'LHip_c_1', 'LKnee_c_1', 'LAnkle_c_1', 'REye_c_1', 'LEye_c_1', 'REar_c_1', 'LEar_c_1'], axis=1, inplace=True)
    return df_fin


# In[378]:


data=pd.DataFrame()


# In[379]:


for i in range(len(onlyfiles)):
#for i in range(0, 4):
    file_out= "out_"+str(i)
    print onlyfiles[i]
    #print file_out
    file_out = arraytransform(path0=path ,file0 =onlyfiles[i])
    print file_out.shape
    #print file_out.head(5)
    data = data.append(file_out)
print 'data shape', data.shape    
print data.head(5)
    


# In[380]:


data.index = pd.RangeIndex(len(data.index))


# In[381]:


# Check for NaN values
print "Total number of missing values in the sample",
data.isnull().values.ravel().sum()


# In[382]:


print "Number of missing values before imputation in Neck_x :", data[['Neck_x']].isnull().values.ravel().sum()
print "Number of missing values before imputation in Neck_y :", data[['Neck_y']].isnull().values.ravel().sum()
print "Number of missing values before imputation in Neck_x_1 :", data[['Neck_x_1']].isnull().values.ravel().sum()
print "Number of missing values before imputation in Neck_y_1 :", data[['Neck_y_1']].isnull().values.ravel().sum()


# In[420]:


def AR1x(var, var_out, data0):
    var_1=var+"_1"
    var_predicted_df =var+"_predicted_df" 
    Y_train = data0[var]
    X_train = data0[var_1]
    #X_train = sm.add_constant(X_train) # No intercept in the equation
    res = sm.OLS(Y_train, X_train, missing="drop").fit()

    # Inspect the results
    print(res.summary())
    
    var_predicted=var_out
    var_predicted_df=var+"_predicted_df"
    var_predicted_df =pd.DataFrame(res.params[0]*X_train)
    var_predicted_df.columns=[var_predicted]
    print "Predicted table shape:", var_predicted_df.shape
    data0 = pd.concat([data0,var_predicted_df], axis=1, join_axes=[data0.index])
    print data0[[var, var_1,var_predicted]].head(10) 
    return data0


# In[384]:


data = AR1x(var="Neck_x",var_out ="Neck_x_AR1x_predicted",  data0=data)
data = AR1x(var="Neck_y", var_out ="Neck_y_AR1y_predicted", data0=data)
data.head(5)


# 1.1. If Neck_x and Neck_y are missing and Neck_x_1 is not missing, impute the values:
# data[‘Neck_x’] = f([‘Neck_x_1’])

# In[385]:


print "Number of cases Neck_x and Neck_y are missing and Neck_x_1 is not missing:", len(data[(data.loc[:,'Neck_x']==np.nan) & (data.loc[:,'Neck_y']==np.nan) & (data.loc[:,'Neck_x_1']!=np.nan)])


# In[ ]:


data[['Neck_x', 'Neck_x_1','Neck_y', 'Neck_y_1']].describe()


# In[386]:


data['Neck_x'] =np.where((data.loc[:,'Neck_x']==np.nan) & (data.loc[:,'Neck_y']==np.nan) & (data.loc[:,'Neck_x_1']!=np.nan) , data["Neck_x_AR1x_predicted"],data["Neck_x"])


# 1.2. If Neck_x and Neck_y are missing and Neck_y_1 is not missing, impute the values:
# data[‘Neck_y’] = f([‘Neck_y_1’])

# In[387]:


print "Number of cases Neck_x and Neck_y are missing and Neck_y_1 is not missing:", len(data[(data.loc[:,'Neck_x']==np.nan) & (data.loc[:,'Neck_y']==np.nan) & (data.loc[:,'Neck_y_1']!=np.nan)])


# In[388]:


data['Neck_y'] =np.where((data.loc[:,'Neck_y']==np.nan) & (data.loc[:,'Neck_y']==np.nan) & (data.loc[:,'Neck_y_1']!=np.nan), data["Neck_y_AR1y_predicted"],data['Neck_y'])


# 1.3. if Neck_x is missing (i.e, 0), impute missing values with:
# data[‘Neck_x’] =f('Neck_x_1’,'Neck_y','Neck_y_1')

# In[389]:


def AR1xy(var, var2, var_out, data0):
    
    var_1=var+"_1"
    var2_1=var2+"_1"

    Y_train = data0[var]
    X_train = data0[[var_1, var2, var2_1]]
        
    res = sm.OLS(Y_train, X_train, missing="drop").fit()

    # Inspect the results
    #print(res.summary())
    #print('Parameters: ', res.params)

    var_predicted_df=var+"_predicted_df"
    
    var_predicted_df =pd.DataFrame(res.params[0]*X_train[var_1]+res.params[1]*X_train[var2]+res.params[2]*X_train[var2_1])
    var_predicted_df.columns=[var_out]
    print "Predicted table shape:", var_predicted_df.shape
    data0 = pd.concat([data0,var_predicted_df], axis=1, join_axes=[data0.index])
    print data0[[var_1,var2, var2_1,var_out]].head(10) 
    return data0
    


# In[390]:


data=AR1xy(var="Neck_x",var2="Neck_y", var_out ="Neck_x_AR1xy_predicted",  data0=data)


# In[391]:


data = AR1xy(var="Neck_y",var2="Neck_x", var_out ="Neck_y_AR1xy_predicted", data0=data)


# In[392]:


data['Neck_x'] =np.where((data.loc[:,'Neck_x']==np.nan), data["Neck_x_AR1xy_predicted"],data["Neck_x"])


# 1.4. if Neck_x is missing (i.e, 0), impute missing values with:
# data[‘Neck_y’] =f('Neck_y_1’,'Neck_x','Neck_x_1')

# In[393]:


data['Neck_y'] =np.where((data.loc[:,'Neck_y']==np.nan), data["Neck_y_AR1xy_predicted"],data["Neck_y"])


# In[394]:


# Limiting min=0 
data['Neck_x'] =data['Neck_x'].apply(lambda x: max(0, x))
data['Neck_y'] =data['Neck_y'].apply(lambda x: max(0, x))


# In[395]:


# Limiting max=1
data['Neck_x'] =data['Neck_x'].apply(lambda x: min(1, x))
data['Neck_y'] =data['Neck_y'].apply(lambda x: min(1, x))


# In[396]:


data['Neck_x_1']=data['Neck_x'].shift()
data['Neck_y_1']=data['Neck_y'].shift()


# In[397]:


print "Number of missing values after imputation in Neck_x :", data[['Neck_x']].isnull().values.ravel().sum()
print "Number of missing values after imputation in Neck_y :", data[['Neck_y']].isnull().values.ravel().sum()
print "Number of missing values after imputation in Neck_x_1 :", data[['Neck_x_1']].isnull().values.ravel().sum()
print "Number of missing values after imputation in Neck_y_1 :", data[['Neck_y_1']].isnull().values.ravel().sum()


# In[432]:


data[['Neck_x', 'Neck_x_1','Neck_y', 'Neck_y_1']].describe()


# 2.1. If  data[“RShoulder_x”]  is missing (i.e, 0) and data[“RShoulder_x_1”] is not missing, impute missing values with:
# 
# E[“RShoulder_x”] = f(‘RShoulder_x_1”)
# 
# 2.2. If  both data[“RShoulder_x”]  is missing (i.e, 0), impute the values with:
# E["RShoulder_x”] = f(“Neck_x”, “Neck_x_1” , “Neck_y”, “Neck_y_1”)

# In[348]:


print "Number of missing values before imputation:", data[['RShoulder_x']].isnull().values.ravel().sum()


# In[398]:


data = AR1x(var="RShoulder_x",var_out ="RShoulder_x_AR1x_predicted",  data0=data)


# In[401]:


data['RShoulder_x']=np.where((data['RShoulder_x'].isnull()), data["RShoulder_x_AR1x_predicted"], data['RShoulder_x'])
#&(data['RShoulder_x_1']!=np.nan)


# In[408]:


print "Number of missing values after imputation:", data[['RShoulder_x']].isnull().values.ravel().sum()


# In[404]:


def AR1xy2(var, var1, var2, var_out, data0):
    
    var1_1=var1+"_1"
    
    var2_1=var2+"_1"

    Y_train = data0[var]
    X_train = data0[[var1, var1_1, var2, var2_1]]
        
    res = sm.OLS(Y_train, X_train, missing="drop").fit()

    # Inspect the results
    print(res.summary())
    #print('Parameters: ', res.params)

    var_predicted_df=var+"_predicted_df"
    
    var_predicted_df =pd.DataFrame(res.params[0]*X_train[var1]+res.params[1]*X_train[var1_1]+res.params[2]*X_train[var2]+res.params[3]*X_train[var2_1])
    var_predicted_df.columns=[var_out]
    print "Predicted table shape:", var_predicted_df.shape
    data0 = pd.concat([data0,var_predicted_df], axis=1, join_axes=[data0.index])
    print data0[[var_1,var2, var2_1,var_out]].head(10) 
    return data0


# In[406]:


data=AR1xy2(var="RShoulder_x", var1="Neck_x", var2="Neck_y", var_out="RShoulder_x_AR1xy_predicted", data0=data)


# In[415]:


data['RShoulder_x']=np.where((data['RShoulder_x'].isnull()), data["RShoulder_x_AR1xy_predicted"], data['RShoulder_x'])


# In[417]:


print "Number of missing values after imputation:", data[['RShoulder_x']].isnull().values.ravel().sum()


# 3.1. If  data[“RShoulder_y”]  is missing (i.e, 0) and data[“RShoulder_y_1”] is not missing, impute missing values with:
# E[“RShoulder_y”] = f[“RShoulder_y_1”]
# 3.2. If  both data[“RShoulder_y”] are missing (i.e, 0), impute the values with:
# E["RShoulder_x”] = f(“Neck_x”, “Neck_x_1” , “Neck_y”, “Neck_y_1”)

# In[419]:


print "Number of missing values before imputation:", data[['RShoulder_y']].isnull().values.ravel().sum()


# In[421]:


data = AR1x(var="RShoulder_y",var_out ="RShoulder_y_AR1y_predicted",  data0=data)


# In[422]:


data['RShoulder_y']=np.where((data['RShoulder_y'].isnull()), data["RShoulder_y_AR1y_predicted"], data['RShoulder_y'])


# In[423]:


print "Number of missing values after imputation:", data[['RShoulder_y']].isnull().values.ravel().sum()


# In[425]:


data=AR1xy2(var="RShoulder_y", var1="Neck_x", var2="Neck_y", var_out="RShoulder_y_AR1xy_predicted", data0=data)


# In[426]:


data['RShoulder_y']=np.where((data['RShoulder_y'].isnull()), data["RShoulder_y_AR1xy_predicted"], data['RShoulder_y'])


# In[427]:


print "Number of missing values after imputation:", data[['RShoulder_y']].isnull().values.ravel().sum()


# In[428]:


# Limiting min=0 
data['RShoulder_x'] =data['RShoulder_x'].apply(lambda x: max(0, x))
data['RShoulder_y'] =data['RShoulder_y'].apply(lambda x: max(0, x))


# In[429]:


# Limiting max=1
data['RShoulder_x'] =data['RShoulder_x'].apply(lambda x: min(1, x))
data['RShoulder_y'] =data['RShoulder_y'].apply(lambda x: min(1, x))


# In[431]:


data[['RShoulder_x', 'RShoulder_y']].describe()


# 4.1. If  data[“LShoulder_x”]  is missing (i.e, 0) and data[“LShoulder_y_1”] is not missing, impute missing values with:
# E[“LShoulder_x”] = f[“LShoulder_x_1”]
# 4.2. If  both data[“LShoulder_x”] are missing (i.e, 0), impute the values with:
# E["LShoulder_x”] = f(“Neck_x”, “Neck_x_1” , “Neck_y”, “Neck_y_1”)

# In[434]:


print "Number of missing values before imputation:", data[['LShoulder_x']].isnull().values.ravel().sum()
data['LShoulder_x'].describe()


# In[436]:


data=AR1x(var="LShoulder_x",var_out ="LShoulder_x_AR1x_predicted",  data0=data)


# In[437]:


data['LShoulder_x']=np.where((data['LShoulder_x'].isnull()), data["LShoulder_x_AR1x_predicted"], data['LShoulder_x'])


# In[438]:


print "Number of missing values before imputation:", data[['LShoulder_x']].isnull().values.ravel().sum()


# In[440]:


data=AR1xy2(var="LShoulder_x", var1="Neck_x", var2="Neck_y", var_out="LShoulder_x_AR1xy_predicted", data0=data)


# In[441]:


data['LShoulder_x']=np.where((data['LShoulder_x'].isnull()), data["LShoulder_x_AR1xy_predicted"], data['LShoulder_x'])


# In[442]:


print "Number of missing values after imputation:", data[['LShoulder_x']].isnull().values.ravel().sum()


# In[443]:


data['LShoulder_x'].describe()


# In[ ]:





# 5.1. If  data[“LShoulder_y”]  is missing (i.e, 0) and data[“LShoulder_y_1”] is not missing, impute missing values with:
# E[“LShoulder_y”] = f[“LShoulder_y_1”]
# 5.2. If  both data[“LShoulder_y”] are missing (i.e, 0), impute the values with:
# E["LShoulder_y”] = f(“Neck_x”, “Neck_x_1” , “Neck_y”, “Neck_y_1”)

# In[444]:


print "Number of missing values before imputation:", data[['LShoulder_y']].isnull().values.ravel().sum()
data['LShoulder_y'].describe()


# In[445]:


data=AR1x(var="LShoulder_y",var_out ="LShoulder_y_AR1y_predicted",  data0=data)


# In[446]:


data['LShoulder_y']=np.where((data['LShoulder_y'].isnull()), data["LShoulder_y_AR1y_predicted"], data['LShoulder_y'])


# In[447]:


print "Number of missing values after imputation:", data[['LShoulder_y']].isnull().values.ravel().sum()
data['LShoulder_y'].describe()


# In[448]:


data=AR1xy2(var="LShoulder_y", var1="Neck_x", var2="Neck_y", var_out="LShoulder_y_AR1xy_predicted", data0=data)


# In[449]:


data['LShoulder_y']=np.where((data['LShoulder_y'].isnull()), data["LShoulder_y_AR1xy_predicted"], data['LShoulder_y'])


# In[450]:


print "Number of missing values after imputation:", data[['LShoulder_y']].isnull().values.ravel().sum()
data['LShoulder_y'].describe()


# 6.1. If data[“RElbow_x”] is missing (i.e, 0) and data[“RElbow_x_1”] is not missing, impute missing values with:
# E[“RElbow_x”] = f[“RElbow_x_1”]
# 
# 6.2. If both data[“RElbow_x”]  (i.e, 0), impute the values with:   
# E[“RElbow_x”] = f(“Neck_x”, “Neck_x_1” , “Neck_y”, “Neck_y_1”, "RShoulder_x”, "RShoulder_y”, "LShoulder_x”, "LShoulder_y” )

# In[452]:


print "Number of missing values before imputation:", data[['RElbow_x']].isnull().values.ravel().sum()
data['RElbow_x'].describe()


# In[453]:


data=AR1x(var="RElbow_x",var_out ="RElbow_x_AR1x_predicted",  data0=data)


# In[454]:


data['RElbow_x']=np.where((data['RElbow_x'].isnull()), data["RElbow_x_AR1x_predicted"], data['RElbow_x'])


# In[455]:


print "Number of missing values after imputation:", data[['RElbow_x']].isnull().values.ravel().sum()
data['RElbow_x'].describe()


# In[ ]:





# In[ ]:





# In[ ]:





# 6.1. If data[“RElbow_y”] is missing (i.e, 0) and data[“RElbow_x_1”] is not missing, impute missing values with:
# E[“RElbow_y”] = f[“RElbow_y_1”]
# 
# 6.2. If both data[“RElbow_y”]  (i.e, 0), impute the values with:   
# E[“RElbow_y”] = f(“Neck_x”, “Neck_x_1” , “Neck_y”, “Neck_y_1”, "RShoulder_x”, "RShoulder_y”, "LShoulder_x”, "LShoulder_y” )

# In[456]:


print "Number of missing values before imputation:", data[['RElbow_y']].isnull().values.ravel().sum()
data['RElbow_y'].describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[451]:


get_ipython().system(u'jupyter nbconvert --to script Video_data_imputation_whole_sample.ipynb')


# In[ ]:




