

#@packages used
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import pickle
import os
import time
import datetime

# put the path of all the files below
os.chdir("C:\\Users\\qingch\\Desktop\\Foundry\\Results")

import Brazil_Custom_Model_Builder_Code_CQ_v4

#from sklearn.ensemble import RandomForestClassifier
#import time
#import re
#from sklearn.pipeline import Pipeline
#from collections import Counter
#from nltk.stem.porter import PorterStemmer


#Download two packages for text mining in Portuguese
#nltk.download('stopwords')
#nltk.download('rslp')



try:
    #Calling & executing pre_processing function
    #Reading Training and Prediction Data
    start_time = time.clock()    
#    Training_Data = pd.read_pickle('C:\\Users\\qingch\\Desktop\\Foundry\\Sample Data\\input_dataframe.pkl')
    Training_Data = pd.read_csv('Training Set.csv')
#    Training_Data = pd.read_csv('C:\\Users\\qingch\\Desktop\\Foundry\\Sample Data\\input_dataframe.csv')
    a1,a2,a3,a4,a5 = Brazil_Custom_Model_Builder_Code_CQ_v4.pre_processing(Training_Data, Training_Data)
    model_dat = a1
    model_label = a2
    vectorizer_fit = a3
    success_flag = a4
    status_desc = a5  
    print(success_flag,status_desc)
    
	
    
    #Pickling & saving vectorized training and prediction data

#    pickle.dump( vectorizer_fit, open( "vectorizer.pickle", "wb") ) 


	#UnPickling & loading vectorized training and prediction data
	
#	file1 = open('vectorizer.pickle' ,'rb')
#	vectorizer_fit = pickle.load(file1)

    
    #------------------------------------------------#
    #Calling & executing model
	
    b1,b2,b3 = Brazil_Custom_Model_Builder_Code_CQ_v4.model_train(model_label, vectorizer_fit)
	
    model = b1
    success_flag = b2
    status_desc = b3    
    print(success_flag, status_desc)
	#Pickling & saving Model
    with open('model.pickle', 'wb') as handle:
        pickle.dump(model, handle)     
		
	#UnPickling & loading model
#    file2 = open('model.pickle' ,'rb')
#    model = pickle.load(file2)
#    print(model)
	
	#Calling & executing prediction function
    Prediction_Data = Training_Data
    df,success_flag,status_desc = Brazil_Custom_Model_Builder_Code_CQ_v4.prediction(model, Training_Data, Prediction_Data)
    print(success_flag,status_desc)
 
	#Writing predicted output to csv
    df[['CODE', 'PRODUCT DESCRIPTION', 'RECOMMENDATION']].to_csv("Prediction Results.csv",index=False)
    print('Total time cost:', str(datetime.timedelta(seconds=round(time.clock()-start_time,0))))

    #Diagnostic metrics for different classes
    df['CORRECT CLASSIFICATION INDICATOR'] = (df['RECOMMENDATION']==df['CODE']).astype(int)
    df['INCORRECT CLASSIFICATION INDICATOR'] = (df['RECOMMENDATION']!=df['CODE']).astype(int)
    correctly_predicted_values = df.groupby(['CODE'])['CORRECT CLASSIFICATION INDICATOR'].sum()
    falsely_predicted_values = df.groupby(['CODE'])['INCORRECT CLASSIFICATION INDICATOR'].sum()
    accuracy_by_class = df.groupby(['CODE'])['CORRECT CLASSIFICATION INDICATOR'].mean()

except Exception as error:
    print(error)
	#-------------------------------------------------------------------------------------#	




	