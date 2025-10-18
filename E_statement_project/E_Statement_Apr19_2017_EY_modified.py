
# coding: utf-8

# ### Data Loading
# 
# #### Schema :
#     CURL_OUT_EXTR_ID        - Customer unique identifier 
#     estatement              - Is the customer sign up for estatement Y= Yes, N=No
#     CHD_DATE_OF_BIRTH       - Customer date of birth (in days
#     CHD_DATE_LAST_SALE      - Date of customer last purchase
#     CHD_OPEN_DATE           - Date account was open
#     CHD_DEL_NO_CYCLES       - Delinquency status of customer in cycles
#     INCOME                  - Customer Income
#     mkc                     - Have the customer signed out viewing account on mykohlscharge online: Y= Yes, N=No
#     ecs_enrl_date           - Enrollment date for MKC
#     cs_last_logn_date       - Date customer last signed in to MKC
#     ecs_logn_ct             - number of time customer log-ins into MKC
#     total_purch_2016        - Customer 2016 total spend in USD
#     CHD_LFTM_NET_PRCH_AM    -  Total purchases for the life of the account in USD
#     cust_zipcode            - Customer’s postal code  **TBD

# In[2]:

import math
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#import datetime
from datetime import datetime

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.4f}'.format


# In[3]:

import pandas as pd
import numpy as np

target_schema = ['estatement']

column_schema = [
    'CHD_DATE_OF_BIRTH',
    'CHD_DATE_LAST_SALE',
    'CHD_OPEN_DATE',
    'CHD_DEL_NO_CYCLES',
    'INCOME',
    'mkc',
    'ecs_enrl_date',
    'ecs_last_logn_date',
    'ecs_logn_ct',
    'total_purch_2016',
    'CHD_LFTM_NET_PRCH_AM'
]

# Load data from a csv file
# file_path = '../../../resources/'
file_path = '/users/eyankovsky/KohlSolutions/eStatement/'
#file_name =  'admes_000238_output_mkc_600K.csv'
file_name =  'admes_000238_output_mkc_600K_rf.csv'

#df = pd.read_csv(file_path + file_name , delimiter=',', error_bad_lines=False)
df = pd.read_csv(file_path + file_name , delimiter=',', error_bad_lines=False, low_memory=False)

# Split to Data frames
data_size = len(df)

target_df = df[target_schema]
column_df = df[column_schema]


# In[4]:

column_df.head()


# # Data Transformation

# In[9]:

def daysBetweenCalculation (date_to_convert, end_year, end_month, end_date, no_of_days):
    try:
        [m, d, y ] = date_to_convert.split('/', 3)
        mm = m.zfill(2)
        dd = d.zfill(2)

        if len(y) == 2:
            if int(y) < 99 :
                yyyy = '19' + y
            else :
                yyyy = '20' + y
        else :
            yyyy = y

        return int((date(end_year, end_month, end_date) - date(int(yyyy), int(mm), int(dd))).days / no_of_days)

    except:
        return 0


# In[5]:

target_df.loc[:,'trn_estatement'] = np.where(df.loc[:,'estatement']=='Y', 1, 0)


# In[6]:

target_df.head()


# In[12]:

def transferToLog (value):
    try:
        return long(np.log(value + 0.001))
    except RuntimeError:
        return 0.0


# In[8]:

max_age_date = datetime(2001,12,31,0,0,0) 
max_date = datetime(2017,3,1,0,0,0)
hundred_years = (datetime(2017,1,1,0,0,0) - datetime(1917,1,1,0,0,0))


# In[9]:

column_df.head()


# In[10]:

target_df.loc[:,'trn_estatement'] = np.where(target_df['estatement']=='Y', 1, 0)
print target_df['trn_estatement']


# In[11]:

column_df['Customer_birth_date'] = pd.to_datetime(column_df['CHD_DATE_OF_BIRTH'])
column_df.loc[:,'Customer_birth_date'] = np.where(column_df['Customer_birth_date']>=max_age_date, column_df.loc[:,'Customer_birth_date'] - hundred_years, column_df.loc[:,'Customer_birth_date'])
column_df['ext_customer_age'] = ((max_date - column_df['Customer_birth_date'])/np.timedelta64(365*24, 'h')).astype(np.int32)
#print column_df['ext_customer_age']


# In[12]:

column_df['CHD_OPEN_DATE_dt'] = pd.to_datetime(column_df['CHD_OPEN_DATE'])
column_df['ext_Kohls_account_tenure'] = ((max_date - column_df['CHD_OPEN_DATE_dt'])/np.timedelta64(365*24, 'h')).astype(np.int32)
#print column_df['ext_Kohls_account_tenure']


# In[13]:

column_df['trn_mkc'] = column_df[['mkc']].eq('Y').mul(1)
#print column_df['trn_mkc']


# In[14]:

column_df['ecs_enrl_date'] = np.where(column_df['ecs_enrl_date']=='.', '01/01/20', column_df['ecs_enrl_date'] )
column_df.loc[:,'ecs_enrl_date_dt'] = pd.to_datetime(column_df['ecs_enrl_date'])
column_df['ext_MKC_tenure'] =((max_date - column_df['ecs_enrl_date_dt'])/np.timedelta64(365*24, 'h')).astype(np.int32)
# column_df['ext_MKC_tenure'] = np.where(column_df['ext_MKC_tenure']<0, np.nan, column_df['ext_MKC_tenure']) 
# nan transformation is excluded since Random Forest does not accept missing values
#print column_df['ext_MKC_tenure']


# In[15]:

column_df['ecs_last_logn_date'] = np.where(column_df['ecs_last_logn_date']=='.', '01/01/20', column_df['ecs_last_logn_date'] )
column_df.loc[:,'ecs_last_logn_date_dt'] = pd.to_datetime(column_df['ecs_last_logn_date'])
column_df.loc[:,'ext_tm_frm_last_logn'] = ((max_date - column_df['ecs_last_logn_date_dt'])/np.timedelta64(30*24, 'h')).astype(np.int32)
#column_df['ext_tm_frm_last_logn'] = np.where(column_df['ext_tm_frm_last_logn']<0, np.nan, column_df['ext_tm_frm_last_logn'])
#print column_df['ecs_last_logn_date'], column_df['ecs_last_logn_date_dt'], column_df['ext_tm_frm_last_logn']


# In[16]:

column_df['total_purch_2016'] =np.where(column_df['total_purch_2016']=='.', '0', column_df['total_purch_2016'])
column_df['trn_total_purch_2016'] = column_df['total_purch_2016'].apply(pd.to_numeric, errors='coerce')
#print column_df['trn_total_purch_2016']


# In[17]:

column_df['CHD_LFTM_NET_PRCH_AM'] =np.where(column_df['CHD_LFTM_NET_PRCH_AM']==np.nan, 0, column_df['CHD_LFTM_NET_PRCH_AM'])
column_df['trn_total_purch_in_lifetime'] = column_df['CHD_LFTM_NET_PRCH_AM'].apply(pd.to_numeric, errors='coerce')
#print column_df['trn_total_purch_in_lifetime']


# In[18]:

column_df['CHD_DATE_LAST_SALE_dt'] = pd.to_datetime(column_df['CHD_DATE_LAST_SALE'])
column_df['ext_tm_frm_lst_purch'] = ((max_date - column_df['CHD_DATE_LAST_SALE_dt'])/np.timedelta64(30*24, 'h')).astype(np.int32)
#print column_df['ext_tm_frm_lst_purch']


# # Feature Selection 

# In[19]:

# Filter data to be loaded to Random Forest
feature_col_schema = [
'ext_customer_age',
'CHD_DEL_NO_CYCLES',
'ext_Kohls_account_tenure',
'trn_mkc',
'ext_MKC_tenure',
'ext_tm_frm_last_logn',
'trn_total_purch_2016',
'trn_total_purch_in_lifetime',
'ext_tm_frm_lst_purch'
]
# 'trn_customer_income',

feature_target_schema = ['trn_estatement']


# In[20]:

from sklearn.model_selection import train_test_split

test_sample_size = 0.33
random_state_value = 42

# Split Data Set as Train & Test
X_train, X_test, Y_train, Y_test = train_test_split(
    column_df[feature_col_schema],
    target_df[feature_target_schema], test_size = test_sample_size, random_state = random_state_value)

print 'Data Size : ' + str(data_size)
print 'Training Data Size : ' + str(len(X_train))
print 'Test Data Size : ' + str(len(X_test)) 


# In[21]:

X_train.head()
#, X_test, Y_train, Y_test


# In[22]:

Y_train.head()


# In[23]:

X_train.describe()


# In[24]:

X_test.describe()


# In[25]:

Y_test.describe()


# In[26]:

Y_train.describe()


# #  Models 

# ## Random Forest Classifier
# 
# Random Forest is a machine learning algorithm used for classification, regression, and feature selection. It's an ensemble technique, meaning it combines the output of one weaker technique in order to get a stronger result.
# 
# 
# <img src='../../../resources/rf.png'>
# 
# The weaker technique in this case is a decision tree. Decision trees work by splitting the and re-splitting the data by features. If a decision tree is split along good features, it can give a decent predictive output.

# #### Intiate Random Forest Classifier & start training 
# 
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).
# 
# Ref : http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.decision_path
# 
# ###### Random Forest Classifier Parameters
# 
#     1. n_estimators : The number of trees in the forest.
#     2. max_depth : The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# 

# In[49]:

from sklearn.ensemble import RandomForestClassifier

no_of_estimators = 3
tree_depth = 5

rf = RandomForestClassifier(n_estimators=no_of_estimators, max_depth=tree_depth)
rf.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())


# In[50]:

import seaborn as sns
get_ipython().magic(u'matplotlib inline')
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_.astype(float), 5)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print importances
importances.plot.bar()
plt.title("Chart. Variables' Importance resulted from Random Forest model")


# Random Forest Model Validation

# In[51]:

from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_rf_predict = rf.predict(X_test)

rf_rmse = math.sqrt(mean_squared_error(Y_test, Y_rf_predict))
rf_confusion_matrix = confusion_matrix(Y_test, Y_rf_predict)
rf_accuracy = accuracy_score(Y_test, Y_rf_predict)
rf_recall = recall_score(Y_test, Y_rf_predict)
rf_precision = precision_score(Y_test, Y_rf_predict)

print ' RMSE : ' + str(rf_rmse)
print '                                                        '
print ' Confusion Matrix  :'
print '  [ [ True_positive, False_negative ],  '
print '    [ False_positive, True_negative ] ] '
print '                                                        '
print  rf_confusion_matrix
print '                                                        '
print 'Accuracy Score : ' + str(rf_accuracy)
print 'Recall Score  : ' + str(rf_recall)
print 'Precision Score  : ' + str(rf_precision)


# Random Forest Model Visualization

# In[52]:

import pydotplus

from IPython.display import Image, display
from sklearn import tree

# Prepare tree dot files and convert to a PNG.
# Delete .odt file once PDF conversion is completed

i_tree = 0
for tree_in_forest in rf.estimators_:
    
    dot_data = tree.export_graphviz(decision_tree=tree_in_forest, 
                                    out_file=None,
                                    feature_names=feature_col_schema, #column_schema,
                                    filled=True,
                                    impurity=True,
                                    rounded=True,
                                    special_characters=True,
                                   proportion=False)    
    graph = pydotplus.graph_from_dot_data(dot_data)     
    tree_image = Image(graph.create_png())    
    
    nodes = graph.get_node_list()
    edges = graph.get_edge_list()    
    
    print 'Tree : ' + str(i_tree)    
    
    display(tree_image)        
    
    i_tree = i_tree + 1



#     More advanced Random Forest design suggested by Kaggle 3rd place winner (Rossman stores Kaggle project):
#     https://arxiv.org/pdf/1604.06737.pdf

# In[77]:

from sklearn.ensemble import RandomForestClassifier

no_of_estimators = 200
tree_depth = 35
min_samples_split = 2

rf = RandomForestClassifier(n_estimators=no_of_estimators, max_depth=tree_depth, min_samples_split = min_samples_split)
rf.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())


# In[78]:

import seaborn as sns
get_ipython().magic(u'matplotlib inline')
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_.astype(float), 5)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print importances
importances.plot.bar()
plt.title("Chart. Variables' Importance resulted from Random Forest model")


# #### Random Forest Model Validation

# In[79]:

from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_rf_predict = rf.predict(X_test)

rf_rmse = math.sqrt(mean_squared_error(Y_test, Y_rf_predict))
rf_confusion_matrix = confusion_matrix(Y_test, Y_rf_predict)
rf_accuracy = accuracy_score(Y_test, Y_rf_predict)
rf_recall = recall_score(Y_test, Y_rf_predict)
rf_precision = precision_score(Y_test, Y_rf_predict)

print ' RMSE : ' + str(rf_rmse)
print '                                                        '
print ' Confusion Matrix  :'
print '  [ [ True_positive, False_negative ],  '
print '    [ False_positive, True_negative ] ] '
print '                                                        '
print  rf_confusion_matrix
print '                                                        '
print 'Accuracy Score : ' + str(rf_accuracy)
print 'Recall Score  : ' + str(rf_recall)
print 'Precision Score  : ' + str(rf_precision)


# ## Gradient Boosting Classifier

# ##### Gradient Boosting Classifier Parameters
#     1. max_depth :
#     2. n_estimators :
#     3. subsample :
#     4. random_state :
#     5. learning_rate :

# In[151]:

from sklearn.ensemble import GradientBoostingClassifier

#common_args = {'max_depth': 3, 'n_estimators': 5, 'subsample': 0.5, 'random_state': 2}
common_args = {'max_depth': tree_depth, 'n_estimators': no_of_estimators, 'subsample': 0.5, 'random_state': 2}

gb = GradientBoostingClassifier(learning_rate=1, **common_args)
gb.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())


# #### Gradient Boosting Model Validation 

# In[75]:

from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_gb_predict = gb.predict(X_test)

gb_rmse = math.sqrt(mean_squared_error(Y_test, Y_gb_predict))
gb_confusion_matrix = confusion_matrix(Y_test, Y_gb_predict)
gb_accuracy = accuracy_score(Y_test, Y_gb_predict)
gb_recall = recall_score(Y_test, Y_gb_predict)
gb_precision = precision_score(Y_test, Y_gb_predict)

print 'RMSE : ' + str(gb_rmse)
print '                                                        '
print 'Confusion Matrix : '
print '                                                        '
print '[ [ True_positive, False_negative ],           '
print '  [ False_positive, True_negative ] ]          '
print '                                                        '
print  gb_confusion_matrix
print '                                                        '
print 'Accuracy Score : ' + str(gb_accuracy)
print '                                                        '
print 'Recall Score  : ' + str(gb_recall)
print ''
print 'Precision Score  : ' + str(gb_precision)


# In[80]:

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(ftr_rf, ttr_rf, label='Random Forest (test)')
plt.plot(ftr_grd, ttr_grd, label='Gradient Boosting (test)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves for predictions')
plt.legend(loc='best')
plt.show()


# In[169]:

from sklearn.metrics import roc_auc_score
rf_AUC_test = roc_auc_score(Y_test, y_pred_rf)
gb_AUC_test = roc_auc_score(Y_test, y_pred_grd)
print 'Random Forest AUC in a test sampe:',rf_AUC_test


# In[170]:

print 'Gradient Boosting Classifier AUC in a test sampe:', gb_AUC_test


# # Conclusion for Gradient Boosting Classifier:
# High complexity hyper-parameters:
# 
# no_of_estimators = 200
# tree_depth = 35
# 
# do not work well for the Gradient Boosting Classifier.
# The goodness-of-fit indictaors:
# 1) MSE = 19% vs MSE =9-10% for the model with smaller value parameters 
# 2) ROC is unexpectedly dives below 50/50% True Positive/False Positive dotted line;
# 3) AUC = 73.63% vs AUC (for the model with smaller value parameters) ~=90 %
# 4) Accuracy, Precision, Recall indicators are worse off comparing to the model with smaller value parameters

# In[27]:

from sklearn.ensemble import GradientBoostingClassifier

common_args = {'max_depth': 3, 'n_estimators': 5, 'subsample': 0.5, 'random_state': 2}
#common_args = {'max_depth': tree_depth, 'n_estimators': no_of_estimators, 'subsample': 0.5, 'random_state': 2}

gb = GradientBoostingClassifier(learning_rate=0.01, **common_args)
gb.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())


# In[33]:

import math
from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_gb_predict = gb.predict(X_test)

gb_rmse = math.sqrt(mean_squared_error(Y_test, Y_gb_predict))
gb_confusion_matrix = confusion_matrix(Y_test, Y_gb_predict)
gb_accuracy = accuracy_score(Y_test, Y_gb_predict)
gb_recall = recall_score(Y_test, Y_gb_predict)
gb_precision = precision_score(Y_test, Y_gb_predict)

print 'RMSE : ' + str(gb_rmse)
print '                                                        '
print 'Confusion Matrix : '
print '                                                        '
print '[ [ True_positive, False_negative ],           '
print '  [ False_positive, True_negative ] ]          '
print '                                                        '
print  gb_confusion_matrix
print '                                                        '
print 'Accuracy Score : ' + str(gb_accuracy)
print '                                                        '
print 'Recall Score  : ' + str(gb_recall)
print ''
print 'Precision Score  : ' + str(gb_precision)


# In[174]:

import seaborn as sns
get_ipython().magic(u'matplotlib inline')
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(gb.feature_importances_.astype(float),5)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print importances
importances.plot.bar()
plt.title("Chart. Variables' Importance resulted from Gradient Boosting Classifier")


# #### Gradient Boosting Model Visualization 

# ## Bagging using Decision Tree Classifier

# ##### Bagging using Decision Tree Classifier Parameters
# 
#     1. n_estimators :
#     2. max_samples :
#     3. random_state :

# In[154]:

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging =  BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, max_samples=0.5, random_state=2)
bagging.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())


# #### Bagging (Decision Tree Classifier) Model  Validation 

# In[384]:

from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_bg_predict = bagging.predict(X_test)

bg_rmse = sqrt(mean_squared_error(Y_test, Y_bg_predict))
bg_confusion_matrix = confusion_matrix(Y_test, Y_bg_predict)
bg_accuracy = accuracy_score(Y_test, Y_bg_predict)
bg_recall = recall_score(Y_test, Y_bg_predict)
bg_precision = precision_score(Y_test, Y_bg_predict)

print 'RMSE : ' + str(bg_rmse)
print '                                                        '
print 'Confusion Matrix : '
print '                                                        '
print '[ [ True_positive, False_negative ],'
print '  [ False_positive, True_negative ] ]'
print '                                                        '
print  bg_confusion_matrix
print '                                                        '
print 'Accuracy Score : ' + str(bg_accuracy)
print '                                                        '
print 'Recall Score  : ' + str(bg_recall)
print '                                                        '
print 'Precision Score  : ' + str(bg_precision)


# In[83]:

from sklearn.metrics import roc_curve


# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_pred_rf)


y_train_rf = rf.predict_proba(X_train)[:, 1]
ftr_rf, ttr_rf, _ = roc_curve(Y_train, y_train_rf)


# The gradient boosted model by itself
y_pred_grd = gb.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(Y_test, y_pred_grd)

y_train_grd = gb.predict_proba(X_train)[:, 1]
ftr_grd, ttr_grd, _ = roc_curve(Y_train, y_train_grd)

# The bagging model by itself
y_pred_bagging = bagging.predict_proba(X_test)[:, 1]
fpr_bgng, tpr_bgng, _ = roc_curve(Y_test, y_pred_bagging)

y_train_bagging = bagging.predict_proba(X_train)[:, 1]
ftr_bgng, ttr_bgng, _ = roc_curve(Y_train, y_train_bagging)


# Neural Network with embeddings, following
# https://arxiv.org/pdf/1604.06737.pdf
# and
# https://github.com/entron/entity-embedding-rossmann
# 
# https://keras.io/getting-started/functional-api-guide/#more-examples
# 
# 
# may require fixing pip install errors
# https://github.com/dmlc/xgboost/issues/463
# brew install --with-clang llvm
# brew install cmake
# brew install gcc --without-multilib
# 
# git clone --recursive https://github.com/dmlc/xgboost
# cd xgboost; cp make/config.mk ./config.mk; make -j4
# 
# cd python-package/ ; python setup.py install

# Simple NN model

# In[75]:

import keras


# In[301]:

trainx = X_train.as_matrix()
testx = X_test.as_matrix()
#print(trainx)


# In[302]:

trainy = Y_train.as_matrix()
testy = Y_test.as_matrix()
#print(testy)


# In[100]:

from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(9,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(9, activation='relu')(inputs)
x = Dense(9, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(trainx, trainy, epochs=5, batch_size=200)  # starts training


# In[181]:

# This returns a tensor
inputs = Input(shape=(9,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(9, activation='relu')(inputs)
x = Dense(9, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model_nn = Model(inputs=inputs, outputs=predictions)
model_nn.compile(keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model_nn.fit(trainx, trainy, epochs=5, batch_size=200)  # starts training


# In[182]:

y_pred_nn = model_nn.predict(testx, batch_size=128)
y_train_nn = model_nn.predict(trainx, batch_size=128)


# Neural Network with embeddings:
# https://keras.io/getting-started/functional-api-guide/#more-examples

# In[140]:

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model


# In[188]:

from keras.layers import Input, Embedding, Dense
from keras.models import Model

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
inputs = Input(shape=(9,))

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
#x = Embedding(output_dim=5, input_dim=9, input_length=100)(main_input)

x = Embedding(output_dim=9, input_dim=9)(inputs)

x = Dense(18, activation='relu')(inputs)
x = Dense(9, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model_nn_emb = Model(inputs=inputs, outputs=predictions)
model_nn_emb.compile(keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model_nn_emb.fit(trainx, trainy, epochs=5, batch_size=200)  # starts training


# In[180]:

y_pred_nn_emb = model_nn_emb.predict(testx, batch_size=128)
y_train_nn_emb = model_nn_emb.predict(trainx, batch_size=128)


# In[ ]:

# RMSE calcultaions
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculations below are done to verify the threshold and to report summary statistics 
y_predicted=pd.DataFrame(np.zeros((len(y_pred_grd), 5),dtype=float), columns=['rf', 'gb', 'bg', 'nn','nn+emb'])

y_predicted.loc[:, 'rf'] = np.where(y_pred_rf >= 0.5, 1, 0)
rf_rmse = sqrt(mean_squared_error(Y_test, y_predicted['rf']))

y_predicted.loc[:, 'gb'] = np.where(y_pred_grd >= 0.5, 1, 0)
gb_rmse = sqrt(mean_squared_error(Y_test, y_predicted['gb']))

y_predicted.loc[:, 'bg'] = np.where(y_pred_bagging >= 0.5, 1, 0)
bg_rmse = sqrt(mean_squared_error(Y_test, y_predicted['bg']))

y_predicted.loc[:, 'nn'] = np.where(y_pred_nn >= 0.5, 1, 0)
nn_rmse = sqrt(mean_squared_error(Y_test, y_predicted['nn']))

y_predicted.loc[:, 'nn+emb'] = np.where(y_pred_nn_emb >= 0.5, 1, 0)
nn_emb_rmse = sqrt(mean_squared_error(Y_test, y_predicted['nn+emb']))


# In[81]:

# Scenario analysis for Gradient Boosting Classifier
prob_threshold = 0.50
y_predicted=pd.DataFrame(np.zeros((len(y_pred_grd), 5),dtype=float), columns=['rf', 'gb', 'bg', 'nn','nn+emb'])
y_pred_grd = gb.predict_proba(X_test)[:, 1]
y_predicted.loc[:, 'gb'] = np.where(y_pred_grd >= prob_threshold , 1, 0)
gb_rmse = sqrt(mean_squared_error(Y_test, y_predicted['gb']))


from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)


gb_rmse = math.sqrt(mean_squared_error(Y_test, y_predicted['gb']))
gb_confusion_matrix = confusion_matrix(Y_test, y_predicted['gb'])
gb_accuracy = accuracy_score(Y_test, y_predicted['gb'])
gb_recall = recall_score(Y_test, y_predicted['gb'])
gb_precision = precision_score(Y_test, y_predicted['gb'])

print 'RMSE : ' + str(gb_rmse)
print '                                                        '
print 'Confusion Matrix : '
print '                                                        '
print '[ [ True_positive, False_negative ],'
print '  [ False_positive, True_negative ] ]'
print '                                                        '
print  gb_confusion_matrix
print '                                                        '
print 'Accuracy Score : ' + str(gb_accuracy)
print '                                                        '
print 'Recall Score  : ' + str(gb_recall)
print '                                                        '
print 'Precision Score  : ' + str(gb_precision)


# In[70]:

print y_predicted['gb']


# In[73]:

print y_pred_grd


# In[69]:

predict_data_75 = pd.DataFrame({'target': Y_test["trn_estatement"], 'GB_prediction': y_pred_grd, 'GB_target': y_predicted['gb']},
                            columns=['target', 'GB_prediction', 'GB_target'])
print predict_data_75
#predict_data_75.to_csv("/users/eyankovsky/Desktop/Kohls_Use_cases/eStatement/prediction_75thrhld_Apr19_2017.csv", sep=',')


# In[283]:

from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)


nn_rmse = sqrt(mean_squared_error(Y_test, y_predicted['nn']))
nn_confusion_matrix = confusion_matrix(Y_test, y_predicted['nn'])
nn_accuracy = accuracy_score(Y_test, y_predicted['nn'])
nn_recall = recall_score(Y_test, y_predicted['nn'])
nn_precision = precision_score(Y_test, y_predicted['nn'])

print 'RMSE : ' + str(nn_rmse)
print '                                                        '
print 'Confusion Matrix : '
print '                                                        '
print '[ [ True_positive, False_negative ],'
print '  [ False_positive, True_negative ] ]'
print '                                                        '
print  nn_confusion_matrix
print '                                                        '
print 'Accuracy Score : ' + str(nn_accuracy)
print '                                                        '
print 'Recall Score  : ' + str(nn_recall)
print '                                                        '
print 'Precision Score  : ' + str(nn_precision)


# In[282]:

from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)


nn_emb_rmse = sqrt(mean_squared_error(Y_test, y_predicted['nn+emb']))
nn_emb_confusion_matrix = confusion_matrix(Y_test, y_predicted['nn+emb'])
nn_emb_accuracy = accuracy_score(Y_test, y_predicted['nn+emb'])
nn_emb_recall = recall_score(Y_test, y_predicted['nn+emb'])
nn_emb_precision = precision_score(Y_test, y_predicted['nn+emb'])

print 'RMSE : ' + str(nn_emb_rmse)
print '                                                        '
print 'Confusion Matrix : '
print '                                                        '
print '[ [ True_positive, False_negative ],'
print '  [ False_positive, True_negative ] ]'
print '                                                        '
print  nn_emb_confusion_matrix
print '                                                        '
print 'Accuracy Score : ' + str(nn_emb_accuracy)
print '                                                        '
print 'Recall Score  : ' + str(nn_emb_recall)
print '                                                        '
print 'Precision Score  : ' + str(nn_emb_precision)


# In[ ]:

from sklearn.metrics import roc_curve


# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(Y_test, y_pred_rf)


y_train_rf = rf.predict_proba(X_train)[:, 1]
ftr_rf, ttr_rf, _ = roc_curve(Y_train, y_train_rf)


# The gradient boosted model by itself
y_pred_grd = gb.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(Y_test, y_pred_grd)

y_train_grd = gb.predict_proba(X_train)[:, 1]
ftr_grd, ttr_grd, _ = roc_curve(Y_train, y_train_grd)

# The bagging model by itself
y_pred_bagging = bagging.predict_proba(X_test)[:, 1]
fpr_bgng, tpr_bgng, _ = roc_curve(Y_test, y_pred_bagging)

y_train_bagging = bagging.predict_proba(X_train)[:, 1]
ftr_bgng, ttr_bgng, _ = roc_curve(Y_train, y_train_bagging)

# The NN model by itself
fpr_nn, tpr_nn, _ = roc_curve(Y_test, y_pred_nn)
ftr_nn, ttr_nn, _ = roc_curve(Y_train, y_train_nn)

# The NN with embeddings model by itself
fpr_nn_emb, tpr_nn_emb, _ = roc_curve(Y_test, y_pred_nn_emb)
ftr_nn_emb, ttr_nn_emb, _ = roc_curve(Y_train, y_train_nn_emb)


# In[ ]:

from sklearn.metrics import roc_auc_score

rf_AUC_test= roc_auc_score(Y_test, y_pred_rf)
gb_AUC_test= roc_auc_score(Y_test, y_pred_grd)
bagging_AUC_test= roc_auc_score(Y_test, y_pred_bagging)
nn_AUC_test= roc_auc_score(Y_test, y_pred_nn)
nn_emb_AUC_test= roc_auc_score(Y_test, y_pred_nn_emb)

rf_AUC_train = roc_auc_score(Y_train, y_train_rf)
gb_AUC_train = roc_auc_score(Y_train, y_train_grd)
bagging_AUC_train= roc_auc_score(Y_train, y_train_bagging)
nn_AUC_train = roc_auc_score(Y_train, y_train_nn)
nn_emb_AUC_train = roc_auc_score(Y_train, y_train_nn_emb)


# In[284]:

Fit_df = pd.DataFrame({
          'AUC in test': [rf_AUC_test, gb_AUC_test, bagging_AUC_test, nn_AUC_test, nn_emb_AUC_test],
          'AUC in training': [rf_AUC_train, gb_AUC_test, bagging_AUC_train, nn_AUC_train, nn_emb_AUC_train],
          'RMSE (50%)': [rf_rmse, gb_rmse, bg_rmse, nn_rmse, nn_emb_rmse],
          'Accuracy (50%)': [rf_accuracy, gb_accuracy, bg_accuracy, nn_accuracy, nn_emb_accuracy],  
          'Precision (50%)': [rf_recall, gb_recall, bg_recall, nn_recall, nn_emb_recall],
          'Recall (50%)': [rf_precision, gb_precision, bg_precision, nn_precision, nn_emb_precision]
           },
        index = ['Random Forest', 'Gradient Boosting', 'Bagging Classifier', 'Neural Network', 'Neural Network w Embeddings'])


# In[294]:

print "Summary of the models' goodness-of-fit statistics \n", Fit_df


# In[299]:

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='Random Forest (test)')
plt.plot(fpr_grd, tpr_grd, label='Gradient Boosting (test)')
plt.plot(fpr_bgng, tpr_bgng, label='Bagging Classifier (test)')
plt.plot(fpr_nn, tpr_nn, label='Neural Network (test)')
plt.plot(fpr_nn_emb, tpr_nn_emb, label='Neural Network with Embeddings (test)')
#plt.plot(fpr_rf, tpr_rf, label='Random Forest (train)')
#plt.plot(fpr_grd, tpr_grd, label='Gradient Boosting (train)')
#plt.plot(fpr_bgng, tpr_bgng, label='Bagging Classifier (train)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves for predictions')
plt.legend(loc='best')
plt.show()


# In[300]:

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(ftr_rf, ttr_rf, label='Random Forest (train)')
plt.plot(ftr_grd, ttr_grd, label='Gradient Boosting (train)')
plt.plot(ftr_bgng, ttr_bgng, label='Bagging Classifier (train)')
plt.plot(ftr_nn, ttr_nn, label='Neural Network (train)')
plt.plot(ftr_nn_emb, ttr_nn_emb, label='Neural Network with Embeddings (train)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves for predictions')
plt.legend(loc='best')
plt.show()


# In[391]:

predict_data = pd.DataFrame({'target': Y_test["trn_estatement"], 'RF_prediction': y_pred_rf, 'GB_prediction': y_pred_grd, 'BGNG_prediction': y_pred_bagging, 'NN_prediction': np.squeeze(np.asarray(y_pred_nn)), 'NN_emb_prediction': np.squeeze(np.asarray(y_pred_nn_emb))},
                            columns=['target', 'RF_prediction', 'GB_prediction','BGNG_prediction', 'NN_prediction','NN_emb_prediction'])


# In[392]:

print(predict_data)


# In[393]:

# predict_data.reset_index(level=0, inplace=True)
predict_data['index'] = predict_data.index.astype(float)
print predict_data


# In[404]:

df_id = pd.DataFrame(df['CURL_OUT_EXTR_ID'])
df_id['index'] = pd.to_numeric(df_id.index, errors = 'coerce')
#print df_id.dtypes


# In[410]:

print predict_data.dtypes


# In[411]:

predict_data_id=predict_data.merge(df_id, left_on='index', right_on='index', how = 'inner')
predict_data_id.drop('index', axis=1, inplace=True)


# In[413]:

print predict_data_id


# In[409]:

predict_data_id.to_csv("/users/eyankovsky/Desktop/Kohls_Use_cases/eStatement/prediction_Apr19_2017.csv", sep=',')

