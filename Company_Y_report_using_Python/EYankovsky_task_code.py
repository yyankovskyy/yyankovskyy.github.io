
# coding: utf-8

# In[408]:

import math
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display
#import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#import datetime
from datetime import datetime as dt

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3, suppress=True) 

import pandas as pd
import numpy as np
from itertools import islice

import os
   


# In[409]:

path = "/Users/eyankovsky/Desktop/Shutterfly/"
file_name = "part3.csv"
os.chdir(path)


# In[410]:

df = pd.read_csv(file_name , delimiter=',', error_bad_lines=False, low_memory=False, parse_dates=['order_date'],
dtype={
"category_name": np.str,        
"product_name": np.str,
"order_sequence": np.int,
"revenue": np.float64, 
"units": np.int,
"customer_id":np.str,
"order_id":np.int 
}                       
)


# # Exploratory data analysis

# In[411]:

#df=df[df["order_sequence"]<3]
df["order_id"] = df["order_id"].astype(str)
df["customer_id"] = df["customer_id"].astype(str)
print df.head(10)
print df.dtypes


# In[401]:

print len(df["category_name"].unique())
print df["category_name"].unique()


# In[402]:

print len(df["product_name"].unique())
print df["product_name"].unique()


# In[412]:

df.describe()


# In[404]:

var = "units"
#"revenue" 
#"order_sequence"
title = 'Boxplot. Distribution of '+ var
get_ipython().magic(u'matplotlib inline')
fig = df.boxplot(var, vert=True) 
plt.title(title)
# plt.suptitle('')
plt.show()


# In[125]:

print df[(df["customer_id"]=="363293") | 
         (df["customer_id"]=="82783278") |
         (df["customer_id"]=="48865200") |
         (df["customer_id"]=="24783955")]


# # Data transformation

# In[221]:

df=df[df["order_sequence"]<3]
print df.head(10)
print df.dtypes


# # Data check
# # df[(df["customer_id"]=="10009254")]

# In[413]:

df_max=df['order_sequence'].groupby(df["customer_id"]).max().reset_index()
df_max.rename(columns={"order_sequence":"max_order_sequence"}, inplace = True)
df_max.describe()


# In[414]:

df1=df[(df["order_sequence"]<2) & (df["revenue"]>=0)]
df1.drop("order_id", axis=1, inplace = True)
df1.describe()


# In[415]:

# Aggregation by customer_id 
df1_sum = df1.groupby(df1["customer_id"]).sum().reset_index()
df1_sum.drop("order_sequence", axis=1, inplace = True)
df1_sum.rename(columns={"revenue": "total_revenue", "units":"total_units"}, inplace = True)
df1_sum.head(5)


# In[416]:

# Aggregation by customer_id | catgeory_name
df1_cat_sum = df1.groupby([df1["customer_id"], df1["category_name"]]).sum().reset_index()
df1_cat_sum.drop("order_sequence", axis=1, inplace = True)
df1_cat_sum.head(10)


# In[417]:

pivot_cat = df1_cat_sum.pivot_table(values = ["revenue","units"], columns = ["category_name"], index = ['customer_id'], aggfunc= np.sum)
pivot_cat_df = pd.DataFrame(pivot_cat.to_records())
print pivot_cat_df.head()


# In[418]:

pivot_cat_df = pivot_cat_df.add_suffix('_category')
pivot_cat_df.rename(columns = {"customer_id_category": "customer_id"}, inplace = True) 
pivot_cat_df.fillna(0, inplace = True)
print pivot_cat_df.head(5)


# In[ ]:




# In[419]:

# Aggregation by customer_id | product_name
df1_prod_sum = df1.groupby([df1["customer_id"], df1["product_name"]]).sum().reset_index()
df1_prod_sum.drop("order_sequence", axis=1, inplace = True)
df1_prod_sum.head(10)


# In[420]:

pivot_prod = df1_prod_sum.pivot_table(values = ["revenue","units"], columns = ["product_name"], index = ['customer_id'], aggfunc= np.sum)
pivot_prod_df = pd.DataFrame(pivot_prod.to_records())
print pivot_prod_df.head()


# In[421]:

pivot_prod_df = pivot_prod_df.add_suffix('_product')
pivot_prod_df.rename(columns = {"customer_id_product": "customer_id"}, inplace = True) 
pivot_prod_df.fillna(0, inplace = True)
print pivot_prod_df.head(5)


# In[422]:

# df1 preparation
df11 = df1[['order_date', 'customer_id']]
df11.drop_duplicates(subset=['order_date', 'customer_id'], keep='first', inplace=True)
df11.head()


# In[ ]:




# In[423]:

print df11.describe()
print df1_sum.describe()


# In[424]:

# Merge
df1_sum_m=df11.merge(df1_sum, on ="customer_id", how='left')
df1_sum_m.describe()
#df1_sum_m.drop("order_sequence", axis=1, inplace = True)


# In[425]:

df1_sum_cat = df1_sum_m.merge(pivot_cat_df,on ="customer_id", how='left')
df1_sum_cat.describe()


# In[ ]:




# In[426]:

df_m2 = df1_sum_cat.merge(pivot_prod_df,on ="customer_id", how='left')
df_m2.describe()


# In[427]:

df_fin = df_m2.merge(df_max,on ="customer_id", how='left')
df_fin.describe()



# In[428]:

df_fin.head()


# In[429]:

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

#dr = pd.date_range(start='2014-01-01', end='2014-12-31')
#df = pd.DataFrame()
#df_fin['order_date'] = dr

cal = calendar()
holidays = cal.holidays(start='2014-01-01', end='2014-12-31')

df_fin['holiday'] = df_fin["order_date"].isin(holidays)


# In[430]:

df_fin['holiday'].head()


# In[431]:

df_fin['month'] = df_fin["order_date"].dt.month
df_fin['week'] = df_fin["order_date"].dt.week
df_fin['weekday'] = df_fin["order_date"].dt.dayofweek
df_fin['weekend'] = np.where(((df_fin['weekday'] ==5)|(df_fin['weekday'] ==6)), 1, 0)
df_fin['holiday'] = np.where(df_fin['holiday'] ==True, 1, 0)
df_fin['customer_retained'] = np.where(df_fin['max_order_sequence'] >=2, 1, 0)

print df_fin['month'].describe()
print df_fin['week'].describe()
print df_fin['weekday'].describe()
print df_fin['weekend'].describe()
print df_fin['holiday'].describe()
print df_fin['customer_retained'].describe()


# In[432]:

df_fin.drop("order_date", axis =1, inplace = True)
df_fin.drop("max_order_sequence", axis =1, inplace = True)
df_fin.dtypes


# In[433]:

print df_fin.head()
print df_fin.describe()


# In[434]:

df_in = df_fin[(df_fin["customer_retained"]==1)]
df_out = df_fin[(df_fin["customer_retained"]==0)]


# In[435]:

# Check
print "Sample size", len(df_fin)
print "Number of retained customers in a sample", len(df_in)
print "Number of churned customers in a sample", len(df_out)


#    # Developing a predictive model

# In[436]:

from sklearn.model_selection import train_test_split
test_sample_size = 0.20
random_state_value = 42

target_schema = ['customer_retained']


# In[437]:

feature_schema = df_fin.columns.tolist()
feature_schema.remove('customer_retained')
feature_schema.remove('customer_id')
feature_schema.remove('month')
feature_schema.remove('week')
feature_schema.remove('weekday')
print feature_schema


# In[438]:

X_train, X_test, Y_train, Y_test = train_test_split(
    df_fin[feature_schema],
    df_fin[target_schema], test_size = test_sample_size, random_state = random_state_value)

print 'Data Size : ' + str(len(df_fin))
print 'Training Data Size : ' + str(len(X_train))
print 'Test Data Size : ' + str(len(X_test)) 


# In[439]:

# ###### Random Forest Classifier Parameters
# 
#     1. n_estimators : The number of trees in the forest.
#     2. max_depth : The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# 

from sklearn.ensemble import RandomForestClassifier

no_of_estimators = 100
tree_depth = 5

rf = RandomForestClassifier(n_estimators=no_of_estimators, max_depth=tree_depth)
rf.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())




import seaborn as sns
get_ipython().magic(u'matplotlib inline')
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_.astype(float), 5)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print importances
importances.plot.bar()
plt.title("Chart. Variables' Importance resulted from Random Forest model")


# Random Forest Model Validation


from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_rf_predict = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(Y_test, Y_rf_predict))
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


# In[445]:

feature_schema_rev = [
"total_revenue",
"('revenue', 'Calendars')_category",
"('units', 'Calendars')_category",
"('revenue', 'Photo Books')_category",
"('revenue', 'Wall Calendars')_product",
"total_units",
"('units', 'Prints')_category",
"('units', 'Wall Calendars')_product",
"('revenue', 'Prints')_category",
"('revenue', '4x6')_product",
"('units', '4x6')_product",
"('revenue', '8x11 Classic Book')_product",
"('revenue', 'Premium Content')_product",
"('revenue', '8x10')_product",
"('revenue', 'Home Decor')_category",
"('revenue', '5x7')_product",
"('units', '4x4')_product",
"('revenue', 'Magnets')_product",
"('units', '5x7')_product",
"('revenue', 'Gifts')_category"
]

X_train, X_test, Y_train, Y_test = train_test_split(
    df_fin[feature_schema_rev],
    df_fin[target_schema], test_size = test_sample_size, random_state = random_state_value)

print 'Data Size : ' + str(len(df_fin))
print 'Training Data Size : ' + str(len(X_train))
print 'Test Data Size : ' + str(len(X_test)) 


# In[446]:


from sklearn.ensemble import RandomForestClassifier

no_of_estimators = 200
tree_depth = 30

rf = RandomForestClassifier(n_estimators=no_of_estimators, max_depth=tree_depth)
rf.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())




import seaborn as sns
get_ipython().magic(u'matplotlib inline')
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_.astype(float), 5)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print importances
importances.plot.bar()
plt.title("Chart. Variables' Importance resulted from Random Forest model")


# Random Forest Model Validation


from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_rf_predict = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(Y_test, Y_rf_predict))
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


# In[ ]:




# In[447]:

# ## Gradient Boosting Classifier

# ##### Gradient Boosting Classifier Parameters
#     1. max_depth :
#     2. n_estimators :
#     3. subsample :
#     4. random_state :
#     5. learning_rate :


from sklearn.ensemble import GradientBoostingClassifier

common_args = {'max_depth': 30, 'n_estimators': 200, 'subsample': 0.5, 'random_state': 2}
#common_args = {'max_depth': tree_depth, 'n_estimators': no_of_estimators, 'subsample': 0.5, 'random_state': 2}

gb = GradientBoostingClassifier(learning_rate=0.5, **common_args)
gb.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())


# #### Gradient Boosting Model Validation 



from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_gb_predict = gb.predict(X_test)

gb_rmse = np.sqrt(mean_squared_error(Y_test, Y_gb_predict))
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






# In[448]:

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging =  BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, max_samples=0.5, random_state=2)
bagging.fit(X_train.iloc[:,0:].values, Y_train.iloc[:,0:].values.ravel())


# #### Bagging (Decision Tree Classifier) Model  Validation 

# In[384]:

from sklearn.metrics import (mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score)

Y_bg_predict = bagging.predict(X_test)

bg_rmse = np.sqrt(mean_squared_error(Y_test, Y_bg_predict))
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


# In[449]:

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



# In[450]:

# In[167]:

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(ftr_rf, ttr_rf, label='Random Forest (test)')
plt.plot(ftr_grd, ttr_grd, label='Gradient Boosting (test)')
plt.plot(fpr_bgng, tpr_bgng, label='Bagging Classifier (test)')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curves for predictions')
plt.legend(loc='best')
plt.show()


# In[451]:

from sklearn.metrics import roc_auc_score

rf_AUC_test= roc_auc_score(Y_test, y_pred_rf)
gb_AUC_test= roc_auc_score(Y_test, y_pred_grd)
bagging_AUC_test= roc_auc_score(Y_test, y_pred_bagging)

print "AUC for Random Forest", rf_AUC_test
print "AUC for Gradient Boosting", gb_AUC_test
print "AUC for Bagging", bagging_AUC_test


# In[440]:

# Logistic regression
import statsmodels.api as sm


# In[443]:

logit = sm.Logit(Y_train, X_train)


# In[444]:

result = logit.fit()
print result.summary()


# In[ ]:

get_ipython().system(u'')


# In[ ]:



