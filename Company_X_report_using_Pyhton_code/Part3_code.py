# This is a code to address task 3

import urllib, re, os, sys, time 
os.system('cls' if os.name == 'nt' else 'clear') 
path = "/Users/anayankovsky/Desktop/UberTask/aUberPythonTask"
os.chdir(path)


# Check current working directory.
retval = os.getcwd()
print "Current working directory %s" % retval


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt
from datetime import datetime, date, time

import matplotlib
import math
import sklearn.metrics

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.cross_validation import  cross_val_score
from sklearn.cross_validation import  train_test_split
from sklearn.cross_validation import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline



# read_json somehow does not work in this case
# df3 = pd.read_json('uber_data_challenge.json')

df3 = pd.read_csv('uber_data_challenge.csv')
print df3.head()



df3.describe()


# Connversion into a date format
df3["signup_date"] = pd.to_datetime(df3["signup_date"])
df3["last_trip_date"] = pd.to_datetime(df3["last_trip_date"])


print '\n Data Types check:'
print df3.dtypes


# 1.3. Data visualization
%matplotlib
#pd.options.display.mpl_style = 'default'
matplotlib.pyplot.style.use = 'default'
# Histograms of all continuous variables
df3.hist()


fig=df3.boxplot('avg_dist', vert=True)
plt.title('Distribution of Average Distribution, avg_dist')
plt.suptitle("")
plt.show()

fig=df3.boxplot('avg_surge', vert=True)
plt.title('Distribution of Average Surge, avg_surge')
plt.suptitle("")
plt.show()

fig=df3.boxplot('surge_pct', vert=True)
plt.title('Distribution of Surge Percent, surge_pct')
plt.suptitle("")
plt.show()

fig=df3.boxplot('avg_rating_by_driver', vert=True)
plt.title('Distribution of Average Rating by Driver, avg_rating_by_driver')
# fig.set_ylabel(" ")
plt.suptitle(" ")
plt.show()


fig=df3.boxplot('avg_rating_of_driver', vert=True)
plt.title('Distribution of Average Rating of Driver, avg_rating_of_driver')
#fig.set_ylabel(" ")
plt.suptitle(" ")
plt.show()

fig=df3.boxplot('weekday_pct', vert=True)
plt.title('Distribution of Weekday Percent, weekday_pct')
#fig.set_ylabel("weekday_pct")
plt.suptitle(" ")
plt.show()


fig=df3.boxplot('trips_in_first_30_days', vert=True)
plt.title('Distribution of Trips in first 30 days,trips_in_first_30_days')
#fig.set_ylabel("trips_in_first_30_days")
plt.suptitle(" ")
plt.show()




# 1.4. Data transformation

#  Data cleaning and addition
# Substitute NaN with missing value

df3["avg_rating_of_driver"].replace('NaN', np.nan, inplace=True)
df3["avg_rating_by_driver"].replace('NaN', np.nan, inplace=True)
#df3["phone"].replace('NaN', '99', inplace=True)



df3.describe()
days_fromJul1_to_last_trip =(datetime(2014, 7, 1, 0, 0, 0) - df3["last_trip_date"])/np.timedelta64(1,'D')
df3["days_fromJul1_to_last_trip"] = days_fromJul1_to_last_trip.astype(int)

df3.loc[:,'active'] = np.where(df3["days_fromJul1_to_last_trip"]<=30, 1, 0)

tenure = df3["last_trip_date"] - df3["signup_date"]
df3.loc[:,'tenure'] = (tenure / np.timedelta64(1, 'D')).astype(int)

df3.loc[:,'ln_avg_rating_by_driver']  = np.where(df3["avg_rating_by_driver"] == np.nan, 99, np.log(df3["avg_rating_by_driver"] + 0.001))
df3.loc[:,'ln_avg_rating_of_driver']  = np.where(df3["avg_rating_of_driver"] == np.nan, 99, np.log(df3["avg_rating_of_driver"] + 0.001))


df3.loc[:,'ln_weekday_pct']=np.log(df3["weekday_pct"] + 0.001)
df3.loc[:,'ln_surge_pct']=np.log(df3["surge_pct"] + 0.001)
df3.loc[:,'ln_avg_surge']=np.log(df3["avg_surge"] + 0.001)
df3.loc[:,'ln_avg_dist']=np.log(df3["avg_dist"] + 0.001)
df3.loc[:,'ln_trips_in_first_30_days']=np.log(df3["trips_in_first_30_days"] + 0.001)

df3.loc[:,'astapor'] = np.where(df3["city"]=="Astapor", 1, 0)
df3.loc[:,'kings_landing'] = np.where(df3["city"]=="King's Landing", 1, 0)
df3.loc[:,'winterfell'] = np.where(df3["city"]=="Winterfell", 1, 0)


#df3.loc[:,'phone_cor'].replace('NaN', '99', inplace=True)
df3.loc[:,'iphone'] = np.where(df3["phone"]=="iPhone", 1, 0)
df3.loc[:,'android'] = np.where(df3["phone"]=="Android", 1, 0)
#df3[:,'uber_black'] = np.where(df3["uber_black_user"]==1, 1, 0)

# Dummy variables for outlying observations:
df3.loc[:,'avg_dist_outlier'] = np.where(df3["avg_dist"]>80, 1, 0)
df3.loc[:,'surge_pct_outlier'] = np.where(df3["surge_pct"]>60, 1, 0)
df3.loc[:,'avg_surge_outlier'] = np.where(df3["avg_surge"]>7, 1, 0)
df3.loc[:,'trips_in_first_30_days_outlier'] = np.where(df3["trips_in_first_30_days"]>30, 1, 0)


# Decision tree with 10-fold cross-validation

df3_dtree =df3[['active','uber_black_user','tenure','astapor','kings_landing','winterfell','iphone','android','avg_dist','weekday_pct',
'avg_rating_by_driver','avg_rating_of_driver','surge_pct','avg_surge','trips_in_first_30_days']]

df3_tree_clean = df3_dtree.dropna()
df3_tree_clean.describe()


targets= df3_tree_clean[['active']]

# Decision tree require only non-negative values 


predictors = df3_tree_clean[['uber_black_user','tenure','astapor','kings_landing','winterfell','iphone','android','avg_dist','weekday_pct','avg_rating_by_driver','avg_rating_of_driver','surge_pct','avg_surge','trips_in_first_30_days']]

predictors.describe()


classifier = DecisionTreeClassifier(min_samples_split=20, random_state=99, max_depth=5)
pred_train, pred_test, tar_train, tar_test = train_test_split (predictors, targets, test_size = 0.3)

classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)

# Confusion matrix
sklearn.metrics.confusion_matrix(tar_test, predictions)

sklearn.metrics.accuracy_score(tar_test, predictions)




# Decision tree with 10-fold cross validation
from sklearn.cross_validation import KFold
cv = KFold(predictors.shape[0], 10, shuffle=True, random_state=33)

scores = cross_val_score(classifier, predictors, targets, scoring = 'accuracy', cv=cv)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



###############
# Charts

tar_pred = classifier.predict_proba(pred_test)[:, 1]
fpr, tpr, _ = roc_curve(tar_test, tar_pred)



# ROC Curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Decision Tree')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
# classifier_enc=OneHotEncoder()
# classifier.fit(classifier_enc.transform(predictors), targets)


from sklearn import tree
from io import StringIO
# from IPython.display import Image
from PIL import Image
dotfile = StringIO()
import pydotplus 

import urllib, re, os, sys, time 
os.system('cls' if os.name == 'nt' else 'clear') 

chart_path = "/Users/anayankovsky/Desktop/UberTask/aUberPythonTask/Charts"
os.chdir(chart_path)


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image

dtree = StringIO()
tree.export_graphviz(classifier, out_file=dtree, feature_names = predictors.columns)
graph = pydot.graph_from_dot_data(dtree.getvalue())
graph.write_svg('Part3_dtree_all_Python.svg')






################ 
# Decsion tree classification detailed rules
def get_code(tree, feature_names):
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node):
                if (threshold[node] != -2):
                        print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
                        if left[node] != -1:
                                recurse (left, right, threshold, features,left[node])
                        print "} else {"
                        if right[node] != -1:
                                recurse (left, right, threshold, features,right[node])
                        print "}"
                else:
                        print "return " + str(value[node])

        recurse(left, right, threshold, features, 0)

get_code(classifier, predictors.columns)

# Decsion tree split aggregation/viualization 
def get_lineage(tree, feature_names):
     left      = tree.tree_.children_left
     right     = tree.tree_.children_right
     threshold = tree.tree_.threshold
     features  = [feature_names[i] for i in tree.tree_.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]     

     def recurse(left, right, child, lineage=None):          
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     for child in idx:
          for node in recurse(left, right, child):
               print node

get_lineage(classifier, predictors.columns)

###################################################################
# Analysis for Astapor
df3_city= df3_tree_clean[df3_tree_clean["astapor"]==1]


targets= df3_city[['active']]

# Decision tree require only non-negative values 


predictors = df3_city[['uber_black_user','tenure','iphone','android','avg_dist','weekday_pct','avg_rating_by_driver','avg_rating_of_driver','surge_pct','avg_surge','trips_in_first_30_days']]

predictors.describe()


classifier = DecisionTreeClassifier(min_samples_split=20, random_state=99, max_depth=5)
pred_train, pred_test, tar_train, tar_test = train_test_split (predictors, targets, test_size = 0.3)

classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)

# Confusion matrix
sklearn.metrics.confusion_matrix(tar_test, predictions)

sklearn.metrics.accuracy_score(tar_test, predictions)

# classifier_enc=OneHotEncoder()
# classifier.fit(classifier_enc.transform(predictors), targets)



# Decision tree with 10-fold cross validation
from sklearn.cross_validation import KFold
cv = KFold(predictors.shape[0], 10, shuffle=True, random_state=33)

scores = cross_val_score(classifier, predictors, targets, scoring = 'accuracy', cv=cv)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



###############
# Charts

dtree = StringIO()
tree.export_graphviz(classifier, out_file=dtree, feature_names = predictors.columns)
graph = pydot.graph_from_dot_data(dtree.getvalue())
graph.write_svg('Part3_dtree_astapor_Python.svg')

############
# Decision tree details
get_code(classifier, predictors.columns)

###################################################################
# Analysis for King's Landing
df3_city= df3_tree_clean[df3_tree_clean["kings_landing"]==1]


targets= df3_city[['active']]

# Decision tree require only non-negative values 


predictors = df3_city[['uber_black_user','tenure','iphone','android','avg_dist','weekday_pct','avg_rating_by_driver','avg_rating_of_driver','surge_pct','avg_surge','trips_in_first_30_days']]

predictors.describe()


classifier = DecisionTreeClassifier(min_samples_split=20, random_state=99, max_depth=5)
pred_train, pred_test, tar_train, tar_test = train_test_split (predictors, targets, test_size = 0.3)

classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)

# Confusion matrix
sklearn.metrics.confusion_matrix(tar_test, predictions)

sklearn.metrics.accuracy_score(tar_test, predictions)

# classifier_enc=OneHotEncoder()
# classifier.fit(classifier_enc.transform(predictors), targets)



# Decision tree with 10-fold cross validation
from sklearn.cross_validation import KFold
cv = KFold(predictors.shape[0], 10, shuffle=True, random_state=33)

scores = cross_val_score(classifier, predictors, targets, scoring = 'accuracy', cv=cv)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



###############
# Charts

dtree = StringIO()
tree.export_graphviz(classifier, out_file=dtree, feature_names = predictors.columns)
graph = pydot.graph_from_dot_data(dtree.getvalue())
graph.write_svg('Part3_dtree_kings_landing_Python.svg')

############
# Decision tree details
get_code(classifier, predictors.columns)


###################################################################
# Analysis for Winterfell
df3_city= df3_tree_clean[df3_tree_clean["winterfell"]==1]


targets= df3_city[['active']]

# Decision tree require only non-negative values 


predictors = df3_city[['uber_black_user','tenure','iphone','android','avg_dist','weekday_pct','avg_rating_by_driver','avg_rating_of_driver','surge_pct','avg_surge','trips_in_first_30_days']]

predictors.describe()


classifier = DecisionTreeClassifier(min_samples_split=20, random_state=99, max_depth=5)
pred_train, pred_test, tar_train, tar_test = train_test_split (predictors, targets, test_size = 0.3)

classifier.fit(pred_train, tar_train)
predictions = classifier.predict(pred_test)

# Confusion matrix
sklearn.metrics.confusion_matrix(tar_test, predictions)

sklearn.metrics.accuracy_score(tar_test, predictions)

# classifier_enc=OneHotEncoder()
# classifier.fit(classifier_enc.transform(predictors), targets)



# Decision tree with 10-fold cross validation
from sklearn.cross_validation import KFold
cv = KFold(predictors.shape[0], 10, shuffle=True, random_state=33)

scores = cross_val_score(classifier, predictors, targets, scoring = 'accuracy', cv=cv)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))



###############
# Charts

dtree = StringIO()
tree.export_graphviz(classifier, out_file=dtree, feature_names = predictors.columns)
graph = pydot.graph_from_dot_data(dtree.getvalue())
graph.write_svg('Part3_dtree_winterfell_Python.svg')

############
# Decision tree details
get_code(classifier, predictors.columns)
