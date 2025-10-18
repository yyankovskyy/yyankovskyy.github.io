
# coding: utf-8

# In[3]:


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
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3, suppress=True) 

import pandas as pd
import numpy as np
from itertools import islice
path ='/data/ml01/eugeney/data/BDorstSample'


# # Data preprocessing step that was ran once 

# In[ ]:


import pymongo 
import json

chunksize = 100000
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client['SQLAExport']
collection = db['transactions']

file = '/data/ml01/eugeney/data/BDorstSample/SQLAExport.txt'

for df in pd.read_csv(file, chunksize=chunksize, iterator=True):
    collection.insert_many(df.to_dict('records'))

print df.shape
print df.dtypes 


# #/data/ml01/eugeney/data/BDorstSample/DataJoint.csv
# 
# file_name = '/DataJoint.csv'
# file_in = path + file_name
# 
# print file_in
# 
# df = pd.read_csv(file_in, delimiter=',', error_bad_lines=False, engine='c', nrows=100000000,
#                 usecols=["TO_CHAR.TM_DIM_KY_DTE..YYYYMMDD..", "SKU_DESC", "STR_ID", "PRICE", "SALES_UNITS","SALES_NET_DLRS"],
#                 dtype=
#                  { 'TO_CHAR.TM_DIM_KY_DTE..YYYYMMDD..': np.str, 'SKU_DESC': np.str, 'STR_ID': np.str, 'PRICE': np.str, 
#                   'SALES_NET_DLRS': np.float, 'SALES_CUST_DLRS': np.float, 'SALES_REG_DLRS': np.float, 'SALES_UNITS': np.float}
#                  )
# print df.shape
# print df.dtypes 
# 

# # Selection parameters to enter below:
# 1. Time period in the format like Q1_2016
# 2. Search parameter for a SKU family
# 3. Maximum number of top revenue SKU_groups 
# 

# In[ ]:


print ("Hi, \nWelcome to Kohl's!")
#period_list = input("Please, enter the period of interests in the format followed following the example: \n'Q1_2017', 'Q2_2017'\n >")
period_list =('Q1_2016', 'Q2_2016', 'Q3_2016', 'Q4_2016', 'Q1_2017', 'Q2_2017', 'Q3_2017', 'Q4_2017')
print period_list


# In[ ]:


#SKU_list = input("Please, enter the SKU category you want to search the database in the format followed following the example: 'ROBE'\n >")
SKU_list = 'ROBE'
print SKU_list


# In[10]:


#N_top_element = input("Please, enter number of top revenue elements in the format following the example: \n10\n >")
N_top_element = 20
print N_top_element


# In[ ]:


df["sales_dte"] = df["TO_CHAR.TM_DIM_KY_DTE..YYYYMMDD.."].apply(lambda x: pd.to_datetime(str(x), format ="%Y%m%d"))
df["period"] ='Q'+df["sales_dte"].dt.quarter.astype(str) + '_' + df["sales_dte"].dt.year.astype(str)
df['sales_days'] =1

print df["period"].unique()
df=df.loc[df['period'].isin(period_list)]

print df["period"].unique()


# In[ ]:


df["SKU_DESC"] = df["SKU_DESC"].str.strip()
df1= df[df['SKU_DESC'].apply(lambda x: any(pd.Series(x).str.contains(SKU_list)))]
df1["net_revenue"] = df1["SALES_UNITS"]*df1["SALES_NET_DLRS"]
print 'The data search limited by the your selection criteria resultes in the following data selection'
print 'A list of the SKU families: \n', df1["SKU_DESC"].unique()
print "\nSample size in sales days: ", len(df1)
print '\nData sample summary: \n', df1.describe()


# In[ ]:


# When reaching large size
file_out = path + '/LI_selection.txt'
df1.to_csv(file_out, header=True, index=None, sep=',')


# In[7]:


file_in = path + '/LI_selection.txt'
df1 = pd.read_csv(file_in, delimiter=',', error_bad_lines=False, engine='c')
#usecols=["SKU_DESC", "STR_ID", "PRICE", "SALES_UNITS","SALES_NET_DLRS", "SALES_UNITS", "period", "sales_days"])
print df1.head()
df1.dtypes


# In[11]:


# Extra selection analysis
# Calculate return by SKU_DESC

agg_stat = df1.groupby(['SKU_DESC'])["net_revenue"].sum().reset_index()
agg_stat = agg_stat.sort_values(["net_revenue"],ascending=[0])
agg_stat_top = agg_stat.head(N_top_element)
print agg_stat_top


# In[12]:


path ='/data/ml01/eugeney/data/BDorstSample/'
file_name = 'LI_selection.txt'
file_in = path + file_name
print file_in


# In[13]:


df0 = pd.read_csv(file_in, delimiter=',', error_bad_lines=False, engine='c', 
usecols=["SKU_DESC", "STR_ID", "PRICE", "SALES_UNITS","SALES_NET_DLRS", "SALES_UNITS", "period", "sales_days"])
print df0.head()
df0.dtypes


# In[14]:


dff=df0.merge(agg_stat_top, how = "inner")
dff.head()


# In[15]:


dff["net_revenue"] = dff["SALES_UNITS"]*dff["SALES_NET_DLRS"]
df_agg = dff.groupby(['STR_ID','period','SKU_DESC']).sum().reset_index()
df_agg["net_revenue_per_day"] =  df_agg["net_revenue"]/df_agg["sales_days"] 
print df_agg.dtypes
print df_agg.head()


# In[16]:


print 'Sample size: ', len(df_agg)
print 'Sample revenue: ', np.sum(df_agg["net_revenue"])


# In[17]:


print df_agg["SKU_DESC"].unique()
print len(df_agg["SKU_DESC"].unique())


# In[18]:


df_agg["SKU_DESC_period"] = df_agg["SKU_DESC"].str.extract('(.....)',expand = True) + "_" + df_agg["period"].str.extract('(.......)',expand = True)
df_agg.to_csv(path+'/dffselect_agg.txt')
print df_agg["SKU_DESC_period"].head()


# In[19]:


df_agg = pd.read_csv(path+'/dffselect_agg.txt', delimiter=',', error_bad_lines=False, engine='c')
df_agg.head()


# # Analysis for net revenue per day
# net_rev_pivot = df_agg.pivot_table(values = ["net_revenue_per_day"], columns = ["SKU_DESC_period"], index = ["STR_ID"], aggfunc= np.sum)
# print net_rev_pivot

# In[20]:


# Analysis for net revenue
net_rev_pivot = df_agg.pivot_table(values = ["net_revenue"], columns = ["SKU_DESC_period"], index = ["STR_ID"], aggfunc= np.sum)
print net_rev_pivot


# # Analysis for net revenue per day
# nrv_df = pd.DataFrame(net_rev_pivot.to_records())
# nrv_df.columns = [hdr.replace("('net_revenue_per_day', ", "").replace(")", "") \
#                      for hdr in nrv_df.columns]
# #nrv_df.index = nrv_df["STR_ID"]
# print nrv_df

# In[21]:


# Analysis for net revenue
nrv_df = pd.DataFrame(net_rev_pivot.to_records())
nrv_df.columns = [hdr.replace("('net_revenue', ", "").replace(")", "")                      for hdr in nrv_df.columns]
#nrv_df.index = nrv_df["STR_ID"]
print nrv_df


# In[22]:


print len(nrv_df)
print len(list(nrv_df))
print nrv_df.dtypes


# In[23]:


col = list(nrv_df)
#col.remove('Unnamed: 0')
print col
s = pd.DataFrame(col, columns=['column'])
print s


# In[24]:


l = len(list(nrv_df))
for i in range(1,l+1):
    s['column'][i-1:i] = s['column'][i-1:i].str.replace('(.......)',"'C"+str(i-1) + '_____',1)    
print s['column']   


# In[25]:


dfList = s['column'].tolist()
print dfList


# # Analysis for net revenue per day
# nrv_df.to_csv(path+'/nrv_dff.txt')

# In[26]:


# Analysis for net revenue
nrv_df.to_csv(path+'/nrv_dff_nr_select.txt')


# # Analysis for net revenue per day
# nrv_df = pd.read_csv(path+'/nrv_dff.txt')
# print nrv_df.head()
# print nrv_df.describe()

# In[27]:


# Analysis for net revenue
   
nrv_df = pd.read_csv(path+'/nrv_dff_nr_select.txt', names= dfList, skiprows=1)
#nrv_df = pd.read_csv(path+'/nrv_dff_nr_select.txt', skiprows=1)
print nrv_df.head()
print nrv_df.describe()


# Hierarchical clustering approach 

# In[28]:


np.random.seed(4711)  # for repeatability of this tutorial
X = nrv_df
print X.shape  
X.fillna(0, inplace=True)
X.describe()


# Hierarchical cluster analysis 
# following
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

# In[29]:


print "Number of the predictors' variables in the data set:", len(list(X))


# In[30]:


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# scipy.cluster.hierarchy.linkage
Z = linkage(X, 'ward')


# Another thing you can and should definitely do is check the Cophenetic Correlation Coefficient of your clustering with help of the cophenet() function. This (very very briefly) compares (correlates) the actual pairwise distances of all your samples to those implied by the hierarchical clustering. The closer the value is to 1, the better the clustering preserves the original distances, which in our case is pretty close:

# In[31]:


from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
c


# # Elbow method to define tree's depth:
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Selecting-a-Distance-Cut-Off-aka-Determining-the-Number-of-Clusters

# In[32]:


last = Z[-18:, 2]
last_rev = last[::-1]
idxs = np.arange(1, len(last) + 1)
plt.plot(idxs, last_rev)

acceleration = np.diff(last, 2)  # 2nd derivative of the distances
acceleration_rev = acceleration[::-1]
plt.plot(idxs[:-2] + 1, acceleration_rev)
plt.title("Diagnostic chart. Elbow method results for selecting an optimum number of clusters")
plt.show()

k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
print "Optimal number of clusters by Elbow method:", k
print "Conservative number of clusters: 12"


# # 12 is assumed to be a number of clusters 
# #since 3 own reserch versions shown that 12 is often picked up as a conservative number of the clusters  

# In[33]:


no_clusters =12


# In[34]:


# Increadsing recursion limit
import sys
sys.setrecursionlimit(10000)

# Alternatively
# pip install -u setuptools pip


# In[35]:


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('cluster size (stores)')
        plt.ylabel('distance between the cluster centers')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


# In[36]:


ddata = fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=no_clusters,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()

print ddata


# In[37]:


list1= min(ddata['dcoord'])
list1.remove(0.0)
list1.remove(0.0)
minlist = min(list1)
print "Cluster threshold = ", minlist



# In[38]:


max_d = minlist-0.000001  # max_d as in max_distance
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=no_clusters,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10
    #, max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()


# In[39]:


# Retrieving the clusters
from scipy.cluster.hierarchy import fcluster

clusters = fcluster(Z, max_d, criterion='distance')
print clusters
print len(clusters)

cluster_df =pd.DataFrame(clusters)                         
                         
print  "Number of cluster", np.sort(cluster_df[0].unique())


# In[40]:


dfc = pd.DataFrame(np.column_stack((nrv_df["STR_ID"], clusters)), columns = ['STR_ID', "cluster"])


# In[41]:


print dfc.head()


# In[42]:


# Reporting on the 


# In[43]:


df_merge = nrv_df.merge(dfc, on='STR_ID', how = "outer") 


# In[44]:


df_merge.head(10)


# Visualization of the results

# In[45]:


import pylab as pl
get_ipython().magic(u'matplotlib inline')


# In[46]:


df_agg = pd.read_csv(path+'dffselect_agg.txt',
usecols=["STR_ID", "period", "SKU_DESC", "SALES_NET_DLRS", "SALES_UNITS", "sales_days", "net_revenue", "net_revenue_per_day"])
df_agg.head(5)


# In[47]:


df_merge = df_agg.merge(dfc, on='STR_ID', how = "outer")


# In[48]:


df_merge.sort_values(by=(["cluster"]), ascending=True) 
print df_merge.dtypes

#df.sort_values(by=('Labs', 'II'), ascending=False)


# In[112]:


#print df_merge.head()


# In[139]:


df_agr = df_merge[['STR_ID','period','cluster', "SKU_DESC", "net_revenue","sales_days"]].groupby(["SKU_DESC", 'STR_ID','period','cluster']).sum().reset_index()
df_agr["net_revenue_per_day"] =  df_agr["net_revenue"]/df_agr["sales_days"] 
#df_agr.head(10)


# In[152]:


import datetime as dt

df_agr["Date"] = dt.date(2014,12,31)

df_agr["Date"] = np.where(df_agr["period"]=="Q1_2015", dt.date(2015,3,31), df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q2_2015", dt.date(2015,6,30),df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q3_2015", dt.date(2015,9,30), df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q4_2015", dt.date(2015,12,31), df_agr["Date"])

df_agr["Date"] = np.where(df_agr["period"]=="Q1_2016", dt.date(2016,3,31),df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q2_2016", dt.date(2016,6,30),df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q3_2016", dt.date(2016,9,30),df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q4_2016", dt.date(2016,12,31),df_agr["Date"])

df_agr["Date"] = np.where(df_agr["period"]=="Q1_2017", dt.date(2017,3,31), df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q2_2017", dt.date(2017,6,30),df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q3_2017", dt.date(2017,9,30),df_agr["Date"])
df_agr["Date"] = np.where(df_agr["period"]=="Q4_2017", dt.date(2016,12,11),df_agr["Date"])

print df_agr["Date"].head()
print df_agr["Date"].dtypes


# In[149]:


df_agr = df_agr.set_index(pd.DatetimeIndex(df_agr['Date']))
print df_agr.head()


# In[150]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# In[151]:


print df_agr.head()
print df_agr.dtypes


# In[153]:


# Data preparation for visualization for net revenue

df_cluster_agg  = df_agr[["cluster","net_revenue", "Date"]].groupby(['cluster', 'Date']).mean().reset_index()
print df_cluster_agg

df_store_agg = df_agr[["cluster", "STR_ID","net_revenue", "Date"]].groupby(["cluster", "STR_ID", "Date"]).mean().reset_index()
print df_store_agg


df_store_tot = df_agr[["cluster", "STR_ID","net_revenue", "Date"]].groupby(["cluster","STR_ID", 'Date']).sum().reset_index()
print df_store_tot


# In[154]:


print "Net revenue clusters distribution moments report: median, mean, std"
print df_agr[["cluster","net_revenue"]].groupby(['cluster']).median().reset_index()
print df_agr[["cluster","net_revenue"]].groupby(['cluster']).mean().reset_index()
print df_agr[["cluster","net_revenue"]].groupby(['cluster']).std().reset_index()


# In[160]:


# Analysis for Net Revenue (total)

def visio_tot(ymax0, rep0, cluster0):
    ymax=ymax0 
    rep = rep0
    cluster=cluster0
    step =6

    lower = 0 + step*rep
    upper =6 + step*rep

    color_list = ["blue","green","cyan","magenta","yellow","black"]

    #labels = ['1', '2', '3', '4']
    get_ipython().magic(u'matplotlib inline')
    fig = plt.figure()
    ax = plt.subplot(111)  
    axes = plt.gca()
    axes.set_ylim([0,ymax])
    plt.grid(True)

    count =0
    for i in store_list[lower:upper]:
        table = "df_t_"+str(i)
        table= df_store_tot[(df_store_tot['cluster']==cluster) & (df_store_tot['STR_ID']== i)]
            
        #fig=ax.plot(table["Quarter"],table["net_revenue"], marker='*', color=color_list[count], label =table['STR_ID'].unique())
        fig=ax.plot(table["Date"],table["net_revenue"], marker='*', color=color_list[count], label =table['STR_ID'].unique())
        count = count +1
                                                 
    #plt.xticks(df_t["Quarter"], labels)
    title = "Stores' profiles in Total Net Revenue performance in cluster #"+ str(cluster) 
    plt.title(title)


    plt.xlabel("Quarter")
    plt.ylabel('Total net revenue per store')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
    plt.show()


# In[165]:


# Analysis for Net Revenue (average)

def visio_avg(ymax0, rep0, cluster0):
    ymax=ymax0 
    rep = rep0
    cluster=cluster0
    step =6

    lower = 0 + step*rep
    upper =6 + step*rep

    color_list = ["blue","green","cyan","magenta","yellow","black"]

    #labels = ['1', '2', '3', '4']
    get_ipython().magic(u'matplotlib inline')
    fig = plt.figure()
    ax = plt.subplot(111)  
    axes = plt.gca()
    axes.set_ylim([0,ymax])
    plt.grid(True)
 
    count =0
    for i in store_list[lower:upper]:
        table = "df_s_"+str(i)
        table= df_store_agg[(df_store_agg['cluster']==cluster) & (df_store_agg['STR_ID']== i)]
            
        #fig=ax.plot(table["Quarter"],table["net_revenue"], marker='*', color=color_list[count], label =table['STR_ID'].unique())
        fig=ax.plot(table["Date"],table["net_revenue"], marker='*', color=color_list[count], label =table['STR_ID'].unique())
        count = count +1
                                                 
                                                           
    fig=ax.plot(df_c["Date"],df_c["net_revenue"], marker='+',color='red', linewidth=3, label =df_c['cluster'].unique())
    #fig=ax.plot(df_c["Quarter"],df_c["net_revenue"], marker='+',color='red', linewidth=3, label =df_c['cluster'].unique())
    #plt.xticks(df_s["Date"])
    title = "Stores' profiles in Average Net Revenue performance in cluster #"+ str(cluster) 
    plt.title(title)


    plt.xlabel("Quarter")
    plt.ylabel('Average net revenue per store')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
    plt.show()


# In[166]:


#nclusters=len(df_s["cluster"].unique())
#print nclusters
no_clusters=12
print no_clusters


# In[167]:


df_cluster_agg.head()


# In[168]:


for j in range(1, no_clusters):
    
    df_c = df_cluster_agg[(df_cluster_agg['cluster']==j)]
    df_s = df_store_agg[(df_store_agg['cluster']==j)]
    df_t = df_store_tot[(df_store_agg['cluster']==j)]
    
    y_max_tot = df_t["net_revenue"].max()
    y_max_avg = df_s["net_revenue"].max()
    
    print "Report on stores' profile in cluster %s" %(j)
    print '------------------------------------------'
    print "Number of stores", len(df_s["STR_ID"].unique())
    
    print 'Store average net revenue in a cluster: median, mean, std, max' 
    print df_s["net_revenue"].median()
    print df_s["net_revenue"].mean()
    print df_s["net_revenue"].std()
    print df_s["net_revenue"].max()

    print 'Store total net revenue in a cluster: median, mean, std, max'
    print df_t["net_revenue"].median()
    print df_t["net_revenue"].mean()
    print df_t["net_revenue"].std()
    print df_t["net_revenue"].max()

    store_list = df_s["STR_ID"].unique()
    print "List of stores", store_list

    ncharts = np.round(len(df_s["STR_ID"].unique())/6)
    for i in range(2):
        #for i in range(ncharts):
        visio_tot(ymax0 =y_max_tot, rep0=i, cluster0 = j)
        visio_avg(ymax0 =y_max_avg, rep0=i, cluster0 = j)


# In[ ]:


get_ipython().system(u'jupyter nbconvert --to script a_LI_Clustering_2017_generalized_approach_net_revenue_mongodb.ipynb')


# In[ ]:




