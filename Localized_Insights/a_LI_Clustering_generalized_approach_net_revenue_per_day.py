
# coding: utf-8

# In[1]:


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


#/data/ml01/eugeney/data/BDorstSample/DataJoint.csv

file_name = '/DataJoint.csv'
file_in = path + file_name

print file_in

df = pd.read_csv(file_in, delimiter=',', error_bad_lines=False, engine='c', nrows=100000000,
                usecols=["TO_CHAR.TM_DIM_KY_DTE..YYYYMMDD..", "SKU_DESC", "STR_ID", "PRICE", "SALES_UNITS","SALES_NET_DLRS"],
                dtype=
                 { 'TO_CHAR.TM_DIM_KY_DTE..YYYYMMDD..': np.str, 'SKU_DESC': np.str, 'STR_ID': np.str, 'PRICE': np.str, 
                  'SALES_NET_DLRS': np.float, 'SALES_CUST_DLRS': np.float, 'SALES_REG_DLRS': np.float, 'SALES_UNITS': np.float}
                 )
print df.shape
print df.dtypes 


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


# In[2]:


df_agg = pd.read_csv(path+'/dffselect_agg.txt', delimiter=',', error_bad_lines=False, engine='c')
df_agg.head()


# In[3]:


# Analysis for net revenue per day
net_rev_pivot = df_agg.pivot_table(values = ["net_revenue_per_day"], columns = ["SKU_DESC_period"], index = ["STR_ID"], aggfunc= np.sum)
print net_rev_pivot


# In[4]:


# Analysis for net revenue per day
nrv_df = pd.DataFrame(net_rev_pivot.to_records())
nrv_df.columns = [hdr.replace("('net_revenue_per_day', ", "").replace(")", "")                      for hdr in nrv_df.columns]
#nrv_df.index = nrv_df["STR_ID"]
print nrv_df


# In[5]:


print len(nrv_df)
print len(list(nrv_df))
print nrv_df.dtypes


# In[7]:


col = list(nrv_df)
#col.remove('Unnamed: 0')
print col
s = pd.DataFrame(col, columns=['column'])
print s


# In[8]:


l = len(list(nrv_df))
for i in range(1,l+1):
    s['column'][i-1:i] = s['column'][i-1:i].str.replace('(.......)',"'C"+str(i-1) + '_____',1)    
print s['column']   


# In[13]:


dfList = s['column'].tolist()
print dfList


# In[14]:


# Analysis for net revenue per day
nrv_df.to_csv(path+'/nrv_dff.txt')


# In[17]:


# Analysis for net revenue per day
nrv_df = pd.read_csv(path+'/nrv_dff.txt', names= dfList, skiprows=1)
print nrv_df.head()
print nrv_df.describe()


# Hierarchical clustering approach 

# In[18]:


np.random.seed(4711)  # for repeatability of this tutorial
X = nrv_df
print X.shape  
X.fillna(0, inplace=True)
X.describe()


# Hierarchical cluster analysis 
# following
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

# In[19]:


print "Number of the predictors' variables in the data set:", len(list(X))


# In[20]:


from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# scipy.cluster.hierarchy.linkage
Z = linkage(X, 'ward')


# Another thing you can and should definitely do is check the Cophenetic Correlation Coefficient of your clustering with help of the cophenet() function. This (very very briefly) compares (correlates) the actual pairwise distances of all your samples to those implied by the hierarchical clustering. The closer the value is to 1, the better the clustering preserves the original distances, which in our case is pretty close:

# In[21]:


from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
c


# # Elbow method to define tree's depth:
# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/#Selecting-a-Distance-Cut-Off-aka-Determining-the-Number-of-Clusters

# In[22]:


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

# In[23]:


no_clusters =12


# In[24]:


# Increadsing recursion limit
import sys
sys.setrecursionlimit(10000)

# Alternatively
# pip install -u setuptools pip


# In[25]:


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


# In[26]:


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


# In[27]:


list1= min(ddata['dcoord'])
list1.remove(0.0)
list1.remove(0.0)
minlist = min(list1)
print "Cluster threshold = ", minlist



# In[28]:


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


# In[29]:


# Retrieving the clusters
from scipy.cluster.hierarchy import fcluster

clusters = fcluster(Z, max_d, criterion='distance')
print clusters
print len(clusters)

cluster_df =pd.DataFrame(clusters)                         
                         
print  "Number of cluster", np.sort(cluster_df[0].unique())


# In[30]:


dfc = pd.DataFrame(np.column_stack((nrv_df["STR_ID"], clusters)), columns = ['STR_ID', "cluster"])


# In[31]:


print dfc.head()


# In[32]:


# Reporting on the 


# In[33]:


df_merge = nrv_df.merge(dfc, on='STR_ID', how = "outer") 


# In[34]:


df_merge.head(10)


# Visualization of the results

# In[35]:


import pylab as pl
get_ipython().magic(u'matplotlib inline')


# In[37]:


df_agg = pd.read_csv(path+'/dffselect_agg.txt',
usecols=["STR_ID", "period", "SKU_DESC", "SALES_NET_DLRS", "SALES_UNITS", "sales_days", "net_revenue", "net_revenue_per_day"])
df_agg.head(5)


# In[38]:


df_merge = df_agg.merge(dfc, on='STR_ID', how = "outer")


# In[39]:


df_merge.sort_values(by=(["cluster"]), ascending=True) 
print df_merge.dtypes

#df.sort_values(by=('Labs', 'II'), ascending=False)


# In[40]:


print df_merge.head()


# In[41]:


df_agr = df_merge[['STR_ID','period','cluster', "SKU_DESC", "net_revenue","sales_days"]].groupby(["SKU_DESC", 'STR_ID','period','cluster']).sum().reset_index()
df_agr["net_revenue_per_day"] =  df_agr["net_revenue"]/df_agr["sales_days"] 
df_agr.head(10)


# In[107]:


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

#print df_agr["Date"].dtype


# In[91]:


df_agr = df_agr.set_index(pd.DatetimeIndex(df_agr['Date']))
print df_agr.head()


# In[44]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')


# In[97]:


# Data preparation for visualization for net revenue

df_cluster_agg_pd  = df_agr[["cluster","net_revenue_per_day"]].groupby(['cluster','Date']).mean().reset_index()
#print df_cluster_agg_pd

df_store_agg_pd = df_agr[["cluster", "STR_ID","net_revenue_per_day"]].groupby(["cluster", "STR_ID", "Date"]).mean().reset_index()
#print df_store_agg


# In[93]:


# Analysis for Net Revenue per day
fig = df_agr.boxplot("net_revenue_per_day", by="cluster",  vert=True) 
plt.title('Boxplot. Distribution of net revenue in clusters per SKU family')
plt.suptitle('')
plt.show()


# In[103]:


# Analysis for Net Revenue (average)

def visio_avg(ymax0, rep0, cluster0):
    ymax=ymax0 
    rep = rep0
    cluster=cluster0
    step =6

    lower = 0 + step*rep
    upper =6 + step*rep

    color_list = ["blue","green","cyan","magenta","yellow","black"]

    labels = ['1', '2', '3', '4']
    get_ipython().magic(u'matplotlib inline')
    fig = plt.figure()
    ax = plt.subplot(111)  
    axes = plt.gca()
    axes.set_ylim([0,ymax])
    plt.grid(True)
 
    count =0
    for i in store_list[lower:upper]:
        table = "df_s_"+str(i)
        table= df_store_agg_pd[(df_store_agg_pd['cluster']==cluster) & (df_store_agg_pd['STR_ID']== i)]
        
        fig=ax.plot(table["Date"],table["net_revenue_per_day"], marker='*', color=color_list[count], label =table['STR_ID'].unique())
        count = count +1
                                                 
                                                           
    fig=ax.plot(df_c["Date"],df_c["net_revenue_per_day"], marker='+',color='red', linewidth=3, label =df_c['cluster'].unique())
    title = "Stores' profiles in Average Net Revenue Per Day performance in cluster #"+ str(cluster) 
    plt.title(title)


    plt.xlabel("Quarter")
    plt.ylabel('Average net revenue per day')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=5)
    plt.show()


# In[104]:


no_clusters = 12


# In[105]:


for j in range(1, no_clusters):
    
    df_c = df_cluster_agg_pd[(df_cluster_agg_pd['cluster']==j)]
    df_s = df_store_agg_pd[(df_store_agg_pd['cluster']==j)]
    
    y_max_avg = df_s["net_revenue_per_day"].max()
    
    print "Report on stores' profile in cluster %s" %(j)
    print '------------------------------------------'
    print "Number of stores", len(df_s["STR_ID"].unique())
    
    print 'Store average net revenue in a cluster: median, mean, std, max' 
    print df_s["net_revenue_per_day"].median()
    print df_s["net_revenue_per_day"].mean()
    print df_s["net_revenue_per_day"].std()
    print df_s["net_revenue_per_day"].max()

    store_list = df_s["STR_ID"].unique()
    print "List of stores", store_list

    ncharts = np.round(len(df_s["STR_ID"].unique())/6)
    for i in range(2):
        for i in range(ncharts):
            visio_avg(ymax0 =y_max_avg, rep0=i, cluster0 = j)


# In[106]:


get_ipython().system(u'jupyter nbconvert --to script a_LI_Clustering_2017_generalized_approach_per_day.ipynb')


# In[ ]:




