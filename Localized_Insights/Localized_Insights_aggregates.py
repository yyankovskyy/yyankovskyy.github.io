# -*- coding: utf-8 -*-
"""
"""

import math 
import StringIO 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from sklearn import metrics 
from sklearn.metrics import confusion_matrix 
from datetime import datetime 
pd.options.display.max_rows = 100
pd.options.display.float_format = '{:.1f}'.format
from itertools import islice
import datetime

file_path = 'F:/aData/LI/' 

# Loading data from various data sources
# For complet list of the data sources refer to
# https://docs.google.com/spreadsheets/d/1OeOInH4sFUkAc4nYcLADJ6nWGkmdGI53MPNj7KetzXo/edit#gid=153317972
###############################################################################
# Data source 1: Kohl's GIV Sales data
# from a csv file 

file_name = 'SQLAExport.txt' 


sku = '95791339
sales = pd.Dataframe(sku)

sample = []
with open(file_path + file_name) as f:
    for line in islice(f, 10):
      np.where(sales['SKU_NBR'] == '99617672', sample.append(line),0)

df = pd.DataFrame(sample)
print df 

column_schema = [        
'TM_DIM_KY_DTE',
'SKU_NBR',
'SKU_DESC',
'STR_ID',
'PRICE',
'SALES_NET_DLRS',
'SALES_CUST_DLRS',
'SALES_REG_DLRS',
'SALES_UNITS',
'BOH_DLRS',
'BOH_UNITS',
'BOH_COST',
'EOH_DLRS',
'EOH_UNITS',
'EOH_COST']

# with open(file_path + file_name) as f:
#    for line in islice(f, 2):
#        print line
  
      
#df = pd.read_csv(StringIO(file_path + file_name), sep="\t", header=True, names=column_schema,
sales = pd.read_csv(file_path + file_name, sep="\t", header=True, names=column_schema,
error_bad_lines=False, low_memory=False,
dtype=
{'TM_DIM_KY_DTE': np.str,
'SKU_NBR': np.str,
'SKU_DESC': np.str,
'STR_ID': np.str,
'PRICE': np.float64,
'SALES_NET_DLRS': np.float64, 
'SALES_CUST_DLRS': np.float64,
'SALES_REG_DLRS': np.float64,
'SALES_UNITS': np.int64,
'BOH_DLRS': np.float64,
'BOH_UNITS': np.int64,
'BOH_COST': np.float64,
'EOH_DLRS': np.float64,
'EOH_UNITS': np.float64,
'EOH_COST': np.float64}
, nrows = 100000)

# nrows = 100000
#df['Date'] = datetime.datetime.strftime(df['TM_DIM_KY_DTE'],'%Y%m%d').strftime('%m-%d-%Y')


sales['Date'] =pd.to_datetime(sales['TM_DIM_KY_DTE'])


print sales
print sales.dtypes
 #20160423        99547093        MICRO WITH LACE WAIST HIPSTER   377     9.00    0.00    0.00    0.00    0       36.00   4       9.000   36.00   4       9.000

# Search for most popular item in a sample
df_count = sales.groupby('SKU_DESC').count()
df_freq = sales.groupby('SKU_DESC').size() 
df_freq.describe()


# Highest frequency 
# SKU_DESC:
# BALCONETTE PUSH UP w/LACE         4905

df_select=sales[sales['SKU_DESC'].map(str.strip) == 'BALCONETTE PUSH UP w/LACE']
select_freq = df_select.groupby('SKU_NBR').size()

# Sample selection
import numpy as np 
import pandas as pd 
file_path = 'F:/aData/LI/' 

file_name = 'SQLAExport.txt' 

column_schema = [        
'TM_DIM_KY_DTE',
'SKU_NBR',
'SKU_DESC',
'STR_ID',
'PRICE',
'SALES_NET_DLRS',
'SALES_CUST_DLRS',
'SALES_REG_DLRS',
'SALES_UNITS',
'BOH_DLRS',
'BOH_UNITS',
'BOH_COST',
'EOH_DLRS',
'EOH_UNITS',
'EOH_COST']

reader = pd.read_csv(file_path + file_name, sep="\t", header=True, names=column_schema,
error_bad_lines=False, low_memory=False,
dtype=
{'TM_DIM_KY_DTE': np.str,
'SKU_NBR': np.str,
'SKU_DESC': np.str,
'STR_ID': np.str,
'PRICE': np.float64,
'SALES_NET_DLRS': np.float64, 
'SALES_CUST_DLRS': np.float64,
'SALES_REG_DLRS': np.float64,
'SALES_UNITS': np.int64,
'BOH_DLRS': np.float64,
'BOH_UNITS': np.int64,
'BOH_COST': np.float64,
'EOH_DLRS': np.float64,
'EOH_UNITS': np.float64,
'EOH_COST': np.float64}
,
iterator=True, chunksize=1000) 


 # nrows = 10)
i=0
for chunk in reader:
    i+=1
    print i
    sample = pd.concat([chunk[chunk['SKU_NBR'] =='99617672']])
   
df=reader.get_chunk(1000000)
df.describe()
df.head()
   
   
#sample = pd.concat([chunk[chunk['SKU_NBR'] =='99617672'] for chunk in reader])



# Data source 1.2
# Kohl's stores geo location (postal address, geo coordinates, State, county, locality)
# Note:
# The original data have being changed:
#1)  1st zero-line is added to preserve an original 1st line due to skipping the 1st line by pd.read_csv
#
# 2) Counties' names were found for the folowing stores with counties
# administrative_area_level_2 = NA 
# id administrative_area_level_2
# 53 Douglas County
# 571 Waukesha County 
# 709 Butler County
# 830 Roanoke County 
# 914 Augusta County
# 942 Spotsylvania County
# 944 Frederick County 
# 972 Rockingham County 
# 991 Princess Anne County
# 995 Suffolk County
# 1014 York County
# 1076 Princess Anne County
# 1078 Norfolk Count
  
 
file_path = 'F:/aData/LI/' 
file_name = 'Kohls_stores_addresses_adj.csv' 

column_schema = [        
'store_id',
'longitude',
'latitude',
'type',	
'loctype',
'address',
'north',
'south',
'east',
'west',
'street_number',
'street',
'city',
'county',
'state',
'country',
'postal_code',
'postal_code_suffix',
'neighborhood',
'establishment',
'subpremise',	
'administrative_area_level_3',
'premise',
'city_hall',
'political']

geo = pd.read_csv(file_path + file_name, header=True, names = column_schema,
error_bad_lines=False, low_memory=False,
dtype=
{'store_id': np.str,
'longitude': np.float64,
'latitude': np.float64,
'type': np.str,	
'loctype': np.str,
'address': np.str,
'north': np.float64,
'south': np.float64,
'east': np.float64,
'west': np.float64,
'street_number': np.str,
'street': np.str,
'city': np.str,
'county': np.str,
'state': np.str,
'country': np.str,
'postal_code': np.str,
'postal_code_suffix': np.str,
'neighborhood': np.str,
'establishment': np.str,
'subpremise': np.str,	
'administrative_area_level_3': np.str,
'premise': np.str,
'city_hall': np.str,
'political': np.str
})
 
geo = geo.drop('type', 1)
geo = geo.drop('loctype', 1)
geo = geo.drop('north', 1)
geo = geo.drop('south', 1)
geo = geo.drop('east', 1)
geo = geo.drop('west', 1)
geo = geo.drop('establishment', 1)
geo = geo.drop('subpremise', 1)
geo = geo.drop('administrative_area_level_3', 1)
geo = geo.drop('premise', 1)
geo = geo.drop('city_hall', 1)
geo = geo.drop('political', 1)

# To exclude, a store_id = 11 with NA data
geo = geo.drop(10, 0)
	
print geo.dtypes
 
### Loading weather indicators for 2016 #####################################################
file_path = 'F:/aData/LI/' 
file_name = 'Weather_indicators_2016.csv'
df_weather_2016 = pd.read_csv(file_path + file_name, header=None, 
                              error_bad_lines=False, low_memory=False,
                              nrows = 2)

df_weather_2016['Date'] = df_weather_2016[1].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

df_weather_2016 = df_weather_2016.drop(4, 1)
df_weather_2016 = df_weather_2016.drop(5, 1)
df_weather_2016 = df_weather_2016.drop(6, 1)
df_weather_2016 = df_weather_2016.drop(7, 1)

#print df_weather_2016
print df_weather_2016.dtypes

### Key data statistics
#print min(df_weather_2016['Date'])
#print max(df_weather_2016['Date'])
 
#with open(file_path + file_name) as f:
#    for line in islice(f, 1000000000):
#      print line

 ### Loading weather indicators for 2017 #####################################################

file_name = 'Weather_indicators_2017.csv'
df_weather_2017 = pd.read_csv(file_path + file_name, header=None, 
                              error_bad_lines=False, low_memory=False,
                              nrows = 2)

df_weather_2017['Date'] = df_weather_2017[1].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

df_weather_2017 = df_weather_2017.drop(4, 1)
df_weather_2017 = df_weather_2017.drop(5, 1)
df_weather_2017 = df_weather_2017.drop(6, 1)
df_weather_2017 = df_weather_2017.drop(7, 1)

#print df_weather_2017
print df_weather_2017.dtypes
 
 
# Data source 1.5
# EIPBD (Kohl’s internal data) staff indicators:
# Staff turnover; Associates’ average tenure with Kohl’s; Associate’s average age

# Option 1 using Associate.txt with limited information
#file_name = 'Associate.txt' 

#column_schema = ['LOC_ID', 'associate_id','FIRST_NM','LAST_NM','ORIG_HIR_DTE','TRMNTN_DTE']

#asct = pd.read_csv(file_path + file_name, header=True, names = column_schema,
#error_bad_lines=False, low_memory=False, sep="|",
#dtype=
#{'LOC_ID': np.int32, 'associate_id': np.str, 'FIRST_NM': np.str ,
#'LAST_NM': np.str,'ORIG_HIR_DTE': np.str,'TRMNTN_DTE': np.str
#},
#nrows =10)

#sales['Date'] =pd.to_datetime(sales['TM_DIM_KY_DTE'])
#asct['hire_date'] =pd.to_datetime( asct['ORIG_HIR_DTE']) 
#asct.loc[:, 'termination_date'] = np.where(asct['TRMNTN_DTE']=='?', 
#np.nan, asct['TRMNTN_DTE'].apply(lambda x: pd.to_datetime(str(x))))

#asct= asct.drop('ORIG_HIR_DTE', 1)
#asct= asct.drop('TRMNTN_DTE', 1)

#print asct.dtypes
#asct.describe()
#print asct

# Option 2 using Associate.csv with more information
file_path = 'F:/aData/LI/' 
file_name = 'Associate_adj.csv'

# 1. Duplicate columns are deleted: 
# EI_LOC_ID, FRST_M_NM, LAST_NM
# from the original file 'Associate_adj.csv'
# 2. Extra row is added in the top to preserve the 1st row



column_schema = [
'EI_ASSOC_ID','TM_DIM_KY_DTE','FRST_M_NM','HIR_DTE','ORIG_HIR_DTE','REHIR_DTE',
'TRMNTN_DTE','STRT_DTE','CMPSN_RT_AMT','ANNL_RT_AMT','HRLY_RT_AMT','EMP_JOB_CDE',
'EMPT_STAT_CTG_CDE','EMP_DEPT_ID','EI_LOC_ID','EMPT_TERM_CDE','EMP_TYP_CDE',
'SLRY_ADM_PLN_NBR','PAY_GRD_CDE','HMN_RSRC_STAT_CDE','EMP_STAT_CDE','REHIR_IND',
'EI_REPLT_LOC_ID','LOC_ID','SLS_INCLN_DTE_TYP_CDE','ORD_STP_DAY_CNT',
'EXTD_MALL_NM','ORD_STRT_DAY_CNT','STR_MGR_NM','STR_FMT_CDE','TERR_ID',
'REGN_ID','DIST_ID','PLNG_CTG_TYP_CDE','TOT_STR_SQ_FTG_QTY','TOT_SLG_SQ_FTG_QTY',
'E_BUS_IND','DST_IND','PRCH_IND','IN_STR_PRCH_IND','STR_DIRECTIONS_TXT',
'OPNG_EVNT_YR_NBR','OPNG_EVNT_MN_NBR','STR_ACQRD_DTE','GRND_OPNG_DTE','FINCL_CLO_DTE',
'RMDL_DTE','LOC_NM','LOC_10_ABBR_NM','LOC_5_ABBR_NM',
'LOC_TYP_CDE','PRNT_LOC_ID','TMZN_CDE','LOC_OPN_DTE','LOC_OCPNC_DTE',
'LOC_CLO_DTE','LNE_1_ADDR','LNE_2_ADDR','CTY_NM','ST_PROV_CDE','PSTL_CDE','PHN_NBR',
'PSTL_5_CDE','PSTL_4_CDE','CNTRY_CDE','FLSH_SLS_IND','MET_CLUSTR_ID','CLMT_LOC_CLUSTR_ID',
'AD_VER_LOC_CLUSTR_ID','PROMO_REGN_CLUSTR_ID','MKT_CLUSTR_ID','GEOG_REGN_CLUSTR_ID',
'STR_CPCTY_CLUSTR_ID','STR_VOL_SP_CLUSTR_ID','TRNSTN_MKT_CLUSTR_ID','TRNSTN_CLMT_CLUSTR_ID',
'MEDN_AGE_NBR','MEDN_HH_INCM_AMT','E_STR_CNT','B_M_STR_CNT','LOC_OPN_IND','MKT_INTNSFTN_CDE',
'ASSOC_HR_NBR','FRST_NM','LAST_NM','MID_INIT_NM', 'A', 'B'
]

asct = pd.read_csv(file_path + file_name, header=True, names = column_schema,
error_bad_lines=False, low_memory=False,
dtype=
{
'EI_ASSOC_ID': np.str,'TM_DIM_KY_DTE': np.str,'FRST_M_NM': np.str,'HIR_DTE': np.str,
'ORIG_HIR_DTE': np.str,'REHIR_DTE': np.str,'TRMNTN_DTE': np.str,'STRT_DTE': np.str,
'CMPSN_RT_AMT': np.float64,'ANNL_RT_AMT': np.float64,'HRLY_RT_AMT': np.str,
'EMP_JOB_CDE': np.str,'EMPT_STAT_CTG_CDE': np.str,'EMP_DEPT_ID': np.str,'EI_LOC_ID': np.str,
'EMPT_TERM_CDE': np.str,'EMP_TYP_CDE': np.str,'SLRY_ADM_PLN_NBR': np.str,'PAY_GRD_CDE': np.str,
'HMN_RSRC_STAT_CDE': np.str,'EMP_STAT_CDE': np.str,'REHIR_IND': np.str,
'EI_REPLT_LOC_ID': np.str,'LOC_ID': np.str,'SLS_INCLN_DTE_TYP_CDE': np.str,
'ORD_STP_DAY_CNT': np.str, 'EXTD_MALL_NM': np.str,'ORD_STRT_DAY_CNT': np.str,
'STR_MGR_NM': np.str,'STR_FMT_CDE': np.str,'TERR_ID': np.str,
'REGN_ID': np.str,'DIST_ID': np.str,'PLNG_CTG_TYP_CDE': np.str,'TOT_STR_SQ_FTG_QTY': np.str,
'TOT_SLG_SQ_FTG_QTY': np.str,'E_BUS_IND': np.str,'DST_IND': np.str,'PRCH_IND': np.str,
'IN_STR_PRCH_IND': np.str,'STR_DIRECTIONS_TXT': np.str,'OPNG_EVNT_YR_NBR': np.str,
'OPNG_EVNT_MN_NBR': np.str,'STR_ACQRD_DTE': np.str,'GRND_OPNG_DTE': np.str,
'FINCL_CLO_DTE': np.str,'RMDL_DTE': np.str,'LOC_NM': np.str,'LOC_10_ABBR_NM': np.str,
'LOC_5_ABBR_NM': np.str,'LOC_TYP_CDE': np.str,'PRNT_LOC_ID': np.str,'TMZN_CDE': np.str,
'LOC_OPN_DTE': np.str,'LOC_OCPNC_DTE': np.str,'LOC_CLO_DTE': np.str,'LNE_1_ADDR': np.str,
'LNE_2_ADDR': np.str,'CTY_NM': np.str,'ST_PROV_CDE': np.str,'PSTL_CDE': np.str,'PHN_NBR': np.str,
'PSTL_5_CDE': np.str,'PSTL_4_CDE': np.str,'CNTRY_CDE': np.str,'FLSH_SLS_IND': np.str,
'MET_CLUSTR_ID': np.str,'CLMT_LOC_CLUSTR_ID': np.str,'AD_VER_LOC_CLUSTR_ID': np.str,
'PROMO_REGN_CLUSTR_ID': np.str,'MKT_CLUSTR_ID': np.str,'GEOG_REGN_CLUSTR_ID': np.str,
'STR_CPCTY_CLUSTR_ID': np.str,'STR_VOL_SP_CLUSTR_ID': np.str,'TRNSTN_MKT_CLUSTR_ID': np.str,
'TRNSTN_CLMT_CLUSTR_ID': np.str,
'MEDN_AGE_NBR': np.float64,'MEDN_HH_INCM_AMT': np.float64,
'E_STR_CNT': np.str,'B_M_STR_CNT': np.str,'LOC_OPN_IND': np.str,'MKT_INTNSFTN_CDE': np.str,
'ASSOC_HR_NBR': np.str,'FRST_NM': np.str,'LAST_NM': np.str,'MID_INIT_NM': np.str,
'A': np.str, 'B': np.str
},
nrows =1000)

# A, B are fake column introduced to 

asct['HIR_DTE'] =pd.to_datetime(asct['HIR_DTE'])
asct['ORIG_HIR_DTE'] =pd.to_datetime(asct['ORIG_HIR_DTE'])
asct['REHIR_DTE'] =pd.to_datetime(asct['REHIR_DTE'])
asct['STRT_DTE'] =pd.to_datetime(asct['STRT_DTE'])
asct['LOC_OPN_DTE'] =pd.to_datetime(asct['LOC_OPN_DTE'])
asct['LOC_OCPNC_DTE'] =pd.to_datetime(asct['LOC_OCPNC_DTE'])

# The data extraction date
asct['TM_DIM_KY_DTE'] =pd.to_datetime(asct['TM_DIM_KY_DTE'])


# Data with missing values marked by ? and 9/9/9999
asct.loc[:, 'TRMNTN_DTE'] = np.where(asct['TRMNTN_DTE']=='?', 
np.nan, asct['TRMNTN_DTE'].apply(lambda x: pd.to_datetime(str(x))))

asct.loc[:, 'LOC_CLO_DTE'] = np.where(asct['LOC_CLO_DTE']=='9/9/9999', 
'01/01/2020', asct['LOC_CLO_DTE'])
    
asct['LOC_CLO_DTE'] =pd.to_datetime(asct['LOC_CLO_DTE'])

asct.loc[:, 'LOC_OPN_DTE'] = np.where(asct['LOC_OPN_DTE']=='EST', 
'01/01/2020', asct['LOC_OPN_DTE'])

asct['LOC_OPN_DTE'] =pd.to_datetime(asct['LOC_OPN_DTE'])

# Limiting the sample to active stores (not closed)
# Approach 1: eliminate the stores closed as of begining of the period
# max_closure_date = datetime.datetime(2016,1,1,0,0,0)
# asct[asct['LOC_CLO_DTE'] > max_closure_date] 

# Approach 2: eliminate the stores closed as of the end of the period
# The stores that are to be closed can be compared to other stores 
#max_closure_date = asct[0,'TM_DIM_KY_DTE']
asct = asct[asct['LOC_CLO_DTE'] > asct['TM_DIM_KY_DTE']] 
asct = asct[asct['LOC_OPN_DTE'] <= asct['TM_DIM_KY_DTE']]
 




# asct['PSTL_CDE'] needs to add 0s to 1111, and split 12121-45840
asct['postal_code'] = asct['PSTL_CDE'].str[:5]
asct['postal_code'] = asct['postal_code'] 

# Remove illegitimate, character symbols
asct = asct[~asct['postal_code'].str.contains('([A-Z]\w{0,})', regex= True)]

asct['postal_code'] = np.where(~asct['postal_code'].str.contains('(\d\d\d\d\d)', regex= True),
'0'+ asct['postal_code'], asct['postal_code'])

# Tenure
# asct.loc[:, 'TRMNTN_DTE'] = np.where(asct['TRMNTN_DTE']=='?', 
# np.nan, asct['TRMNTN_DTE'].apply(lambda x: pd.to_datetime(str(x))))


asct['store_tenure'] = ((asct['TM_DIM_KY_DTE'] - asct['LOC_OPN_DTE'])/np.timedelta64(365*24,'h')).astype(np.int32)

asct['staff_tenure'] = ((np.where(asct['TRMNTN_DTE'].isnull(), asct['TM_DIM_KY_DTE'], asct['TRMNTN_DTE']) 
-asct['HIR_DTE'])/np.timedelta64(365*24,'h')).astype(np.int32) 

# Left = to calculate turnover
asct['terminated_rate'] = np.where(asct['TRMNTN_DTE'].isnull(),0,1)
asct['EMPT_STAT_CTG_CDE'] = asct['EMPT_STAT_CTG_CDE'].map(lambda x: x.strip())

asct['part_time_rate'] = np.where(asct['EMPT_STAT_CTG_CDE']=='P', 1, 0)


# Form an aggregation table

asct_agg = asct.groupby(['LOC_ID', 'LNE_1_ADDR', 'LNE_2_ADDR', 'CTY_NM', 'ST_PROV_CDE','postal_code']).mean().reset_index()


###########################################################################
# Merge county data
# geo['county'] adding 0 to 1111 postal codes
geo['postal_code_adj'] = np.where(geo['postal_code'].str.contains('(\d\d\d\d\d)', regex= True), geo['postal_code'], '0'+ geo['postal_code'])

################################################################
# Data merge
asct_c = asct_agg.merge(geo, left_on='postal_code', right_on ='postal_code_adj', how = 'left')

# To merge with the demographics data county and state should be combined
asct_c["county, state"] = asct_c["county"].map(str) +', '+ asct_c["state"]

# Data source 1.6
# Calendar of the marketing events (obtained by Garima)
# Note on adjusrtment to the original data source KohlsMarketingCalendar.csv:
# 1) "EVENT_TYPE" = "Non Event" is substituted by "Non_Event"
# 2) 1st line of zeros is added, since the 1st line after the header is not read

file_name = 'KohlsMarketingCalendar_adj.csv' 
#file_name = 'KohlsMarketingCalendar_adj.txt' 

#with open(file_path + file_name) as f:
#    for line in islice(f, 1):
#       print line

column_schema = [
'MKTG_CAL_DTE',
'EVENT_TYPE',
'VALENTINES_DAY',
'PRESIDENTS_DAY',
'EASTER',
'MOTHERS_DAY',
'MEMORIAL_DAY',
'FATHERS_DAY',
'INDEPENDENCE_DAY',
'LABOR_DAY',
'COLUMBUS_DAY',
'HALLOWEEN',
'VETERANS_DAY',
'THANKSGIVING',
'CHRISTMAS',
'NEW_YEARS',
'MLK_DAY',
'SUPER_BOWL',
'HANUKKAH',
'TAB_SUPPORT_FLAG',
'DIRECTMAIL_SUPPORT_FLAG',
'RADIO_SUPPORT_FLAG',
'TV_SUPPORT_FLAG',
'FULLFILE_EMAIL_SUPPORT_FLAG',
'ONLINE_SUPPORT_FLAG',
'KOHLSCASHEARN_5_25',
'KOHLSCASHREDEEM_5_25',
'KOHLSCASHEARN_10_50',
'KOHLSCASHREDEEM_10_50',
'KOHLSCASHEARN_15_50',
'KOHLSCASHREDEEM_15_50',
'KOHLSCASHEARN',
'KOHLSCASHREDEEM',
'LTO_15',
'LTO_20',
'LTO_25',
'LTO_30',
'LTO',
'FRIENDSFAMILY_MAIL_15',
'FRIENDSFAMILY_MAIL_20',
'FRIENDSFAMILY_MAIL_1520',
'FRIENDSFAMILY_MAIL_2025',	
'FRIENDSFAMILY_GPO_2025',
'FRIENDSFAMILY_GPO_20',	
'FRIENDSFAMILY_GPO_25',
'FRIENDSFAMILY',
'ASSOCIATE_SHOP',
'ASSOCIATE_MERCH_OFFER',
'CREDIT_EVENT',
'PAD_MVC_BURG',
'PAD_MVC',
'CASP_10_25_KC',
'CASP_10_30_KC',
'CASP_1030_1550_KC',
'CASP_15_KC',
'CASP_20_KC',	
'CASP_25_KC',
'BMSM_1520_KC',	
'BMSM_2025_KC',
'KC_SHOPPING_PASS',
'CASP_10_25_BC',
'CASP_10_30_BC',
'CASP_1030_1550_BC',
'CASP_15_BC',
'CASP_20_BC',
'CASP_25_BC',
'BMSM_1520_BC',
'BMSM_2025_BC',
'BC_SHOPPING_PASS',
'GPO_10_25',	
'GPO_10_30',
'GPO_10_50',
'GPO_TIER_10_30_15_50',
'GPO_CASP_15',	
'GPO_CASP_20',
'GPO_KC_25',
'GPO_BMSM_1520',
'GPO_BMSM_1525',
'GPO',
'LOYALTY_TRIPLE_POINTS',
'EMAIL_MYSTERY_5_10_15_DOLLAR',
'EMAIL_MYSTERY_20_30_40_PERCENT',
'EMAIL_MYSTERY_OFFER',
'ATHLETICSHOE_10_40_GPO',
'BABY_10_25_DM',
'BABY_10_30_GPO',
'BABY_20_GPO',
'COMFORTATHLETICSHOE_10_40_GPO',
'COMFORTSHOE_10_30_DM',
'COMFORTSHOE_10_40_DM',
'COMFORTSHOE_10_40_GPO',
'COMFORTSHOE_10_50_GPO',
'CUDDLDUDS_10_40_GPO',
'FATHERSDAY_10_30_GPO',
'FATHERSDAY_10_40_GPO',
'FATHERSDAYJEWELRY_10_50_GPO',
'HANES_10_40_DM',
'HOME_10_50_DM',
'HOME_10_50_GPO',
'HOME_15_50_GPO',
'HOME_15PT_GPO',
'INTIMATES_10_20_DM',
'INTIMATES_10_25_DM',
'INTIMATES_10_30_DM',
'INTIMATES_10_40_GPO',
'JEWELRY_10_50_DM',
'JEWELRY_10_50_GPO',
'JEWELRY_15_75_GPO',
'JEWELRY_15PT_GPO',
'JEWELRY_20PT_GPO',
'JEWELRY_25PT_GPO',
'JEWELRYWATCHES_10_40_GPO',
'JEWELRYWATCHES_10_GPO',
'JEWELRYWATCHES_20PT_GPO',
'JUNIORS_10_20_DM',
'JUNIORS_10_30_GPO',	
'JUNIORS_10_40_GPO',
'KIDS_10_20_DM',	
'KIDS_10_25_DM',
'KIDS_10_30_GPO',
'KIDS_10_40_GPO',
'KIDS_20PT_GPO',
'KIDSSHOES_10_30_GPO',
'LCBEDDING_10_50_GPO',
'LUGGAGE_50_200_DM',	
'LUGGAGE_50_200_GPO',
'MENS_10_25_GPO',
'MENS_10_30_DM',
'MENS_10_30_GPO',
'MENS_10_40_GPO',
'MENS_10_50_GPO',
'MENS_20PT_GPO',
'MENSBASICS_10_40_GPO',
'MENSBIGTALL_10_20_DM',
'MENSBIGTALL_10_25_DM',
'MENSBIGTALL_10_30_DM',	
'MENSBIGTALL_10_30_GPO',
'MENSDRESS_10_40_GPO',
'MISSES_10_40_GPO',	
'MISSES_10_50_GPO',
'MOTHERSDAY_10_50_GPO',
'NATIONALBRANDS_10_30_DM',
'NFL_10_50_GPO',
'PATIO_50_200_GPO',
'ROCKREPUBLIC_10_GPO',
'SHOEHANDBAG_10_25_DM',
'SHOEHANDBAG_10_40_DM',
'SHOEHANDBAG_10_50_DM',
'SHOEHANDBAG_10_50_GPO',
'SHOES_10_40_GPO',
'SHOES_10_50_GPO',
'TOYS_10_50_GPO',
'YM_10_20_DM',
'YM_10_40_GPO',	
'MERCH_GPO',
'TAB_KOHLS_AT_BOTTOM',
'TAB_KOHLS_HIDDEN',
'TAB_WHITE_BACKGROUND',
'TAB_REMOVED_CLEAR_BB_DB_MESSAGE',
'ONLINE_ONLY',	
'AFFILIATE_ONLY',
]	


mktcr = pd.read_csv(file_path + file_name, header=True, names = column_schema,
error_bad_lines=False, low_memory=False,
nrows =20)



mktcr['MKTG_CAL_Date'] =pd.to_datetime(mktcr['MKTG_CAL_DTE'])


print mktcr.dtypes
mktcr.describe()

###############################################################################
# 1.7	"Geo coordinates/postal address of Kohl's competitors:
file_path = 'F:/aData/LI/' 

# Importing 
file_name = 'competitors_list_adj.csv'

# The records associated with data on
# Mall del Norte, 5300 San Dario Ave, Laredo, TX 78041, USA
# were reported in km instead of miles and had to be converted:
# 1 mile = 1.6 km

#Reading the 
with open(file_path + file_name) as f:
   for line in islice(f, 2):
      print line

cmtrs_df = pd.read_csv(file_path + file_name, header=False, 
                      error_bad_lines=False, low_memory=False)
                            
cmtrs_df.head()
cmtrs_df["postal_code"] = cmtrs_df["address"].str.extract('(\d\d\d\d\d)')


max_drive_time = 24*60 # one day to drive 
cmtrs_df["Macys_drive_time"] = np.nan
cmtrs_df["Macys_drive_time"] = np.where(cmtrs_df["Macy's_drive_time_from_kohls"].str.contains('day'), max_drive_time,)

cmtrs_df["Macys_drive_time"] = np.where(cmtrs_df["Macy's_distance_from_kohls"].str.contains('NAN'), "2,900 mi",cmtrs_df["Macy's_distance_from_kohls




max_distance =2900
cmtrs_df["Macy's_distance_from_kohls"] = np.where(cmtrs_df["Macy's_distance_from_kohls"].str.contains('ft'), "0.1 mi",cmtrs_df["Macy's_distance_from_kohls"])
cmtrs_df["Macy's_distance_from_kohls"] = np.where(cmtrs_df["Macy's_distance_from_kohls"].str.contains('NAN'), "2,900 mi",cmtrs_df["Macy's_distance_from_kohls"])

cmtrs_df["Macy_distance"] = max_distance # max

# Observations' cases
cmtrs_df["Macy_distance"] = np.where(cmtrs_df["Macy's_distance_from_kohls"].str.contains('(\d.\d)'),cmtrs_df["Macy's_distance_from_kohls"].str.extract('(\d.\d)'),cmtrs_df["Macy_distance"])
cmtrs_df["Macy_distance"] = np.where(cmtrs_df["Macy's_distance_from_kohls"].str.contains('(\d\d.\d)'),cmtrs_df["Macy's_distance_from_kohls"].str.extract('(\d\d.\d)'),cmtrs_df["Macy_distance"])
cmtrs_df["Macy_distance"] = np.where(cmtrs_df["Macy's_distance_from_kohls"].str.contains('(\d\d\d)'),cmtrs_df["Macy's_distance_from_kohls"].str.extract('(\d\d\d)'),cmtrs_df["Macy_distance"])
cmtrs_df["Macy_distance"] = np.where(cmtrs_df["Macy's_distance_from_kohls"].str.contains('(\d\,\d\d\d)'),cmtrs_df["Macy's_distance_from_kohls"].str.extract('(\d\d\d)'),cmtrs_df["Macy_distance"])








##################################
cmtrs_df["Target_distance_0"] = np.where(cmtrs_df["Target_distance_from_kohls"].str.contains('([A-Z]\w{0,})')=='ft', '0.0 mi',cmtrs_df["Target_distance_from_kohls"])



cmtrs_df["Target_distance"] = cmtrs_df["Target_distance_from_kohls"].str.extract('(\d\d\.\d)').astype(float)
cmtrs_df["Target_distance"] = np.where(cmtrs_df["Target_distance"]==np.nan,cmtrs_df["Target_distance_from_kohls"].str.extract('(\d.\d)').astype(float),cmtrs_df["Target_distance"])  )
                            
###############################################################################


# 1.8. Demographics data merged by county
file_path = 'F:/aData/LI/Demographics/' 

# Importing 
file_name = 'US_county_income_by_poverty_level_2015.csv'

#Reading the 
with open(file_path + file_name) as f:
   for line in islice(f, 2):
      print line

county_poverty= pd.read_csv(file_path + file_name, header=True, 
                            error_bad_lines=False, low_memory=False)

poverty = pd.DataFrame(county_poverty['Geography'])

#poverty['Geography'] = county_poverty['Geography']
poverty['below_poverty_under_6_yrs'] = (county_poverty['Estimate; Under 6 years: - Under .50']+
county_poverty['Estimate; Under 6 years: - .50 to .74'] +
county_poverty ['Estimate; Under 6 years: - .75 to .99'])/county_poverty ['Estimate; Under 6 years:']


poverty['below_poverty_6_11__yrs'] = (county_poverty['Estimate; 6 to 11 years: - Under .50']+
county_poverty['Estimate; 6 to 11 years: - .50 to .74'] +
county_poverty ['Estimate; 6 to 11 years: - .75 to .99'])/county_poverty ['Estimate; 6 to 11 years:']

poverty['below_poverty_12_17_yrs'] = (county_poverty['Estimate; 12 to 17 years: - Under .50']+
county_poverty['Estimate; 12 to 17 years: - .50 to .74'] +
county_poverty ['Estimate; 12 to 17 years: - .75 to .99'])/county_poverty ['Estimate; 12 to 17 years:']


poverty['below_poverty_18_24_yrs'] = (county_poverty['Estimate; 18 to 24 years: - Under .50']+
county_poverty['Estimate; 18 to 24 years: - .50 to .74'] +
county_poverty ['Estimate; 18 to 24 years: - .75 to .99'])/county_poverty ['Estimate; 18 to 24 years:']

poverty['below_poverty_25_34_yrs'] = (county_poverty['Estimate; 25 to 34 years: - Under .50']+
county_poverty['Estimate; 25 to 34 years: - .50 to .74'] +
county_poverty ['Estimate; 25 to 34 years: - .75 to .99'])/county_poverty ['Estimate; 25 to 34 years:']


poverty['below_poverty_35_44_yrs'] = (county_poverty['Estimate; 35 to 44 years: - Under .50']+
county_poverty['Estimate; 35 to 44 years: - .50 to .74'] +
county_poverty ['Estimate; 35 to 44 years: - .75 to .99'])/county_poverty ['Estimate; 35 to 44 years:']

poverty['below_poverty_45_54_yrs'] = (county_poverty['Estimate; 45 to 54 years: - Under .50']+
county_poverty['Estimate; 45 to 54 years: - .50 to .74'] +
county_poverty ['Estimate; 45 to 54 years: - .75 to .99'])/county_poverty ['Estimate; 45 to 54 years:']

poverty['below_poverty_55_64_yrs'] = (county_poverty['Estimate; 55 to 64 years: - Under .50']+
county_poverty['Estimate; 55 to 64 years: - .50 to .74'] +
county_poverty ['Estimate; 55 to 64 years: - .75 to .99'])/county_poverty ['Estimate; 55 to 64 years:']

poverty['below_poverty_65_74_yrs'] = (county_poverty['Estimate; 65 to 74 years: - Under .50']+
county_poverty['Estimate; 65 to 74 years: - .50 to .74'] +
county_poverty ['Estimate; 65 to 74 years: - .75 to .99'])/county_poverty ['Estimate; 65 to 74 years:']

poverty['below_poverty_over_74_yrs'] = (county_poverty['Estimate; 75 years and over: - Under .50']+
county_poverty['Estimate; 75 years and over: - .50 to .74'] +
county_poverty ['Estimate; 75 years and over: - .75 to .99'])/county_poverty ['Estimate; 75 years and over:']

print poverty.head()

########################################
# Importing Race/Gender/Age
file_path = 'F:/aData/LI/Demographics/' 
file_name = 'US_county_aggeragate_demographics_2015.csv'

#Reading the 
#with open(file_path + file_name) as f:
#   for line in islice(f, 2):
#      print line

dmgphcs= pd.read_csv(file_path + file_name, header=True, 
                            error_bad_lines=False, low_memory=False)

racex = pd.DataFrame(dmgphcs['Geography'])


racex['population'] = dmgphcs['Estimate; SEX AND AGE - Total population']
racex['female_pcnt'] = dmgphcs['Percent; SEX AND AGE - Total population - Female']
racex['median_age'] = dmgphcs['Estimate; SEX AND AGE - Median age (years)']
racex['over_21_yrs_pcnt'] = dmgphcs['Percent; SEX AND AGE - 21 years and over']
racex['over_65_yrs_pcnt'] = dmgphcs['Percent; SEX AND AGE - 65 years and over']

racex['female_over_18_yrs_pcnt'] = dmgphcs['Percent; SEX AND AGE - 18 years and over - Female']
racex['female_over_65_yrs_pcnt'] = dmgphcs['Percent; SEX AND AGE - 65 years and over - Female']

racex['whites_pcnt'] = dmgphcs['Percent; RACE - Race alone or in combination with one or more other races - Total population - White']
racex['blacks_pcnt'] = dmgphcs['Percent; RACE - Race alone or in combination with one or more other races - Total population - Black or African American']
racex['natives_pcnt'] = dmgphcs[['Percent; RACE - Race alone or in combination with one or more other races - Total population - American Indian and Alaska Native', 'Percent; RACE - Race alone or in combination with one or more other races - Total population - Native Hawaiian and Other Pacific Islander']].sum(axis=1)
racex['asians_pcnt'] = dmgphcs['Percent; RACE - Race alone or in combination with one or more other races - Total population - Asian']
racex['indians_pcnt'] = dmgphcs['Percent; RACE - One race - Asian - Asian Indian']
racex['chinese_pcnt'] = dmgphcs['Percent; RACE - One race - Asian - Chinese']
racex['filipino_pcnt'] = dmgphcs['Percent; RACE - One race - Asian - Filipino']

racex['latinos_pcnt'] = dmgphcs['Percent; HISPANIC OR LATINO AND RACE - Total population - Hispanic or Latino (of any race)']
racex['people_per_house_unit'] = dmgphcs['Estimate; SEX AND AGE - Total population']/dmgphcs['Estimate; HISPANIC OR LATINO AND RACE - Total housing units']

# Importing Education ######################################################
file_path = 'F:/aData/LI/Demographics/' 
file_name = 'US_county_education_2015.csv'

#Reading the 
#with open(file_path + file_name) as f:
#   for line in islice(f, 2):
#      print line

edu_df= pd.read_csv(file_path + file_name, header=True, 
                            error_bad_lines=False, low_memory=False)

edu = pd.DataFrame(edu_df['Geography'])

edu['high_school_diploma_over_24_yrs_pcnt'] = edu_df['Percent; Estimate; Population 25 years and over - High school graduate (includes equivalency)']
edu['assct_other_degree_over_24_yrs_pcnt'] =edu_df[["Percent; Estimate; Population 25 years and over - Some college, no degree","Percent; Estimate; Population 25 years and over - Associate's degree"]].sum(axis=1)
edu['BA_degree_over_24_yrs_pcnt'] = edu_df["Percent; Estimate; Population 25 years and over - Bachelor's degree"]
edu['grad_degree_over_24_yrs_pcnt'] = edu_df['Percent; Estimate; Population 25 years and over - Graduate or professional degree']

edu['females_high_school_diploma_over_24_yrs_pcnt'] =  edu_df['Percent Females; Estimate; Population 25 years and over - High school graduate (includes equivalency)']
edu['females_assct_other_degree_over_24_yrs_pcnt'] = edu_df[["Percent Females; Estimate; Population 25 years and over - Some college, no degree","Percent Females; Estimate; Population 25 years and over - Associate's degree"]].sum(axis=1)
edu['females_BA_degree_over_24_yrs_pcnt'] = edu_df["Percent Females; Estimate; Population 25 years and over - Bachelor's degree"]
edu['females_grad_degree_over_24_yrs_pcnt'] = edu_df['Percent Females; Estimate; Population 25 years and over - Graduate or professional degree']

edu['median_earnings_over_24_yrs'] =edu_df['Total; Estimate; MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings']
edu['females_median_earnings_over_24_yrs'] = edu_df['Females; Estimate; MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings']

edu['females_high_school_median_earnings_over_24_yrs'] = edu_df['Females; Estimate; MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings - High school graduate (includes equivalency)']
edu['females_assct_other_degree_median_earnings_over_24_yrs'] = edu_df["Females; Estimate; MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings - Some college or associate's degree"]
edu['females_BA_degree_median_earnings_over_24_yrs'] = edu_df["Females; Estimate; MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings - Bachelor's degree"]
edu['females_grad_degree_median_earnings_over_24_yrs'] = edu_df["Females; Estimate; MEDIAN EARNINGS IN THE PAST 12 MONTHS (IN 2015 INFLATION-ADJUSTED DOLLARS) - Population 25 years and over with earnings - Graduate or professional degree"]


# Demographics Data merge
poverty_racex_c = poverty.merge(racex, left_on='Geography', right_on ='Geography', how = 'left')
demogr_c = poverty_racex_c.merge(edu, left_on='Geography', right_on ='Geography', how = 'left')




#### Unresolved issue #######################################################

# 1.4. Extract SKU Categories from gold_at_sku):
# 1. Brand (e.g., Nike)
# 2. Department (e.g., shoes);
# 3. Gender = {male, female};
# 4. Color (e.g, white = ‘1’, grey = ‘3, etc.)

 
### Tiroshan's code ###
 
import pandas as pd
import glob

# Load data from a csv file
#file_path = 'resources/gold_atg_sku_dump/'

file_path = 'F:/aData/LI/'
file_name =  '000000_0'

gold_atg_sku_schema = [
    'sku_id',
    'parent_product_id',
    'childsku_seq_nbr',
    'mdse_type_cde',
    'nrf_color_family_desc',
    'sku_status_cde',
    'sku_color',
    'sku_size_desc',
    'size_sort_cde',
    'special_sku_type_cde',
    'sku_list_price_type_cde',
    'dept_nm',
    'dept_nbr',
    'sub_class_nbr',
    'class_nm',
    'sku_cost',
    'sku_supressed_price',
    'sku_pricing_status_cde',
    'sku_pricing_status_value',
    'sku_size_range',
    'sku_list_price_type_value',
    'sku_status_value',
    'sku_primary_size',
    'sku_secondary_size',
    'sku_active_flg',
    'nrf_color_cde',
    'nrf_color_desc',
    'nrf_color_family_cde',
    'nrf_size_cde',
    'nrf_size_desc',
    'direct_ship_sku_flg',
    'sku_surcharge_fee',
    'sku_gift_wrap_flg',
    'pack_slip_desc',
    'ship_pack_cde',
    'ship_service_cde',
    'sku_replenish_ind',
    'sku_return_info',
    'ship_pack_cde_desc',
    'web_eligible_flg',
    'ship_charge_eligible_flg',
    'ship_restriction_cde',
    'ship_restriction_value',
    'ship_service_cde_desc',
    'channel_avail_cde',
    'channel_avail_desc',
    'vendor_color_desc',
    'ship_alone_flg',
    'offer_eligible_flg',
    'special_sku_type_value',
    'mdse_type_value',
    'gma_nbr',
    'gma_desc',
    'style_id',
    'style_desc',
    'cust_choice_id',
    'cust_choice_desc',
    'division_nbr',
    'division_nm',
    'class_nbr',
    'sub_class_nm',
    'tax_cde',
    'tax_eligible_flg',
    'hazardous_flg',
    'fraud_risk',
    'emp_dis_eligible_flg',
    'max_purchase_qty',
    'sku_min_ship_lead_days',
    'sku_max_ship_lead_days',
    'seller_cde',
    'seller_value',
    'dummy_sku_id',
    'product_version',
    'product_creation_dt',
    'product_start_dt',
    'product_end_dt',
    'product_web_display_nm',
    'product_desc',
    'product_long_desc',
    'parent_cat_id',
    'product_type',
    'product_admin_display',
    'product_nonreturnable',
    'product_brand',
    'product_disallow_recommend_flg',
    'product_manufacturer',
    'product_brand_type_cde',
    'product_type_cde',
    'product_sale_assort_price_nm',
    'product_alt_disp_nm',
    'product_size_chart_guide_text',
    'product_meta_title',
    'product_meta_desc',
    'product_featured_sku',
    'product_primary_type_value',
    'product_type_value',
    'product_sub_type_value',
    'product_feature_cde',
    'product_market_place_flg',
    'product_short_disp_nm',
    'product_ignore_inventory',
    'product_fixed_prod_badges',
    'product_meta_keywords',
    'product_status_cde',
    'product_status_value',
    'product_sub_type_cde',
    'product_warranty_text',
    'product_guarantee_text',
    'product_guide_link',
    'product_size_chart_guide_link',
    'product_primary_size_type',
    'product_secondary_size_type',
    'product_brand_id',
    'product_brand_type_value',
    'product_rebate_desc',
    'product_guide_text',
    'product_new_flg',
    'product_sale_event_id',
    'product_min_ship_lead_days',
    'product_max_ship_lead_days',
    'product_video_url',
    'product_primary_type_cde',
    'product_workflow_status',
    'product_go_live_dt',
    'product_pack_slip_desc',
    'product_sale_assort_future_dt',
    'product_sale_assort_percent_off',
    'product_sale_assort_price_point',
    'product_mvt_active_flg',
    'product_vendor_nm',
    'activity',
    'age_appropriate',
    'age_level_appr_range',
    'beverage_type',
    'birthstone',
    'body_support_rating',
    'body_type',
    'boot_shaft_height',
    'by_service',
    'capacity',
    'carat',
    'care_instructions',
    'child_age_range',
    'closure',
    'consumer_material',
    'consumer_nrf_color',
    'consumer_outsole_material',
    'consumer_persona_theme',
    'consumer_silhouette',
    'consumer_size_description',
    'device_platform_code',
    'diamond_clarity',
    'diamond_clarity_grade',
    'diamond_color_grade',
    'display_resolution',
    'display_size_number',
    'drawer_count_quantity',
    'ecom_trend',
    'electronic_storage_capacity_code',
    'feature',
    'fill',
    'finish_wash',
    'firmness_rating',
    'fit',
    'formality_occasion',
    'gemstone',
    'gemstone_carat',
    'gemstone_cut',
    'gemstone_quantity',
    'gender',
    'handle_style',
    'heel_height',
    'leg_opening',
    'length',
    'length_characteristic',
    'light_bulb_count_quantity',
    'material_type',
    'material_weight_rating',
    'mattress_thickness',
    'media_rating',
    'megapixel_quantity',
    'metal_color',
    'neckline',
    'nrf_color_family_description',
    'nrf_size_description',
    'number_of_openings',
    'number_of_set_pieces',
    'pant_front',
    'pattern',
    'persona_category',
    'persona_group',
    'persona_subject',
    'persona_theme',
    'pet_type',
    'quality_rating',
    'recommended_use',
    'rise',
    'room',
    'scent',
    'shelf_quantity',
    'shoe_width',
    'silhouette',
    'sleeve_length',
    'sports_league',
    'sports_player_name',
    'sports_team',
    'supporting_nrf_color_family',
    'thread_count',
    'tier_count_quantity',
    'type',
    'upf_rating',
    'watch_clock_movement',
    'wicks_quantity',
    'window_treatment_top_construction'
]

df = pd.read_csv(file_path + file_name, sep=',', skipinitialspace=True, header=None, error_bad_lines=False,
                 names=gold_atg_sku_schema, nrows =10)

print df[['sku_id', 'parent_product_id', 'sku_color', 'nrf_color_family_desc']]


























##############################################################################
################# Eugene's code ##############################################

file_name = "gold_atg_sku_dump_000000_2.txt"
 
#Reading the 
with open(file_path + file_name) as f:
   for line in islice(f, 1):
      print line

column_schema = [
'activity',
'age_appropriate',
'age_level_appr_range',
'beverage_type',	
'birthstone' ,	
'body_support_rating',
'body_type',
'boot_shaft_height',	
'by_service',
'capacity',	
'carat',	
'care_instructions',
'channel_avail_cde',	
'channel_avail_desc',
'child_age_range',
'childsku_seq_nbr',	
'class_nbr',	
'class_nm',	
'closure',
'consumer_material',	
'consumer_nrf_color',	
'consumer_outsole_material',	
'consumer_persona_theme',		
'consumer_silhouette',	
'consumer_size_description',	
'cust_choice_desc',		
'cust_choice_id',		
'dept_nbr',		
'dept_nm',	
'device_platform_code',	
'diamond_clarity',	
'diamond_clarity_grade',	
'diamond_color_grade',	
'direct_ship_sku_flg',
'display_resolution',	
'display_size_number',	
'division_nbr',	
'division_nm'	,	
'drawer_count_quantity',	
'dummy_sku_id',	
'ecom_trend',
'electronic_storage_capacity_code',	
'emp_dis_eligible_flg',	
'feature',
'fill',
'finish_wash',	
'firmness_rating',	
'fit',	
'formality_occasion',
'fraud_risk',	
'gemstone',	
'gemstone_carat',	
'gemstone_cut',	
'gemstone_quantity',	
'gender',	
'gma_desc',		
'gma_nbr',
'handle_style',
'hazardous_flg',	
'heel_height'	,
'leg_opening',	
'length',	
'length_characteristic',
'light_bulb_count_quantity',	
'material_type',
'material_weight_rating',	
'mattress_thickness',		
'max_purchase_qty',
'mdse_type_cde',		
'mdse_type_value',		
'media_rating',
'megapixel_quantity',		
'metal_color'	,
'neckline',
'nrf_color_cde',		
'nrf_color_desc',		
'nrf_color_family_cde'	,		
'nrf_color_family_desc',		
'nrf_color_family_description',	
'nrf_size_cde',		
'nrf_size_desc',	
'nrf_size_description'	,		
'number_of_openings',		
'number_of_set_pieces',		
'offer_eligible_flg',		
'pack_slip_desc',		
'pant_front',		
'parent_cat_id',		
'parent_product_id',		
'pattern',		
'persona_category',
'persona_group',
'persona_subject',
'persona_theme',	
'pet_type',
'product_admin_display',		
'product_alt_disp_nm',		
'product_brand',		
'product_brand_id',		
'product_brand_type_cde',		
'product_brand_type_value',		
'product_creation_dt',		
'product_desc',	
'product_disallow_recommend_flg',
'product_end_dt',		
'product_feature_cde',		
'product_featured_sku',		
'product_fixed_prod_badges',	
'product_go_live_dt',		
'product_guarantee_text',		
'product_guide_link',		
'product_guide_text',		
'product_ignore_inventory'	,	
'product_long_desc',		
'product_manufacturer'	,		
'product_market_place_flg',	
'product_max_ship_lead_days',	
'product_meta_desc',		
'product_meta_keywords',		
'product_meta_title',		
'product_min_ship_lead_days',
'product_mvt_active_flg',	
'product_new_flg',	
'product_nonreturnable',
'product_pack_slip_desc',		
'product_primary_size_type',		
'product_primary_type_cde',		
'product_primary_type_value',		
'product_rebate_desc',		
'product_sale_assort_future_dt'	,		
'product_sale_assort_percent_off',	
'product_sale_assort_price_nm',		
'product_sale_assort_price_point',	
'product_sale_event_id	',		
'product_secondary_size_type',		
'product_short_disp_nm',		
'product_size_chart_guide_link'	,		
'product_size_chart_guide_text'	,		
'product_start_dt',		
'product_status_cde',		
'product_status_value'	,		
'product_sub_type_cde'	,		
'product_sub_type_value',		
'product_type',
'product_type_cde',		
'product_type_value',		
'product_vendor_nm',		
'product_version',	
'product_video_url',		
'product_warranty_text',		
'product_web_display_nm',		
'product_workflow_status',
'quality_rating',
'recommended_use',		
'rise',		
'room',
'scent',		
'seller_cde',		
'seller_value',		
'shelf_quantity',		
'ship_alone_flg',
'ship_charge_eligible_flg',
'ship_pack_cde',		
'ship_pack_cde_desc',		
'ship_restriction_cde'	,	
'ship_restriction_value',		
'ship_service_cde',		
'ship_service_cde_desc',	
'shoe_width',	
'silhouette',	
'size_sort_cde',	
'sku_active_flg',	
'sku_color',	
'sku_cost',
'sku_gift_wrap_flg',
'sku_id',	
'sku_list_price_type_cde',		
'sku_list_price_type_value',		
'sku_max_ship_lead_days',
'sku_min_ship_lead_days',	
'sku_pricing_status_cde',		
'sku_pricing_status_value',		
'sku_primary_size',		
'sku_replenish_ind',
'sku_return_info',		
'sku_secondary_size',		
'sku_size_desc',	
'sku_size_range',	
'sku_status_cde',	
'sku_status_value',	
'sku_supressed_price',	
'sku_surcharge_fee',
'sleeve_length',		
'special_sku_type_cde',		
'special_sku_type_value',	
'sports_league',	
'sports_player_name',
'sports_team'	,
'style_desc',		
'style_id',		
'sub_class_nbr',		
'sub_class_nm',	
'supporting_nrf_color_family',
'tax_cde',	
'tax_eligible_flg',
'thread_count',	
'tier_count_quantity',	
'type',		
'upf_rating',	
'vendor_color_desc',	
'watch_clock_movement'	,		
'web_eligible_flg',
'wicks_quantity',	
'window_treatment_top_construction'
]

gold_atg_sku = pd.read_csv(file_path + file_name, header=False, names = column_schema,
error_bad_lines=False, low_memory=False,
dtype=
{
'activity' : np.str,
'age_appropriate' : np.str,
'age_level_appr_range' : np.str,
'beverage_type'	 : np.str,
'birthstone'  : np.str,	
'body_support_rating' : np.str,
'body_type' : np.str,
'boot_shaft_height' : np.str,	
'by_service' : np.str,
'capacity'	: np.str,	
'carat' : np.str,	
'care_instructions'	: np.str,
'channel_avail_cde'	: np.str,	
'channel_avail_desc' : np.str,
'child_age_range'  : np.str,
'childsku_seq_nbr': np.float,	
'class_nbr' :	np.str,	
'class_nm'	: np.str,	
'closure'	 : np.str,
'consumer_material'	: np.str,	
'consumer_nrf_color' : np.str,	
'consumer_outsole_material' : np.str,	
'consumer_persona_theme'	: np.str,		
'consumer_silhouette'	: np.str,	
'consumer_size_description': np.str,	
'cust_choice_desc': np.str,		
'cust_choice_id': np.str,		
'dept_nbr'	: np.str,		
'dept_nm'	: np.str,	
'device_platform_code'	: np.str,	
'diamond_clarity'	: np.str,	
'diamond_clarity_grade'	: np.str,
'diamond_color_grade'	 : np.str,
'direct_ship_sku_flg' :	np.float,
'display_resolution'	: np.str,	
'display_size_number'	: np.str,	
'division_nbr'	: np.str,	
'division_nm'	: np.str,	
'drawer_count_quantity'	: np.str,	
'dummy_sku_id'	: np.str,	
'ecom_trend'	 : np.str,
'electronic_storage_capacity_code'	: np.str,	
'emp_dis_eligible_flg'	: np.float,	
'feature'	: np.str,
'fill'	 : np.str,
'finish_wash'  : np.str,	
'firmness_rating'	: np.str,	
'fit'	: np.str,	
'formality_occasion'  : np.str,
'fraud_risk' : np.str,	
'gemstone'	 : np.str,	
'gemstone_carat' : np.str,	
'gemstone_cut'	: np.str,	
'gemstone_quantity': np.str,	
'gender'	 : np.str,	
'gma_desc'	: np.str,		
'gma_nbr'	: np.str,
'handle_style'	 : np.str,
'hazardous_flg'	: np.float,	
'heel_height'	: np.str,
'leg_opening'	: np.str,	
'length'	: np.str,	
'length_characteristic'	: np.str,
'light_bulb_count_quantity'	: np.str,	
'material_type'	 : np.str,
'material_weight_rating'	: np.str,	
'mattress_thickness'	: np.str,		
'max_purchase_qty'	: np.float,
'mdse_type_cde'	: np.str,		
'mdse_type_value'	: np.str,		
'media_rating'	 : np.str,
'megapixel_quantity'	: np.str,		
'metal_color'	 : np.str,
'neckline'	 : np.str,
'nrf_color_cde'	: np.str,		
'nrf_color_desc'	: np.str,		
'nrf_color_family_cde'	: np.str,		
'nrf_color_family_desc'	: np.str,		
'nrf_color_family_description'	: np.str,	
'nrf_size_cde'	: np.str,		
'nrf_size_desc'	: np.str,	
'nrf_size_description'	: np.str,		
'number_of_openings'	: np.str,		
'number_of_set_pieces'	: np.str,		
'offer_eligible_flg'	: np.str,		
'pack_slip_desc'	: np.str,		
'pant_front'	: np.str,		
'parent_cat_id'	: np.str,		
'parent_product_id'	: np.str,		
'pattern'	: np.str,		
'persona_category'	 : np.str,
'persona_group'	 : np.str,
'persona_subject' : np.str,
'persona_theme'	 : np.str,	
'pet_type'	 : np.str,
'product_admin_display'	: np.str,		
'product_alt_disp_nm'	: np.str,		
'product_brand'	: np.str,		
'product_brand_id'	: np.str,		
'product_brand_type_cde'	: np.str,		
'product_brand_type_value'	: np.str,		
'product_creation_dt'	: np.str,		
'product_desc'	: np.str,	
'product_disallow_recommend_flg'	: np.float,
'product_end_dt'	: np.str,		
'product_feature_cde'	: np.str,		
'product_featured_sku'	: np.str,		
'product_fixed_prod_badges'	: np.str,	
'product_go_live_dt'	: np.str,		
'product_guarantee_text'	: np.str,		
'product_guide_link'	: np.str,		
'product_guide_text'	: np.str,		
'product_ignore_inventory'	: np.float,	
'product_long_desc'	: np.str,		
'product_manufacturer'	: np.str,		
'product_market_place_flg'	: np.float,	
'product_max_ship_lead_days'	: np.float,	
'product_meta_desc'	: np.str,		
'product_meta_keywords'	: np.str,		
'product_meta_title'	: np.str,		
'product_min_ship_lead_days'	: np.float,
'product_mvt_active_flg'	: np.float,	
'product_new_flg'	: np.float,	
'product_nonreturnable'	: np.float,
'product_pack_slip_desc'	: np.str,		
'product_primary_size_type'	: np.str,		
'product_primary_type_cde'	: np.str,		
'product_primary_type_value'	: np.str,		
'product_rebate_desc'	: np.str,		
'product_sale_assort_future_dt'	: np.str,		
'product_sale_assort_percent_off' 	: np.float,	
'product_sale_assort_price_nm'	: np.str,		
'product_sale_assort_price_point' 	: np.float,	
'product_sale_event_id	': np.str,		
'product_secondary_size_type'	: np.str,		
'product_short_disp_nm'	: np.str,		
'product_size_chart_guide_link'	: np.str,		
'product_size_chart_guide_text'	: np.str,		
'product_start_dt'	: np.str,		
'product_status_cde'	: np.str,		
'product_status_value'	: np.str,		
'product_sub_type_cde'	: np.str,		
'product_sub_type_value'	: np.str,		
'product_type'	: np.float,
'product_type_cde'	: np.str,		
'product_type_value'	: np.str,		
'product_vendor_nm'	: np.str,		
'product_version'	: np.float,	
'product_video_url'	: np.str,		
'product_warranty_text'	: np.str,		
'product_web_display_nm'	: np.str,		
'product_workflow_status'	: np.float,
'quality_rating'	 : np.str,
'recommended_use'	: np.str,		
'rise'	: np.str,		
'room'	 : np.str,
'scent'	: np.str,		
'seller_cde'	: np.str,		
'seller_value'	: np.str,		
'shelf_quantity'	: np.str,		
'ship_alone_flg'	: np.float,
'ship_charge_eligible_flg'	: np.float,
'ship_pack_cde'	: np.str,		
'ship_pack_cde_desc'	: np.str,		
'ship_restriction_cde'	: np.str,	
'ship_restriction_value'	: np.str,		
'ship_service_cde'	: np.str,		
'ship_service_cde_desc'	: np.str,	
'shoe_width'	: np.str,	
'silhouette'	 : np.str,	
'size_sort_cde'	: np.str,	
'sku_active_flg'	: np.float,	
'sku_color'	: np.str,	
'sku_cost'	: np.float,
'sku_gift_wrap_flg'	: np.float,
'sku_id'	: np.str,	
'sku_list_price_type_cde'	: np.str,		
'sku_list_price_type_value'	: np.str,		
'sku_max_ship_lead_days'	: np.float,
'sku_min_ship_lead_days'	: np.float,	
'sku_pricing_status_cde'	: np.str,		
'sku_pricing_status_value'	: np.str,		
'sku_primary_size'	: np.str,		
'sku_replenish_ind'	: np.float,
'sku_return_info'	: np.str,		
'sku_secondary_size'	: np.str,		
'sku_size_desc'	: np.str,	
'sku_size_range'	: np.str,	
'sku_status_cde'	: np.str,	
'sku_status_value'	: np.str,	
'sku_supressed_price'	: np.int,	
'sku_surcharge_fee'	: np.float,
'sleeve_length': np.str,		
'special_sku_type_cde'	: np.str,		
'special_sku_type_value'	: np.str,	
'sports_league'	 : np.str,	
'sports_player_name'	 : np.str,
'sports_team'	 : np.str,
'style_desc'	: np.str,		
'style_id'	: np.str,		
'sub_class_nbr': np.str,		
'sub_class_nm'	: np.str,	
'supporting_nrf_color_family'	: np.str,
'tax_cde': np.str,	
'tax_eligible_flg'	: np.float,
'thread_count'	: np.str,	
'tier_count_quantity': np.str,	
'type'	: np.str,		
'upf_rating'	: np.str,	
'vendor_color_desc'	: np.str,	
'watch_clock_movement'	: np.str,		
'web_eligible_flg' : np.float,
'wicks_quantity'	: np.str,	
'window_treatment_top_construction'	 : np.str
},
nrows =2)
 
 
 
 
 
 
 
 
 
 
 
 
 
