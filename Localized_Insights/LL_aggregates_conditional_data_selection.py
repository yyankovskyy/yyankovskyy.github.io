# -*- coding: utf-8 -*-
"""

"""

import numpy as np 
import pandas as pd 


file_path = '/Users/eyankovsky/Desktop/a_LI/' 


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
header=True,
reader = pd.read_csv(file_path + file_name, sep="\t",  names=column_schema,
error_bad_lines=False, low_memory=False,
dtype=
{'TM_DIM_KY_DTE': np.str,
'SKU_NBR': np.str,
'SKU_DESC': np.str,
'STR_ID': np.str,
'PRICE': np.str,
'SALES_NET_DLRS': np.str, 
'SALES_CUST_DLRS': np.str,
'SALES_REG_DLRS': np.str,
'SALES_UNITS': np.str,
'BOH_DLRS': np.str,
'BOH_UNITS': np.str,
'BOH_COST': np.str,
'EOH_DLRS': np.str,
'EOH_UNITS': np.str,
'EOH_COST': np.str},
iterator=True, chunksize=1000) 

   

i=0
for chunk in reader:
    i+=1
    print i
    sample = pd.concat([chunk[chunk['SKU_NBR'] =='99617672']], ignore_index = True)    
    if (i%100)==0:
        sample.to_csv(file_path+ 'SKU_sample'+str(i)+'.csv', sep='\t')  
    if (i%1000) ==0:
        break
# Results:
# 10 csv files:
# 6 empty files;
# 4 files with 5 'SKU_NBR' =='99617672' observations

i=0
for chunk in reader:
    i+=1
    print i
    sample = pd.concat([chunk[chunk['SKU_NBR'] =='99617672']], ignore_index = True)    
    if (i%1000)==0:
        sample.to_csv(file_path+ 'SKU_sample'+str(i)+'out_of_1000.csv', sep='\t')  
    if (i%1000) ==0:
        break
# Results:
# 1 empty file    

i=0
for chunk in reader:
    i+=1
    print i
    sample = pd.concat(chunk[chunk['SKU_NBR'] =='99617672'], ignore_index = True) 
    if (i%100)==0:
        sample.to_csv(file_path+ 'SKU_sample'+str(i)+'.csv', sep='\t')  
    if (i%1000) ==0:
        break

import csv

i=0
for chunk in reader:
    i+=1
    print i
    sample = pd.concat([chunk[chunk['SKU_NBR'] =='99617672']], ignore_index = True) 
    if (i%100)==0:
        with open(file_path+ 'SKU_sample'+str(i)+'.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow([])
            writer.writerow(sample)
    if (i%1000) ==0:
        break


i=0
for chunk in reader:
    i+=1
    print i
    sample = pd.concat(sample, chunk[])    
    if (i%10)==0:
        sample.to_csv(file_path+ 'SKU_sample'+str(i)+'.csv', sep='\t')    
    if (i%100) ==0:
        break
    





 
 
 
 
 
