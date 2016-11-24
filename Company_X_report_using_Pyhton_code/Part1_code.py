# This is a code to address task 1


import urllib, re, os, sys, time 
os.system('cls' if os.name == 'nt' else 'clear') 
path = "/Users/
os.chdir(path)


# Check current working directory.
retval = os.getcwd()
print "Current working directory %s" % retval




import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from datetime import datetime
import matplotlib.pylab as plt


df1 = pd.read_json('logins.json')
df1.login_time

df1.login_time = pd.to_datetime(df1["login_time"])

df1["login_time_sorted"]=sorted(df1["login_time"])  
del df1["login_time"]


print df1.head()
print '\n Data Types:'
print df1.dtypes


dt1 = datetime(1970,1,1,0,0,0)
groups_by_15_min= (df1["login_time_sorted"] - dt1)/np.timedelta64(60*15, 's')
df1["groups_by_15_min"]=groups_by_15_min.astype(np.int64) /1

df1["login_time_15_min_interval"] = dt1 + df1["groups_by_15_min"]*np.timedelta64(60*15, 's')



print df1.head()

df2 = pd.DataFrame({'logins' : df1.groupby( ["login_time_15_min_interval"] ).size()}).reset_index()

#logins=df1["login_time_15_min_interval"].groupby([df1["login_time_15_min_interval"]]).agg(['count'])

dates = pd.DatetimeIndex(df2["login_time_15_min_interval"])

#### Data visualization 
%matplotlib
fig=plt.plot(df2["login_time_15_min_interval"], df2["logins"])
plt.title('Number of logins in 15-min intervals')
plt.xlabel('Time, 15-min intervals')
plt.ylabel('Number of logins')


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(df2["logins"], nlags=24)
lag_pacf = pacf(df2["logins"], nlags=24, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df2["logins"])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df2["logins"])),linestyle='--',color='gray')
plt.title('Autocorrelation Function')


#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df2["logins"])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df2["logins"])),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

########################################################


month = dates.month
weekday = dates.weekday
hour=dates.hour

df2["Month"]=pd.Series(month, index=df2.index)
df2["Weekday"]=pd.Series(weekday, index=df2.index)
df2["Hour"]=pd.Series(hour, index=df2.index)

# Boxplot charts


%matplotlib

# Box plot by month
fig1=df2.boxplot('logins', by="Month")
plt.title('Distribution of logins in 15-min intervals aggregated by a month')
fig1.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

#fig1.savefig((os.path.join(chart_path, fig1)) )

# Box plot by weekday
fig2=df2.boxplot('logins', by="Weekday")
plt.title('Distribution  of logins in 15-min intervals aggregated by a weekday')
fig2.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()


# Box plot by hour
fig3=df2.boxplot('logins', by="Hour")
plt.title('Distribution of logins in 15-min intervals aggregated by an hour')
fig3.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()


# Charts by a weekday in a month

df_Jan = df2[df2["Month"]==1]
fig4=df_Jan.boxplot('logins', by="Weekday")
plt.title('Distribution of logins in 15-min intervals aggregated by a weekday in January')
fig4.set_ylim(0, 80)
fig4.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Feb = df2[df2["Month"]==2]
fig5=df_Feb.boxplot('logins', by="Weekday")
plt.title('Distribution of logins in 15-min intervals aggregated by a weekday in February')
fig5.set_ylim(0, 80)
fig5.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Mar = df2[df2["Month"]==3]
fig6=df_Mar.boxplot('logins', by="Weekday")
plt.title('Distribution of logins in 15-min intervals aggregated by a weekday in March')
fig6.set_ylim(0, 80)
fig6.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Apr = df2[df2["Month"]==4]
fig7=df_Apr.boxplot('logins', by="Weekday")
plt.title('Distribution of logins in 15-min intervals aggregated by a weekday in April')
fig7.set_ylim(0, 80)
fig7.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

# Charts by an hour in a weekday
df_Sat = df2[df2["Weekday"]==5]
fig8=df_Sat.boxplot('logins', by="Hour")
plt.title('Distribution of logins in 15-min intervals aggregated hourly on Saturdays')
fig8.set_ylim(0, 80)
fig8.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Sun = df2[df2["Weekday"]==6]
fig9=df_Sun.boxplot('logins', by="Hour")
plt.title('Distribution of logins in 15-min intervals aggregated hourly on Sundays')
fig9.set_ylim(0, 80)
fig9.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Mon = df2[df2["Weekday"]==0]
fig10=df_Mon.boxplot('logins', by="Hour")
plt.title('Distribution of logins in 15-min intervals aggregated hourly on Mondays')
fig10.set_ylim(0, 80)
fig10.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Tue = df2[df2["Weekday"]==1]
fig11=df_Tue.boxplot('logins', by="Hour")
plt.title('Distribution of logins in 15-min intervals aggregated hourly on Tuesdays')
fig11.set_ylim(0, 80)
fig11.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Wed = df2[df2["Weekday"]==2]
fig12=df_Wed.boxplot('logins', by="Hour")
plt.title('Distribution of logins in 15-min intervals aggregated hourly on Wednesdays')
fig12.set_ylim(0, 80)
fig12.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Thu = df2[df2["Weekday"]==3]
fig13=df_Thu.boxplot('logins', by="Hour")
plt.title('Distribution of logins in 15-min intervals aggregated hourly on Thursdays')
fig13.set_ylim(0, 80)
fig13.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()

df_Fri = df2[df2["Weekday"]==4]
fig14=df_Fri.boxplot('logins', by="Hour")
plt.title('Distribution of logins in 15-min intervals aggregated hourly on Fridays')
fig14.set_ylim(0, 80)
fig14.set_ylabel("Number of logins")
plt.suptitle("")
plt.show()
