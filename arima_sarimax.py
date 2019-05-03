#==============================================================================
#Required package imports
#==============================================================================
import pandas as pd
import plotly
import datetime
import gc
import sys
import warnings
import itertools
import numpy as np
warnings.filterwarnings("ignore")
import statsmodels.api as sm
import matplotlib
from pyramid.arima import auto_arima #http://www.alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from statistics import mean

#==============================================================================
#get data
#==============================================================================
#inputfile_path = sys.argv[0]
inputfile_path = "/data/Unbrickings PGH level 5_v1 test.xlsx"
data = pd.read_excel(inputfile_path, sheet_name='Sheet1' , header=0)


data = data[['Groups','rptg_dt','hier_node_level_5_name','Product','Tier','apple_hq_name','Sum(unbrickings_qty)']]
data.columns = ['Groups','rptg_dt','Country','Product','Tier','apple_hq_name','qty']


data = data.dropna()
data['week_start_dt'] = data['rptg_dt'].dt.to_period('W').apply(lambda r: r.start_time)
data = data[['Groups','Country','Product','week_start_dt','Tier','apple_hq_name','qty']]
data = data[data.Groups == 'Group-1']
gc.collect()


data_all = data[['Groups','week_start_dt','qty']]
data_all = data_all.groupby(['Groups','week_start_dt']).sum()
data_all = data_all.reset_index()

data_GRP = data_all
data_GRP = data_GRP[['week_start_dt','qty']]
data_GRP.set_index('week_start_dt', inplace=True)

#1.1
data_GRP_train = data_GRP['2015-10-05':'2018-09-24']
data_GRP_test = data_GRP['2018-10-01':'2019-01-28']

#1.2
#data_GRP_train = data_GRP['2015-10-05':'2018-06-25']
#data_GRP_test = data_GRP['2018-07-02':'2019-01-14']

#1.3
#data_GRP_train = data_GRP['2015-10-05':'2018-03-26']
#data_GRP_test = data_GRP['2018-04-02':'2019-01-14']

#1.4
#data_GRP_train = data_GRP['2015-10-05':'2017-12-25']
#data_GRP_test = data_GRP['2018-01-01':'2019-01-14']

#==============================================================================
#Creating functions to create External variables
#==============================================================================
def  markmonth(row):
    if row['week_start_dt'].month == 10:
        return 12
    elif row['week_start_dt'].month == 11:
        return 11
    elif row['week_start_dt'].month == 12:
        return 10
    elif row['week_start_dt'].month == 1:
        return 9
    elif row['week_start_dt'].month == 2:
        return 8
    elif row['week_start_dt'].month == 3:
        return 7
    elif row['week_start_dt'].month == 4:
        return 6
    elif row['week_start_dt'].month == 5:
        return 5
    elif row['week_start_dt'].month == 6:
        return 4
    elif row['week_start_dt'].month == 7:
        return 3
    elif row['week_start_dt'].month == 8:
        return 2
    elif row['week_start_dt'].month == 9:
        return 1
    
    
def mark_highsalemonth(row):
    if row['week_start_dt'].month == 12:
        return 1
    elif ((row['week_start_dt'].month == 11) and \
          (row['week_start_dt'].day in [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])):
        return 1
    elif ((row['week_start_dt'].month == 1) and \
          (row['week_start_dt'].day in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])):
        return 1
    else:
        return 0


def mark_christmas(row):
    if row['week_start_dt'].strftime("%U") in [50,51,52,1,2]:
        return 1
    else:
        return 0
    

def mark_BlackFriday(row):
    if row['week_start_dt'].strftime("%U") in [46,47,48]:
        return 1
    else:
        return 0
    

def mark_firstweek(row):
    if row['weekofmonth'] in [5,4]:
        return 1
    else:
        return 0
    
    
def mark_russia(row):
    if row['week_start_dt'].month == 3 and row['weekofmonth'] in [1,2]:
        return 1
    else:
        return 0
    
#==============================================================================
#Create External Variable Data Frame
#==============================================================================
data_GRP_train_exg = data_GRP_train
data_GRP_train_exg = data_GRP_train_exg.reset_index()

#==============================================================================
#create external variables by applyin the functions to the data frame
#create week of a month variable which will denote every week by ranking them in desc 
#i.e first week gets the hights number 
#==============================================================================
data_GRP_train_exg['year'] = data_GRP_train_exg['week_start_dt'].dt.strftime("%Y")
data_GRP_train_exg['month'] = data_GRP_train_exg['week_start_dt'].dt.month
data_GRP_train_exg['weekofmonth'] = data_GRP_train_exg.groupby(['year' , 'month'])['week_start_dt']\
                                                               .rank(ascending=False, method='dense').astype(int)
data_GRP_train_exg = data_GRP_train_exg.reset_index()
data_GRP_train_exg['month_wt'] = data_GRP_train_exg.apply(markmonth, axis=1)
data_GRP_train_exg['high_sale'] = data_GRP_train_exg.apply(mark_highsalemonth, axis=1)
data_GRP_train_exg['christmas_week'] = data_GRP_train_exg.apply(mark_christmas, axis=1)
data_GRP_train_exg['blackfriday'] = data_GRP_train_exg.apply(mark_BlackFriday, axis=1)
data_GRP_train_exg['first_week'] = data_GRP_train_exg.apply(mark_firstweek, axis=1)

# Use it only for the group with russia in it
#data_GRP_train_exg['russia8march'] = data_GRP_train_exg.apply(mark_russia, axis=1)

#==============================================================================
# Select which externals variables to use by adding in the below list
#==============================================================================
data_GRP_train_exg = data_GRP_train_exg[['week_start_dt','month_wt','high_sale']]
data_GRP_train_exg.set_index('week_start_dt', inplace=True)

#==============================================================================
#blow code is used to extract the Trend seasonality and noise
#==============================================================================
"""
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(data_GRP_train, freq=52, model='multiplicative') #multiplicative 0r additive
#NOISE
residual = decomposition.resid
#SEASIONAL
seasonal = decomposition.seasonal
#TREND
trend = decomposition.trend
"""

#==============================================================================
#Apply the train data and external variables data to the model   
#==============================================================================
stepwise_model = auto_arima(data_GRP_train, start_p=0, start_q=0,exogenous=data_GRP_train_exg, \
                           max_p=3, max_q=3, m=52, \
                           start_P=0, start_Q=0,seasonal=True, \
                           d=0, D=1, max_order=None,#alpha=0.5,\
                            stationary=False,trace=True, \
                           error_action='ignore', \
                           suppress_warnings=True, \
                           stepwise=True)

#Print the AIC and the Order of variables. 
print(stepwise_model.aic())

#print Model Summary
print(stepwise_model.summary())

#==============================================================================
#fit the model with train data.
#==============================================================================
stepwise_model.fit(data_GRP_train)

#==============================================================================
#generate first day of week date to use in forecasted output
#==============================================================================
oneday = datetime.timedelta(days=1)
oneweek = datetime.timedelta(days=7)

year = [2018,2019]
days = []
for i in year:
    start = datetime.datetime(year=i, month=1, day=1)
    while start.weekday() != 0:
        start += oneday

    while start.year == i:
        days.append(start)
        start += oneweek


df_test = pd.DataFrame({'week_start_dt':days})
df_test.set_index('week_start_dt', inplace=True)
df_test = df_test['2018-10-01':'2019-03-25']
#df_test = df_test['2018-07-02':'2019-03-25']
#df_test = df_test['2018-04-02':'2019-03-25']
#df_test = df_test['2018-01-01':'2019-03-25']

#==============================================================================
#predict the future points
#==============================================================================
future_forecast = stepwise_model.predict(n_periods=len(df_test))

#==============================================================================
#convert the predicted points in to a Data Frame
#==============================================================================
future_forecast = pd.DataFrame(future_forecast.astype(int), index = df_test.index,columns=['Prediction'])

#==============================================================================
#concat actual and predicted side by side
#==============================================================================
out_comp = pd.concat([data_GRP_test,future_forecast],axis=1)
out_comp = out_comp.reset_index()
out_comp['year_month'] = out_comp.apply(lambda x: str(x.week_start_dt.year)+str(x.week_start_dt.month), axis=1)
out_comp.set_index('week_start_dt', inplace=True)
out_comp['per_diff'] = out_comp.apply(lambda row: (abs(row.qty - row.Prediction)/row.qty)*100, axis=1)

#==============================================================================
#Plot predicted vs Actuals
#==============================================================================
"""
my_dpi=99
plt.figure(figsize=(1500/my_dpi, 330/my_dpi), dpi=my_dpi)
plt.plot(out_comp.qty,color='blue') #Actual
plt.plot(out_comp.Prediction,color='red') #Predicted

"""

#==============================================================================
#Calculating distribution percentage for each country
#==============================================================================
data_country = data
data_country['week_number'] = data_country['week_start_dt'].dt.strftime("%Y%U")
data_country = data_country[['week_number','Country','qty']]
data_country_g1 = data_country.groupby(['week_number']).sum()
data_countryv1 = data_country.groupby(['week_number','Country']).sum()
data_countryv1 = data_countryv1.reset_index()

data_country_perc_dist = pd.merge(data_country_g1, data_countryv1, on='week_number', how='inner', suffixes=('_left','_right'))

def cal_per(row):
    try:
        return row['qty_right']/row['qty_left']
    except:
        return 0.0
    

data_country_perc_dist['perc_dist'] = data_country_perc_dist.apply(cal_per, axis = 1)
data_country_perc_dist['week'] = data_country_perc_dist.apply(lambda x: x.week_number[4:], axis=1)
data_country_perc_dist = data_country_perc_dist[['week','Country', 'perc_dist']]
data_country_perc_dist = data_country_perc_dist.groupby(['week','Country'])['perc_dist'].mean().to_frame().reset_index()


out_comp = out_comp.reset_index()
out_comp['week'] = out_comp['week_start_dt'].dt.strftime("%U")

out_comp_nw = out_comp[['week_start_dt','Prediction','week']]
match_set = pd.merge(out_comp_nw, data_country_perc_dist, on=['week'])
match_set['predcited'] = match_set.apply(lambda x: x.Prediction*x.perc_dist, axis=1)
match_set = match_set[['week_start_dt','Country', 'predcited']]


#==============================================================================
#create test set to test the Country propagation.
#==============================================================================
data_countryv1_test = data
data_countryv1_test = data_countryv1_test[['week_start_dt','Country','qty']]
data_countryv1_test = data_countryv1_test.groupby(['week_start_dt','Country']).sum()

data_countryv1_test = data_countryv1_test['2018-10-01':'2019-01-14']
#data_countryv1_test = data_countryv1_test['2018-07-02':'2019-01-14']
#data_countryv1_test = data_countryv1_test['2018-04-02':'2019-01-14']
#data_countryv1_test = data_countryv1_test['2018-01-01':'2019-01-14']

#==============================================================================
#merge actual test data and predicted data
#==============================================================================
group_res = pd.merge(match_set, data_countryv1_test, on=['week_start_dt', 'Country'], how='left', suffixes=('_left','_right'))
group_res['predcited'] = group_res['predcited'].astype(int)
group_res.to_csv("/Users/MO351263/Desktop/GerSale/data/CL5_G1_q1_2019_res_EXOG_v3.csv", sep=',', encoding='utf-8')

#==============================================================================
#Apply propogation to next level
#==============================================================================
var = 'Product' #Product, Tier or apple_hq_name

data_grp_prod = data
data_grp_prod = data_grp_prod[['Country',var,'week_start_dt','qty']]
data_grp_prod = data_grp_prod.groupby(['week_start_dt','Country',var]).sum()
data_grp_prod = data_grp_prod.reset_index()
data_grp_prod.set_index('week_start_dt', inplace=True)

#==============================================================================
#select the propagation proportion calculation duration.
#==============================================================================
data_grp_prod = data_grp_prod['2018-10-01':'2018-12-24']

data_grp_prod_ctry = data_grp_prod.groupby(['week_start_dt','Country']).sum()

country_prod_perc_dist = pd.merge(data_grp_prod_ctry, data_grp_prod, on=['week_start_dt','Country'], how='inner', suffixes=('_left','_right'))

#country_prod_perc_dist['perc_dist'] = country_prod_perc_dist.apply(lambda x: x.qty_right/x.qty_left, axis = 1)
country_prod_perc_dist['perc_dist'] = country_prod_perc_dist.apply(cal_per, axis = 1)
country_prod_perc_dist = country_prod_perc_dist.groupby(['Country',var])['perc_dist'].mean().to_frame().reset_index()

group_res['tmp'] = 1
country_prod_perc_dist['tmp'] = 1

match_set_prod = pd.merge(group_res, country_prod_perc_dist, on=['tmp','Country'])
match_set_prod = match_set_prod.drop('tmp', axis=1)
match_set_prod['pred'] = match_set_prod.apply(lambda x: x.predcited*x.perc_dist, axis=1)
match_set_prod['pred'] = match_set_prod['pred'].astype(int)
match_set_prod = match_set_prod[['week_start_dt','Country',var,'pred']]

#==============================================================================
#create test set to test the Country Tier or Product or Partner propagation.
#==============================================================================
data_grp_prod_test = data
data_grp_prod_test = data_grp_prod_test[['week_start_dt','Country',var,'qty']]
data_grp_prod_test = data_grp_prod_test.groupby(['week_start_dt','Country',var]).sum()
data_grp_prod_test = data_grp_prod_test['2018-10-01':'2019-01-28']

#==============================================================================
#merger predicted by propogation with the Actuals 
#==============================================================================
ctry_prod_res = pd.merge(match_set_prod, data_grp_prod_test, on=['week_start_dt', 'Country', var], how='left', suffixes=('_left','_right'))

#==============================================================================
# write the final out to a file
#==============================================================================
ctry_prod_res.to_csv("/data/Country_"+var+"_result_v2.csv", sep=',', encoding='utf-8')
