#2016-04-25 to 2016-05-22, evaluation period sales (2016-05-23 to 2016-06-19).

import pandas as pd
import numpy as np
from numpy import save
import pickle
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta



filepath = '/Users/apple/Documents/workspace/kaggle/m5_forecast'

CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "bool", 'snap_TX': 'bool', 'snap_WI': 'bool' }
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16","sell_price":"float32" }

numcols = [f"d_{day}" for day in range(1731,1942)] #6 months before validation date range 
index_cols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

TRAIN_DTYPES = ({numcol:'category' for numcol in numcols})
TRAIN_DTYPES.update({index_col:'category' for index_col in index_cols})

train = pd.read_csv(filepath+'/m5-forecasting-accuracy/sales_train_evaluation.csv',usecols=numcols+index_cols, dtype=TRAIN_DTYPES)
calendar = pd.read_csv(filepath+'/m5-forecasting-accuracy/calendar.csv',parse_dates=['date'],skiprows=range(1,1731), dtype=CAL_DTYPES)
price = pd.read_csv(filepath+'/m5-forecasting-accuracy/sell_prices.csv', dtype=PRICE_DTYPES)
#sample = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")


"""Reframing Dataframe"""
train = pd.melt(train, id_vars=index_cols, value_vars=numcols, var_name='d', value_name='sales')
train = train.merge(calendar, on= "d", copy = False)
train = train.merge(price, on = ["store_id", "item_id", "wm_yr_wk"], copy = False)

#have to unstack on promo as well 
sales = train.set_index(["store_id", "item_id", "cat_id", "date"])[["sales"]].unstack(level=-1).fillna(0)
sales['sales'] = sales['sales'].astype('int')
sales.columns = sales.columns.get_level_values(-1)

#sub dataframes to extract features from.. should make alterate promo version for every sub
store_sales = sales.groupby('store_id')[sales.columns].sum()
item_sales = sales.groupby('item_id')[sales.columns].sum()
department_sales = sales.groupby('cat_id')[sales.columns].sum()


#helper function for extract features 
def get_date_range(date,minus,periods,freq='D'):
    return pd.date_range(date-timedelta(days=minus),periods=periods,freq=freq)

# takes in a df and date, and returns moving-average features prior to date as a df.
# Also returns next 28 days y values (unit sales) 
def extract_features(df, date):
    X={}

    #if len(df.index.names>1), set prefix to blank else prefix = df.index.names[0]

    for i in [3,7,14]:
        tmp = df[get_date_range(date,i,i)]
        X['diff_mean_%s' %i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_decay%s' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values #h
        X['mean_%s' % i] = tmp.mean(axis=1).values # i 
        X['median_%s' % i] = tmp.median(axis=1).values #j
        X['min_%s' % i] = tmp.min(axis=1).values #k
        X['max_%s' % i] = tmp.max(axis=1).values #l
        X['std_%s' % i] = tmp.std(axis=1).values #m
    
    #if len(df.index.names>1), set prefix to blank else prefix = df.index.names[0]

    X=pd.DataFrame(X)
    
    if len(df.index.names)<2: #len of index=1 when groupedby
        X.columns = ['%s_%s' % (df.index.names[0], c) for c in X.columns]
        X.index = df.index
    else:
        X.index = sales.index

    return X

#THIS SHOULD BE CONVERTED TO A FUNCTION THAT CAN ALL DFS INTO ONE FEATURE DF THAT IT REUTURNS
X_list = []
Y_list = []
for i in range(6):
    start_date = date(2016,3,14)+relativedelta(weeks=i)
    
    y_train = sales[pd.date_range(start_date,periods=28)].values
    
    x_tmp = extract_features(sales,start_date)
    x_tmp2 = extract_features(item_sales,start_date)
    x_tmp3 = extract_features(store_sales,start_date)
    x_tmp4 = extract_features(department_sales,start_date)
    
    #then merge everything on x_tmp using the necessary index as key
    x_tmp=x_tmp.merge(x_tmp2, left_on = 'item_id',right_index=True, how='left')
    x_tmp=x_tmp.merge(x_tmp3, left_on = 'store_id',right_index=True, how='left')
    x_tmp=x_tmp.merge(x_tmp4, left_on = 'cat_id',right_index=True, how='left')
    x_tmp = x_tmp.reset_index()
    print(x_tmp.shape)
    
    X_list.append(x_tmp)
    Y_list.append(y_train)

X_train = pd.concat(X_list,axis=0)
Y_train = np.concatenate(Y_list,axis=0)


val_date = date(2016,4,25)
y_val = sales[pd.date_range(val_date,periods=28)].values

#extract features before val date
xv_tmp = extract_features(sales,val_date)
xv_tmp2 = extract_features(item_sales,val_date)
xv_tmp3 = extract_features(store_sales,val_date)
xv_tmp4 = extract_features(department_sales,val_date)
#then merge everything on xv_tmp using the necessary index as key
xv_tmp=xv_tmp.merge(x_tmp2, left_on = 'item_id',right_index=True, how='left')
xv_tmp=xv_tmp.merge(x_tmp3, left_on = 'store_id',right_index=True, how='left')
xv_tmp=xv_tmp.merge(x_tmp4, left_on = 'cat_id',right_index=True, how='left')
x_val = xv_tmp.reset_index()


test_date = date(2016,5,23)
#X_test, extract features before test_date
xt_tmp = extract_features(sales,test_date)
xt_tmp2 = extract_features(item_sales,test_date)
xt_tmp3 = extract_features(store_sales,test_date)
xt_tmp4 = extract_features(department_sales,test_date)
#merge
xt_tmp=xv_tmp.merge(x_tmp2, left_on = 'item_id',right_index=True, how='left')
xt_tmp=xv_tmp.merge(x_tmp3, left_on = 'store_id',right_index=True, how='left')
xt_tmp=xv_tmp.merge(x_tmp4, left_on = 'cat_id',right_index=True, how='left')
x_test = xv_tmp.reset_index()


X_train.to_pickle('x_train.p')
save('y_train.npy',Y_train) 
x_val.to_pickle('x_val.p') 
save('y_val.npy',y_val)
x_test.to_pickle('x_test.p')





""" The training and validation sets are ready to be fed into models"""

# idea is to call models from a file, by using import, and predicting validation values.
# make df of various validation-period predictions  
# fit model using validation-predictions df with validation-actual values
# use model to make predictions on test period - input will be test-predictions



""" This will be the data wrangling script
 - Purpose is to write x_train, x_val, x_test and y values to seperate csv's
 - Then make additional model scripts that will read in the needed input and will output predicitons to csv as well
 - Finally create an ensemble script using above instructions to give an ensemble test period prediction 
 """ 

 