
import pandas as pd
import numpy as np
from numpy import save
import pickle
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16
}


def run_lgbm(Xtrain,Ytrain,Xval,Yval,Xtest):
    
    MAX_ROUNDS = 5000
    #train_pred = []
    val_pred = []
    test_pred = []
    #cate_vars = []
    
    for i in range(28):
        print("=" * 50)
        print("Step %d" % (i+1))
        print("=" * 50)
        dtrain = lgb.Dataset(
            Xtrain, label=Ytrain[:, i]
        )
        dval = lgb.Dataset(
            Xval, label=Yval[:, i], reference=dtrain
        )
        bst = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dtrain, dval], early_stopping_rounds=125, verbose_eval=50
        )
        print("\n".join(("%s: %.2f" % x) for x in sorted(
            zip(Xtrain.columns, bst.feature_importance("gain")),
            key=lambda x: x[1], reverse=True
        )))
        
        
        val_pred.append(bst.predict(
            Xval, num_iteration=bst.best_iteration or MAX_ROUNDS))
        
        test_pred.append(bst.predict(
            Xtest, num_iteration=bst.best_iteration or MAX_ROUNDS))
    print("")

    
    print("Validation mse:", mean_squared_error(Yval, np.array(val_pred).transpose()))

    return val_pred,test_pred






filepath = '/Users/apple/Documents/workspace/kaggle/m5_forecast'
sample = pd.read_csv(filepath+"/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

X_train = pd.read_pickle('x_train.p')
X_val = pd.read_pickle('x_val.p')
y_train = load('y_train.npy')
y_val = load('y_val.npy')
X_test = pd.read_pickle('x_test.p')

val_pred,test_pred=run_lgbm(X_train,y_train,X_val,y_val,X_test)


#creating submission dfs
day_cols = ['F{}'.format(i) for i in range(1,29)]
val_submission = pd.DataFrame(val_pred,day_cols)
val_submission = val_submission.T
val_submission['id'] = submission_items_val

val_cols = val_submission.columns.tolist() #reordering columns according to submission 
val_cols = val_cols[-1:] + val_cols[:-1]

val_submission = val_submission[val_cols]
val_submission.to_pickle('lgbm_val_submission.p')


test_submission = pd.DataFrame(test_pred,day_cols)
test_submission = test_submission.T
test_submission['id'] = submission_items_test
test_cols = test_submission.columns.tolist() #reordering columns according to submission 
test_cols = test_cols[-1:] + test_cols[:-1]

test_submission = test_submission[test_cols]

submission = pd.concat([val_submission,test_submission],axis=0,ignore_index=True)
submission = submission.set_index('id')
submission = submission.reindex(sample['id'].values)
submission = submission.reset_index()
submission.to_pickle('lgbm_submission.p')