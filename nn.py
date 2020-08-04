import numpy as np
from numpy import load
from datetime import date, timedelta
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
#import gc
filepath = '/Users/apple/Documents/workspace/kaggle/m5_forecast'
sample = pd.read_csv(filepath+"/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

X_train = pd.read_pickle('x_train.p')
X_val = pd.read_pickle('x_val.p')
y_train = load('y_train.npy')
y_val = load('y_val.npy')
X_test = pd.read_pickle('x_test.p')


submission_items_val = X_val['item_id'].astype(str)+'_'+X_val['store_id'].astype(str)+'_validation'
submission_items_test = X_test['item_id'].astype(str)+'_'+X_test['store_id'].astype(str)+'_evaluation'

X_train = X_train.drop(['store_id','item_id','cat_id'],axis=1)
x_val = X_val.drop(['store_id','item_id','cat_id'],axis=1)
x_test = X_test.drop(['store_id','item_id','cat_id'],axis=1)


scaler = StandardScaler()
scaler.fit(pd.concat([X_train, X_val, X_test]))
X_train[:] = scaler.transform(X_train)
X_val[:] = scaler.transform(X_val)
X_test[:] = scaler.transform(X_test)

X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
X_val = X_val.as_matrix()
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

def build_model():
    model = Sequential()
    model.add(LSTM(512, input_shape=(X_train.shape[1],X_train.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(.2))

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(256))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.1))

    model.add(Dense(128))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(64))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(32))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(16))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(.05))

    model.add(Dense(1))

    return model

N_EPOCHS = 1
val_pred = []
test_pred = []

for i in range(28):
	print("=" * 50)
	print("Step %d" % (i+1))
	print("=" * 50)
	y = y_train[:, i]
	y_mean = y.mean()
	xv = X_val
	yv = y_val[:, i]
	model = build_model()
	opt = optimizers.Adam(lr=0.001)
	model.compile(loss='mse', optimizer=opt, metrics=['mse'])

	callbacks = [
	    EarlyStopping(monitor='val_loss', patience=10, verbose=0),
	    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
	    ]
	model.fit(X_train, y - y_mean, batch_size = 65536, epochs = N_EPOCHS, verbose=2,
	        validation_data=(xv,yv-y_mean), callbacks=callbacks )
	val_pred.append(model.predict(X_val)+y_mean)
	test_pred.append(model.predict(X_test)+y_mean)

print(val_pred)


#creating submission dfs
day_cols = ['F{}'.format(i) for i in range(1,29)]
val_submission = pd.DataFrame(val_pred,day_cols)
val_submission = val_submission.T
val_submission['id'] = submission_items_val

val_cols = val_submission.columns.tolist() #reordering columns according to submission 
val_cols = val_cols[-1:] + val_cols[:-1]

val_submission = val_submission[val_cols]
val_submission.to_pickle('nn_val_submission.p')


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
submission.to_pickle('nn_submission.p')


# save validation preidctions using pickle (needs to be used for fitting in ensemble) 
# save test predictions as csv in submission format