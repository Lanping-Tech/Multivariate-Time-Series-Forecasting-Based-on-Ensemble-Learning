from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import numpy as np

from dataset import read_data_from_file
from base_model import dnn_model, cnn_model, rnn_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# data
data_path = 'data/容器HPC运行数据（2022.08.20）.xlsx'
window_size = 3
stride = 1
input_shape = (window_size, 7)
output_shape = 1

# hyper parameters
epochs = 20
batch_size = 32

# load data
data, label = read_data_from_file(data_path, window_size, stride)
train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.1, random_state=42)

train_x = train_x.reshape(-1, np.prod(input_shape))
test_x = test_x.reshape(-1, np.prod(input_shape))


dnn = dnn_model(input_shape, output_shape)
cnn = cnn_model(input_shape, output_shape)
rnn = rnn_model(input_shape, output_shape)

# define base models
dnn_regressor = KerasRegressor(model=dnn, epochs=epochs, batch_size=batch_size, verbose=0)
cnn_regressor = KerasRegressor(model=cnn, epochs=epochs, batch_size=batch_size, verbose=0)
rnn_regressor = KerasRegressor(model=rnn, epochs=epochs, batch_size=batch_size, verbose=0)

final_regressor = RandomForestRegressor(n_estimators=10, random_state=42)

regressor = MultiOutputRegressor(estimator=StackingRegressor(estimators=[('dnn', dnn_regressor), ('cnn', cnn_regressor), ('rnn', rnn_regressor)], final_estimator=final_regressor))

regressor = regressor.fit(train_x, train_y)
pred_y = regressor.predict(test_x)

print(test_y.shape, pred_y.shape)

print('MSE: ', mean_squared_error(test_y, pred_y))
print('MAE: ', mean_absolute_error(test_y, pred_y))
print('R2: ', r2_score(test_y, pred_y))


