from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.multioutput import MultiOutputRegressor


from dataset import read_data_from_file
from base_model import dnn_model, cnn_model, rnn_model

# data
data_path = 'data/容器HPC运行数据.xlsx'
window_size = 3
stride = 1
input_shape = (window_size, 7)
output_shape = 3

# hyper parameters
epochs = 100
batch_size = 32

# load data
data, label = read_data_from_file(data_path, window_size, stride)
print(data.shape)
print(label.shape)

# define base models
dnn_model = KerasRegressor(build_fn=dnn_model(input_shape, output_shape), epochs=epochs, batch_size=batch_size, verbose=0)
cnn_model = KerasRegressor(build_fn=cnn_model(input_shape, output_shape), epochs=epochs, batch_size=batch_size, verbose=0)
rnn_model = KerasRegressor(build_fn=rnn_model(input_shape, output_shape), epochs=epochs, batch_size=batch_size, verbose=0)

regressor = StackingRegressor(estimators=[('dnn', dnn_model), ('cnn', cnn_model), ('rnn', rnn_model)])
regressor = MultiOutputRegressor(regressor)
kfold = KFold(n_splits=10)
results = cross_val_score(regressor, data, label, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))
