import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

def dnn_model(input_shape, output_shape):
    flatten_size = np.prod(input_shape)
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape)) #
    model.add(keras.layers.Dense(units=flatten_size //4, activation='relu'))
    model.add(keras.layers.Dense(units=flatten_size //2, activation='relu'))
    model.add(keras.layers.Dense(units=output_shape))
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
    return model

def cnn_model(input_shape, output_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=output_shape))
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
    return model

def rnn_model(input_shape, output_shape):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=8, return_sequences=True)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=2, return_sequences=True)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=output_shape))
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.mean_squared_error)
    return model
