import keras
import tensorflow as tf
import numpy as np

def dnn_model(input_shape, output_shape):
    def model_init():
        flatten_size = np.prod(input_shape)
        model = keras.Sequential()
        # model.add(keras.layers.Flatten(input_shape=input_shape)) #
        model.add(keras.layers.Dense(units=flatten_size //4, activation='relu', input_shape=(flatten_size,)))
        model.add(keras.layers.Dense(units=flatten_size //2, activation='relu'))
        model.add(keras.layers.Dense(units=output_shape))
        model.compile(loss='mse', optimizer= 'adam', metrics = 'mse')
        return model
    return model_init

def cnn_model(input_shape, output_shape):
    def model_init():
        flatten_size = np.prod(input_shape)
        model = keras.Sequential()
        model.add(keras.layers.Reshape(input_shape, input_shape=(flatten_size,)))
        model.add(keras.layers.Conv1D(filters=32, kernel_size=2, padding='same', activation='relu', input_shape=input_shape))
        model.add(keras.layers.MaxPooling1D(pool_size=2, padding='same'))
        model.add(keras.layers.Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling1D(pool_size=2, padding='same'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=output_shape))
        model.compile(loss='mse', optimizer= 'adam', metrics = 'mse')
        return model
    return model_init

def rnn_model(input_shape, output_shape):
    def model_init():
        flatten_size = np.prod(input_shape)
        model = keras.Sequential()
        model.add(keras.layers.Reshape(input_shape, input_shape=(flatten_size,)))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=32, return_sequences=True)))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=8, return_sequences=True)))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=2, return_sequences=True)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=output_shape))
        model.compile(loss='mse', optimizer= 'adam', metrics = 'mse')
        return model
    return model_init
