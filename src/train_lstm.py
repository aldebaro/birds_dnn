import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse
# from keras_preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# routines for feature extraction
from utils_frontend import *


def train_model(input_path, max_duration=1000):
    X, y = load_data(input_path)

    # Pad sequences for consistent input size
    # X = np.array([np.pad(x, ((0, max_duration - len(x)), (0, 0))) for x in X])

    # Convert labels to categorical
    y = to_categorical(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define LSTM model
    model = Sequential()
    if True:
        # use CNN
        X_train = X_train.reshape(
            (X_train.shape[0], X_train.shape[1], X.shape[2], 1))
        X_test = X_test.reshape(
            (X_test.shape[0], X_test.shape[1], X.shape[2], 1))
        model.add(Conv2D(30, kernel_size=(30, 30),
                  input_shape=(X.shape[1], X.shape[2], 1)))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(30, kernel_size=(20, 20)))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(30, kernel_size=(10, 10)))
        model.add(Conv2D(30, kernel_size=(3, 3)))
        # model.add(Conv1D(64,kernel_size=(3,)))
        model.add(Flatten())
    else:
        # use LSTM
        model.add(LSTM(200, input_shape=(
            X.shape[1], X.shape[2]), return_sequences=True))
        model.add(LSTM(100))
    model.add(Dense(y.shape[1], activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    # Train model
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--features_folder', help='folder with features', default='../outputs/mel')
    args = parser.parse_args()

    max_duration = MAX_DURATION
    train_model(args.features_folder, max_duration=max_duration)
