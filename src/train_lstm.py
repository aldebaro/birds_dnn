import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse
#from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

# routines for feature extraction
from utils_frontend import *

MAX_DURATION = 300

def pad_sequence_of_matrices(X: list, desired_duration: int, value: int):
    '''
    X is a list of numpy arrays.
    '''
    num_matrices = len(X)
    #print(X[0])
    #print(X[0].shape)
    dimension, duration = X[0].shape
    X_out = value * np.ones( (num_matrices, dimension, desired_duration) )
    for i in range(num_matrices):
        thisX = X[i]
        this_dimension, this_duration = thisX.shape
        if this_dimension != dimension:
            raise Exception("Dimensions should be the same", this_dimension, dimension)
        if this_duration > desired_duration:
            print("Warning: truncation", this_duration, desired_duration)
            X_out[i,:,:] = thisX[:,:desired_duration]            
        else: # padding
            X_out[i,:,:this_duration] = thisX
    return X_out

# Load data
def load_data(path):
    path = os.path.normpath(path)
    X, y = [], []
    i = 0
    for label, subfolder in enumerate(os.listdir(path)):
        full_path = os.path.join(path, subfolder)
        for file in os.listdir(full_path):
            file_path = os.path.join(full_path, file)
            spectrogram, label_index_y = read_instances_from_file(file_path)
            #spectrogram = np.load(file_path)  # assuming .npy format
            #AK TODO this is not elegant but will convert from np.array
            #to list, because Keras pad_sequences seems to require list
            X.append(spectrogram)
            y.append(label_index_y)
            print(i, file_path, "shape=", spectrogram.shape)
            i += 1
        #if i > 5:
        #    break
    # Assuming `sequences` is your list of lists or arrays
    #print("len(X)", len(X))
    #print("X=", X)
    # from https://www.kaggle.com/code/rhythmcam/keras-basic-pad-sequences-usage
    X = pad_sequence_of_matrices(X, 100, -2)
    #X = pad_sequences(X, maxlen=100, value=-2, padding='post')
    return X, np.array(y)

def train_model(input_path, max_duration = 1000):
    X, y = load_data(input_path)
    
    # Pad sequences for consistent input size
    #X = np.array([np.pad(x, ((0, max_duration - len(x)), (0, 0))) for x in X])

    # Convert labels to categorical
    y = to_categorical(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define LSTM model
    model = Sequential()
    if True:
        # use CNN
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X.shape[2],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X.shape[2],1))
        model.add(Conv2D(30,kernel_size=(30,30),input_shape=(X.shape[1], X.shape[2],1)))
        model.add(MaxPooling2D(pool_size=(2,1)))
        model.add(Conv2D(30,kernel_size=(20,20)))
        model.add(MaxPooling2D(pool_size=(2,1)))
        model.add(Conv2D(30,kernel_size=(10,10)))
        model.add(Conv2D(30,kernel_size=(3,3)))
        #model.add(Conv1D(64,kernel_size=(3,)))
        model.add(Flatten())
    else:
        # use LSTM
        model.add(LSTM(200, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(LSTM(100))
    model.add(Dense(y.shape[1], activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
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
    train_model(args.features_folder, max_duration = max_duration)