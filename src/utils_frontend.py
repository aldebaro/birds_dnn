'''
This code provides functions for the feature extraction module.
All feature extractors output a matrix organized as frequency x time
(number of rows is the number of frequency points, with the row 0 (top)
representing the highest frequency).
'''
import numpy as np
import pandas as pd
import os
# from numba import jit
import librosa
import librosa.display
import sys
import h5py
import json
from numpy.fft import fft
from scipy.io.wavfile import read, write
from skimage.transform import resize
import matplotlib.pyplot as plt
import torch
import reassignment.reassignment_linear as reassign_lin
# from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler

MAX_DURATION = 300


def load_data(path):
    # Load data
    path = os.path.normpath(path)
    X, y = [], []
    i = 0
    for label, subfolder in enumerate(os.listdir(path)):
        full_path = os.path.join(path, subfolder)
        if not os.path.isdir(full_path):
            continue  # skip files, to process only folders
        for file in os.listdir(full_path):
            file_path = os.path.join(full_path, file)
            spectrogram, label_index_y = read_instances_from_file(file_path)
            # spectrogram = np.load(file_path)  # assuming .npy format
            # AK TODO this is not elegant but will convert from np.array
            # to list, because Keras pad_sequences seems to require list
            X.append(spectrogram)
            y.append(label_index_y)
            print(i, file_path, "shape=", spectrogram.shape)
            i += 1
        # if i > 5:
        #    break
    # Assuming `sequences` is your list of lists or arrays
    # print("len(X)", len(X))
    # print("X=", X)
    # from https://www.kaggle.com/code/rhythmcam/keras-basic-pad-sequences-usage

    # TODO find a sensible value instead of -2
    X = pad_sequence_of_matrices(X, 100, -2)
    # X = pad_sequences(X, maxlen=100, value=-2, padding='post')
    return X, np.array(y)


def pad_sequence_of_matrices(X: list, desired_duration: int, value: int):
    '''
    X is a list of numpy arrays.
    '''
    num_matrices = len(X)
    # print(X[0])
    # print(X[0].shape)
    dimension, duration = X[0].shape
    X_out = value * np.ones((num_matrices, dimension, desired_duration))
    for i in range(num_matrices):
        thisX = X[i]
        this_dimension, this_duration = thisX.shape
        if this_dimension != dimension:
            raise Exception("Dimensions should be the same",
                            this_dimension, dimension)
        if this_duration > desired_duration:
            print("Warning: truncation", this_duration, desired_duration)
            X_out[i, :, :] = thisX[:, :desired_duration]
        else:  # padding
            X_out[i, :, :this_duration] = thisX
    return X_out

# Part 1) File manipulation


def read_instances_from_file(intputHDF5FileName):
    '''
    Read X and y from HDF5 file.
    '''
    h5pyFile = h5py.File(intputHDF5FileName, 'r')
    Xtemp = h5pyFile["X"]
    ytemp = h5pyFile["y"]
    X = np.array(Xtemp[()])
    y = np.array(ytemp[()])
    h5pyFile.close()
    return X, y


def write_instances_to_file(outputHDF5FileName, X, y):
    '''
    Write X and y to HDF5 file.
    '''
    h5pyFile = h5py.File(outputHDF5FileName, 'w')
    h5pyFile["X"] = X  # store in hdf5
    h5pyFile["y"] = y  # store in hdf5
    h5pyFile.close()
    # print('==> Wrote file ', outputHDF5FileName, ' with keys X and y')

# Part 2) Time-frequency feature extraction algorithms


def magnasco_spectrogram(audio):
    '''
    Use Magnasco algorithm with reassign_lin.high_resolution_spectrogram.
    '''
    # defining constants and parameters
    q = 2
    tdeci = 96
    over = 2
    noct = 108
    # noct = 200 #to increase dimension in frequency axis
    minf = 1.5e-2
    # minf = 1e-3
    # maxf = 0.16 # lower max frequency
    maxf = 0.5

    spectrogram = reassign_lin.high_resolution_spectrogram(
        audio, q, tdeci, over, noct, minf, maxf, device=torch.device('cpu'))
    # port the spectrogram to cpu
    # No need to do it if device = cpu
    spectrogram = spectrogram.cpu().numpy().T
    # spectrogram = spectrogram.cpu().numpy()
    spectrogram = np.flipud(spectrogram)
    return spectrogram


'''
Assume Fs = 44100 Hz.
The STFT is calculated with dimension num_freq_bins x num_time_frames
but the code outputs a tranposed array with dimenstion
num_time_frames x num_freq_bins, which are called T and D in the code, respectively.
'''


def stft_spectrogram(audio, n_fft=2048, window_shift=500):
    '''
    STFT
    n_fft = 2048  # default FFT size
    '''
    win_length = n_fft
    hop_length = window_shift
    stft = librosa.stft(audio.astype(float), n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window='hann', center=True, dtype=None, pad_mode='constant')
    stft = np.abs(stft)**2.0
    return stft


def melspectrogram(audio, sr=22050, n_fft=2048, window_shift=500, n_mels=128):
    '''
    Mel-scaled spectrogram.
    https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
    n_mels is the number of filter outputs, which is the number of rows in the output matrix
    '''
    n_fft = n_fft  # FFT size
    sr = sr  # sampling frequency in Hz
    win_length = n_fft
    hop_length = window_shift
    melspectrogram = librosa.feature.melspectrogram(y=audio.astype(float), sr=sr,
                                                    n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',
                                                    center=True, dtype=None, pad_mode='constant', n_mels=n_mels, fmax=sr/2)
    return melspectrogram

# Part 3) Normalization of features to feed neural networks


def normalize_as_maggie(spectrogram):
    # Original author: Jiayi (Maggie) Zhang <jiayizha@andrew.cmu.edu>
    min_value = np.min(spectrogram)
    if min_value >= 0:
        # check because it may be already in log domain
        spectrogram = np.log(spectrogram+1e-8)
    spectrogram /= np.max(np.abs(spectrogram))
    spectrogram += 1
    return spectrogram


def normalize_to_min_max(spectrogram, min=-1, max=1):
    # numbers are now from something like -80 or -60 dB to 0 dB.
    # Move them to defined range. Default is -1 to 1
    # Using sklearn MinMaxScaler, note it works for each column, so
    # reshape to make it a single column array
    min_max_scaler = MinMaxScaler(feature_range=(min, max))
    # Stack everything into a single column to scale by the global min / max
    original_shape = spectrogram.shape
    tmp = spectrogram.reshape(-1, 1)
    spectrogram = min_max_scaler.fit_transform(tmp).reshape(original_shape)
    # print('bbbb', np.max(spectrogram), np.min(spectrogram))
    return spectrogram


def normalize_standardize_along_frequency(spectrogram):
    num_freq_bins, num_time_frames = spectrogram.shape
    for i in range(num_time_frames):
        dft = spectrogram[:, i]
        mu = np.mean(dft)
        std = np.std(dft)
        if std > 0:
            spectrogram[:, i] = (dft - mu) / std
        else:
            spectrogram[:, i] = (dft - mu)
    return spectrogram

# Part 4) Signal statistics and plots


'''
Plot feature matrix.
'''


def plot_feature(features, title):
    plt.axis('off')
    # plt.yticks(y_range, y_axis[::-1])
    # plt.xticks(x_range, x_axis)
    plt.imshow(features, cmap='inferno')
    plt.title(title)
    plt.colorbar()
    plt.show()


'''
Plot feature matrix.
'''


def plot_feature_no_show(features, title):
    plt.clf()
    plt.axis('off')
    # plt.yticks(y_range, y_axis[::-1])
    # plt.xticks(x_range, x_axis)
    plt.imshow(features, cmap='inferno')
    title = title + ", shape=", str(features.shape)
    plt.title(title)
    plt.colorbar()


def get_stats(X):
    min_x = np.min(X)
    max_x = np.max(X)
    mean_x = np.mean(X)
    return min_x, max_x, mean_x


'''
https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
'''


def plot_overall_histogram(X):
    # https://stackoverflow.com/questions/13730468/from-nd-to-1d-arrays
    all_values = X.ravel()
    plt.hist(all_values, bins=100, log=True)
    plt.title('Histogram of feature values from all examples')
    plt.xlabel('feature value')
    plt.ylabel('number of occurrences')
    plt.show()


def values_above_threshold_per_frequency(X):
    show_plots = True
    num_bins = 100
    threshold = np.min(X.ravel())  # can use a number such as 0.5

    # recall X has dimension TIME x FREQ because transpose had been used
    (num_examples, T, D) = X.shape
    if show_plots:
        plot_overall_histogram(X)
    min_x, max_x, mean_x = get_stats(X.ravel())
    # print(min_x, max_x, mean_x)
    range_all_numbers = [min_x, max_x]  # define range
    occurrences_above_reference = np.zeros((D,), dtype=int)
    for i in range(D):  # go over all frequencies
        # for all examples, and all time instants
        values_given_frequency = X[:, :, i].ravel()
        # min_x, max_x, mean_x = get_stats(values_given_frequency)
        # print(min_x, max_x, mean_x)
        n, bins = np.histogram(values_given_frequency,
                               bins=num_bins, range=range_all_numbers)
        # find the bin corresponding to the threshold value
        indices = np.array(np.where(bins < threshold)).ravel()
        if np.any(indices):
            last_index = int(indices[-1])  # did not check this logic
        else:
            last_index = 1  # threshold is the minimum value, and accounts only for first bin
        occurrences_above_reference[i] = np.sum(n[last_index:])
    if show_plots:
        plt.plot(occurrences_above_reference)
        plt.yscale('log')
        plt.grid()
        plt.title('Histogram of values above reference = ' + str(threshold))
        plt.xlabel('frequency index (from 0 to D-1)')
        plt.show()
    return occurrences_above_reference
