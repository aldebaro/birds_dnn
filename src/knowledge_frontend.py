'''
This code uses simple features to represent
the audio sequences.
'''
import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import sys
import json
# from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import argparse
import pickle

# routines for feature extraction
from utils_features import *
from utils_frontend import *


def get_features_filename(input_filename):
    folder_with_label_name, file_id, basename = parse_filename(input_filename)
    this_folder = os.path.join(features_folder, folder_with_label_name)
    if not os.path.exists(this_folder):
        # create folder if it does not exist
        os.makedirs(this_folder, exist_ok=True)
        print("Created folder", this_folder)
    base_file, ext = os.path.splitext(basename)
    output_file = os.path.join(this_folder, base_file + "_fea.hdf5")
    return output_file


def parse_filename(filename):
    '''
    Assumes a full filename such as
    C:\github\birds_dnn\train_audio\asbfly\XC134896.ogg
    '''
    folders = filename.split(os.sep)
    # same as in file train_metadata.csv
    file_id = folders[-2] + '/' + folders[-1]
    folder_with_label_name = folders[-2]  # find the label
    basename = folders[-1]
    return folder_with_label_name, file_id, basename


def calculate_features(wave_file, my_filter, should_plot=True):
    '''
    Calculate features and write to a file.
    '''
    # print("Processing {}...".format(wave_file))
    audio, Fs = librosa.load(wave_file)
    # print(type(features), features.__class__)

    # frequencies for F0 calculation
    minF0Frequency = 200
    maxF0Frequency = 5000

    window_size = 500
    window_shift = 300
    windows = extract_windows(audio, window_size, window_shift)

    features = list()

    # Apply the filter to obtain signal within passband
    filtered_signal = apply_filter(audio, my_filter)
    windows_filtered = extract_windows(
        filtered_signal, window_size, window_shift)

    for win_num in range(len(windows)):
        y = windows[win_num]
        F0, Rmax, lag_max, lags, R = estimate_fundamental_frequency(
            y, Fs, minF0Frequency=minF0Frequency, maxF0Frequency=maxF0Frequency)

        y_filtered = windows_filtered[win_num]
        power = calculate_power(y_filtered)

        this_window_features = [F0, power]
        features.append(this_window_features)
        if should_plot:
            # AK: TODO the x and y-axis are not
            # valid for mel and Magnasco, because the use log scale
            # in frequency and another time interval in time axis
            fig, axs = plt.subplots(2, 1)
            plt.tight_layout()

            # Time domain waveform plot
            axs[0].plot(np.arange(len(audio))/Fs, audio)
            axs[0].set(  # title='Time Domain Waveform',
                # xlabel='Time (s)',
                ylabel='Amplitude')

            # Time domain waveform plot
            axs[1].plot(np.arange(len(audio))/Fs, audio)
            axs[1].set(  # title='Time Domain Waveform',
                # xlabel='Time (s)',
                ylabel='Amplitude')

            # Play the audio file
            # sd.play(audio, Fs)

        # show before playing the song
        if should_plot:
            plt.show()

        # Use this to pause until the file is done playing
        # sd.wait()

    return features


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--general_dir', help='folder with wavs_labels.csv and labels_dictionary.json files', default='../outputs')
    parser.add_argument('--show_plot', action='store_true')  # default is false
    # required arguments
    parser.add_argument('--output_dir', help='output folder', required=True)
    args = parser.parse_args()

    show_plot = args.show_plot  # plot spectrograms
    general_folder = args.general_dir

    # label_file = args.label_file
    output_folder = args.output_dir
    if not os.path.exists(output_folder):
        # create folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)
        print("Created output folder", output_folder)

    features_folder = os.path.join(output_folder)
    if not os.path.exists(features_folder):
        # create folder if it does not exist
        os.makedirs(features_folder, exist_ok=True)
        print("Created features folder", features_folder)

    # save this script in the output folder for reproducibility
    # if os.name == 'nt':
    #    command = "copy " + label_file + " " + output_folder
    # else: #Linux
    #    command = "cp " + label_file + " " + output_folder
    # os.system(command)
    # print("Executed: ", command)
    command_line_file = os.path.join(
        output_folder, 'general_frontend_commandline_args.txt')
    with open(command_line_file, 'w') as f:
        f.write(' '.join(sys.argv[0:]))
        # f.write(str(sys.argv[0:]))
    print("Saved file", command_line_file)

    # define file names
    # output_instances_file = os.path.join(output_folder, output_file)
    label_file = os.path.join(general_folder, "wavs_labels.csv")
    labels_dic_file = os.path.join(general_folder, "labels_dictionary.json")

    # read input file
    df = pd.read_csv(label_file)  # read CSV label file using Pandas
    num_examples = len(df)
    # read dictionary
    a_file = open(labels_dic_file, "r")
    label_dict_str = a_file.read()
    # https://appdividend.com/2022/01/29/how-to-convert-python-string-to-dictionary/
    label_dict = json.loads(label_dict_str)

    list_all_labels = []  # initialize list

    min_time_dimension = 1e30
    max_time_dimension = -1e30

    # Design the filter
    lowcut = 500.0  # Low cut-off frequency
    highcut = 1500.0  # High cut-off frequency
    Fs = 22050.0  # sampling frequency
    my_filter = design_bandpass_filter(lowcut, highcut, Fs)

    for i, row in df.iterrows():
        wavfile = row['filename']

        X_features = calculate_features(
            wavfile, my_filter, should_plot=show_plot)

        # find transpose to make it compatible with general_frontend.py
        X_features = list(np.array(X_features).T)

        # column annotation indicates the labels
        label = str(row['annotation']).strip()
        list_all_labels.append(label)

        # now create the label index (not the string)
        label_index_y = label_dict[label]

        output_file = get_features_filename(wavfile)
        write_instances_to_file(output_file, X_features, label_index_y)

        print(label, "from", os.path.basename(wavfile),
              "converted to", output_file)
        print("Wrote file", output_file)
        if False:  # if wants to double check
            print("X", X_features)
            print('y1', label_index_y)
            X, label_index_y = read_instances_from_file(output_file)
            print("X", X)
            print('y2', label_index_y)

    print("Considering original features")
    print("Time dimension range = ", min_time_dimension, max_time_dimension)
