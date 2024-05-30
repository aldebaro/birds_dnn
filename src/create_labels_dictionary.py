'''
Find all .ogg files in a directory and create CSV to become Pandas dataframe
'''
import matplotlib.pyplot as plt
import argparse
import glob
import os.path
import numpy as np
import librosa
import pandas as pd

should_plot = True  # Plot the histogram or not


def check_labels(label_file, input_folder, extension='ogg'):
    '''
    Goes over all audio files in a folder and verifies 
    whether the label can be found.
    '''
    # read CSV label file using Pandas
    search_string = os.path.join(input_folder, '**/*.' + extension)
    df = pd.read_csv(label_file, delimiter=';')
    print(df)
    # All labels, extracted from train_metadata.csv
    # Their correct order is very important!
    labels = ['asbfly', 'ashdro1', 'ashpri1', 'ashwoo2', 'asikoe2', 'asiope1', 'aspfly1', 'aspswi1', 'barfly1', 'barswa', 'bcnher', 'bkcbul1', 'bkrfla1', 'bkskit1', 'bkwsti', 'bladro1', 'blaeag1', 'blakit1', 'blhori1', 'blnmon1', 'blrwar1', 'bncwoo3', 'brakit1', 'brasta1', 'brcful1', 'brfowl1', 'brnhao1', 'brnshr', 'brodro1', 'brwjac1', 'brwowl1', 'btbeat1', 'bwfshr1', 'categr', 'chbeat1', 'cohcuc1', 'comfla1', 'comgre', 'comior1', 'comkin1', 'commoo3', 'commyn', 'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crbsun2', 'cregos1', 'crfbar1', 'crseag1', 'dafbab1', 'darter2', 'eaywag1', 'emedov2', 'eucdov', 'eurbla2', 'eurcoo', 'forwag1', 'gargan', 'gloibi', 'goflea1', 'graher1', 'grbeat1', 'grecou1', 'greegr', 'grefla1', 'grehor1', 'grejun2', 'grenig1', 'grewar3', 'grnsan', 'grnwar1', 'grtdro1', 'gryfra', 'grynig2', 'grywag', 'gybpri1', 'gyhcaf1', 'heswoo1', 'hoopoe', 'houcro1', 'houspa', 'inbrob1', 'indpit1', 'indrob1', 'indrol2', 'indtit1', 'ingori1', 'inpher1', 'insbab1',
              'insowl1', 'integr', 'isbduc1', 'jerbus2', 'junbab2', 'junmyn1', 'junowl1', 'kenplo1', 'kerlau2', 'labcro1', 'laudov1', 'lblwar1', 'lesyel1', 'lewduc1', 'lirplo', 'litegr', 'litgre1', 'litspi1', 'litswi1', 'lobsun2', 'maghor2', 'malpar1', 'maltro1', 'malwoo1', 'marsan', 'mawthr1', 'moipig1', 'nilfly2', 'niwpig1', 'nutman', 'orihob2', 'oripip1', 'pabflo1', 'paisto1', 'piebus1', 'piekin1', 'placuc3', 'plaflo1', 'plapri1', 'plhpar1', 'pomgrp2', 'purher1', 'pursun3', 'pursun4', 'purswa3', 'putbab1', 'redspu1', 'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar', 'rossta2', 'rufbab3', 'ruftre2', 'rufwoo2', 'rutfly6', 'sbeowl1', 'scamin3', 'shikra1', 'smamin1', 'sohmyn1', 'spepic1', 'spodov', 'spoowl1', 'sqtbul1', 'stbkin1', 'sttwoo1', 'thbwar1', 'tibfly3', 'tilwar1', 'vefnut1', 'vehpar1', 'wbbfly1', 'wemhar1', 'whbbul2', 'whbsho3', 'whbtre1', 'whbwag1', 'whbwat1', 'whbwoo2', 'whcbar1', 'whiter2', 'whrmun', 'whtkin2', 'woosan', 'wynlau1', 'yebbab1', 'yebbul3', 'zitcis1']
    num_labels = len(labels)  # 182 labels

    # Create a dictionary with all label
    labels_dictionary = dict()  # empty dictionary
    for i in range(num_labels):
        labels_dictionary[labels[i]] = i  # i is the i-th label
    print(labels_dictionary)

    # verifies the label for each audio file
    for wave_file in glob.glob(search_string, recursive=True):
        # print("Processing {}...".format(wave_file))
        path = os.path.normpath(wave_file)
        folders = path.split(os.sep)
        # same as in file train_metadata.csv
        file_id = folders[-2] + '/' + folders[-1]
        # https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
        result_df = df.loc[df['filename'] == file_id]
        num_columns = result_df.size
        # if num_rows != 1:
        #    raise Exception("It should have a single row")
        if num_columns != 12:
            raise Exception("It should have 12 columns")
        primary_label = result_df.iloc[0]['primary_label']
        if primary_label != folders[-2]:
            raise Exception("Logic error!")
        # file_name = os.path.basename(wave_file)
        label_index = labels_dictionary[primary_label]
        print(primary_label, " => ", label_index)


def check_sampling_frequency_and_duration(input_folder, extension='ogg'):
    '''
    Verifies if all files have sampling frequency 22050 Hz
    and computes histogram of audio durations.
    '''
    search_string = os.path.join(input_folder, '**/*.' + extension)
    print("search_string=", search_string)
    # find all files with given extension in current folder
    num_files = 0
    min_duration = 1e30
    max_duration = 0
    durations = list()
    for wave_file in glob.glob(search_string, recursive=True):
        # print("Processing {}...".format(wave_file))
        # print("Original audio:", s) #enable if want to see information about the waveform
        # path_and_file_name = os.path.splitext(
        #    wave_file)[0]  # discard extension
        # file_name = os.path.basename(wave_file)
        s, Fs = librosa.load(wave_file)
        num_samples = s.shape[0]
        Ts = 1.0 / Fs  # sampling interval
        duration = num_samples * Ts
        if duration < min_duration:
            min_duration = duration
        if duration > max_duration:
            max_duration = duration
        # print("Duration, min, max", duration, min_duration, max_duration)
        durations.append(duration)
        # print('sampling rate (Hz) =', Fs)
        if Fs != 22050:
            raise Exception("Unexpected sampling rate: " + str(Fs))
        num_files += 1
    print("All " + str(num_files) + " had sampling rate = 22050 Hz")
    # Create histogram
    hist, bins = np.histogram(durations, bins=20)  # 20 bins
    if should_plot:
        # Plot the histogram
        plt.hist(durations, bins=20, alpha=0.7,
                 color='blue', edgecolor='black')
        plt.title('Histogram of durations (in seconds)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder', help='input folder with audio files', default='..\\train_audio')
    parser.add_argument(
        '--output_folder', help='input folder with audio files', default='../besta')
    # parser.add_argument('--Dmin',type=int,help='First index of chosen frequency range. Default is 0, the first in input file',default=0)

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print("ERROR: folder", args.input_folder, "does not exist!")
        exit(-1)

    if not os.path.exists(args.output_folder):
        # create folder if it does not exist
        os.makedirs(args.output_folder, exist_ok=True)
        print("Created output folder", args.output_folder)

    output_file = os.path.join(args.output_folder, "besta.csv")

    # check_sampling_frequency_and_duration(args.input_folder, extension='ogg')

    label_file = '../train_metadata.csv'
    check_labels(label_file, args.input_folder)
