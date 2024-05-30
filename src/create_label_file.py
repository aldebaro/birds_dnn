'''
Find all .ogg files in a directory and create CSV to become Pandas dataframe
'''
# Redirect stdout with > to save to a file.
# Example:
# python automation/create_label_file.py ../wav ../general

import glob
import os.path
import numpy as np
import librosa
import sys
import pandas as pd
import json

show_histograms = True  # use True to list histograms in the end


def write_label_dictionary(label_file, output_labels_dic_file):
    # read input file
    df = pd.read_csv(label_file)  # read CSV label file using Pandas
    # initialize and pre-allocate space
    label_dict = {}  # dictionary
    cur_label = 0
    for i, row in df.iterrows():
        # column annotation indicates the labels
        label = row['annotation'].strip()
        if label not in label_dict:
            label_dict[label] = cur_label
            cur_label += 1
    a_file = open(output_labels_dic_file, "w")
    json.dump(label_dict, a_file)
    a_file.close()


'''
Find one substring from given list of substrings in a string
'''


def find_substring_in_string(string, substrings_list):
    num_substrings = len(substrings_list)
    found = False
    this_substring = None
    this_substring_index = -1
    for i in range(num_substrings):
        substring = substrings_list[i].lower()
        if string.find(substring) != -1:
            if found == True:
                print("ERROR: found two substrings", substring,
                      "and", this_substring, "in string", string)
                exit(-1)
            found = True
            this_substring = substrings_list[i]
            this_substring_index = i
    if not found:
        print("ERROR: could not find any label in", string)
        exit(-1)
    return this_substring, this_substring_index


# input folder with ogg files (the extension must be ogg, not WAV or something else)
def write_to_file(folder, fp):
    expected_Fs = 22050  # expected sample rate (all files should have it)
    # note: the label calla is the same as caya
    # labels = ["caba", "cada", "caga", "caca",
    # speakers = ["Speaker1", "Speaker2", "Speaker3", "Speaker4",

    # assumed header for labels
    header = "filename,annotation"  # offset,starts,stops,speaker"

    # print("Reading folder", folder)
    # print(header)
    fp.write(header)
    fp.write('\n')
    iterator = glob.glob(os.path.join(folder, "**\*.ogg"), recursive=True)
    num_wav_files = len(iterator)
    print("Found", num_wav_files, "files with extension ogg in folder", folder)
    for wave_file in iterator:
        # print("Processing {}...".format(wave_file))

        # read waveform and check sampling frequency
        x, Fs = librosa.load(wave_file)  # x is a numpy array
        if Fs != expected_Fs:
            print("ERROR:", wave_file, "has sampling frequency", Fs)
            exit(-1)

        # process the file name, searching for its label and speaker in filename
        filename = os.path.normpath(wave_file)
        filename = os.path.abspath(filename)

        folders = filename.split(os.sep)
        # same as in file train_metadata.csv
        file_id = folders[-2] + '/' + folders[-1]
        this_label = folders[-2]  # find the label

        # It is assumed:
        # header = "filename,annotation" #assumed header for labels
        # print(filename,",",this_label,",",this_label_index,",0,0,",x.shape[0],",",this_speaker,sep='')
        output_string = filename + "," + this_label
        fp.write(output_string)
        fp.write('\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("ERROR!")
        print("Usage: input_wav_folder output_folder")
        print("Example:")
        print(r"python create_label_file.py ../wav ../output/")
        exit(1)
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(input_folder):
        print("ERROR: folder", input_folder, "does not exist!")
        exit(-1)

    if not os.path.exists(output_folder):
        # create folder if it does not exist
        os.makedirs(output_folder, exist_ok=True)
        print("Created output folder", output_folder)

    output_file = os.path.join(output_folder, "wavs_labels.csv")
    with open(output_file, 'w') as fp:
        write_to_file(input_folder, fp)
    fp.close()
    print("Wrote file", output_file)

    output_labels_dic_file = os.path.join(
        output_folder, "labels_dictionary.json")
    label_file = output_file
    write_label_dictionary(label_file, output_labels_dic_file)
    print("Wrote file", output_labels_dic_file)
