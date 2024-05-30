'''
Find all .ogg files, listen and see.
'''
import matplotlib.pyplot as plt
import argparse
import glob
import os.path
import numpy as np
import librosa
import pandas as pd
import sounddevice as sd
# routines for feature extraction
from utils_frontend import *


def loop_over_all_files(input_folder, extension='ogg', should_plot=True, features='stft', normalization_method="none"):
    '''
    Verifies if all files have sampling frequency 22050 Hz
    and computes histogram of audio durations.
    '''
    search_string = os.path.join(input_folder, '**/*.' + extension)
    print("search_string=", search_string)
    # find all files with given extension in current folder
    num_files = 0
    for wave_file in glob.glob(search_string, recursive=True):
        print("Playing {}...".format(wave_file))
        audio, Fs = librosa.load(wave_file)

        if features == 'magnasco':
            spectrogram = magnasco_spectrogram(audio)
        elif features == 'stft':
            spectrogram = stft_spectrogram(audio)
        elif features == 'mel':
            num_mel_filters = 200
            spectrogram = melspectrogram(
                audio, n_mels=num_mel_filters)
        else:
            raise Exception("Invalid features: " + features)

        top_db = 100  # floor value in dB below the maximum
        spectrogram = librosa.amplitude_to_db(
            spectrogram, ref=np.max, top_db=top_db)

        print("Shape of the features = ", spectrogram.shape)

        if normalization_method != "none":
            if normalization_method == "maggie":
                spectrogram = normalize_as_maggie(spectrogram)
            if normalization_method == "minmax":
                spectrogram = normalize_to_min_max(spectrogram)
            if normalization_method == "std_freq":
                spectrogram = normalize_standardize_along_frequency(
                    spectrogram)

        if should_plot:
            # AK: TODO the x and y-axis are not
            # valid for mel and Magnasco, because the use log scale
            # in frequency and another time interval in time axis
            fig, axs = plt.subplots(3, 1)
            plt.tight_layout()

            # specify the sampling frequency below:
            img = librosa.display.specshow(
                spectrogram, sr=Fs, x_axis='time', y_axis='linear', ax=axs[0])
            axs[0].set(title='Feature = ' + features)
            fig.colorbar(img, ax=axs[0], format="%+2.f dB")
            # plot_feature(spectrogram, "Features " + features)
            # plt.colorbar()

            # Time domain waveform plot
            axs[1].plot(np.arange(len(audio))/Fs, audio)
            axs[1].set(  # title='Time Domain Waveform',
                # xlabel='Time (s)',
                ylabel='Amplitude')
            fig.colorbar(img, ax=axs[1], format="%+2.f dB")

            # Energy plot
            energy = 10.0*np.log10(audio**2 + 1e-30)
            energy[energy < -top_db] = -top_db
            axs[2].plot(np.arange(len(audio))/Fs, energy)
            axs[2].set(  # title='Energy',
                xlabel='Time (s)', ylabel='Energy (dB)')
            # Create colorbar
            mappable = plt.cm.ScalarMappable(cmap='viridis')
            mappable.set_array(energy)

            plt.colorbar(mappable, ax=axs[2],
                         orientation='vertical', label='Energy')

        # Play the audio file
        sd.play(audio, Fs)

        # show before playing the song
        if should_plot:
            plt.show()

        # Use this to pause until the file is done playing
        sd.wait()

        num_files += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder', help='input folder with audio files', default='..\\train_audio')
    parser.add_argument(
        '--features', help='adopted features (stft, mel or magnasco)', default='stft')
    parser.add_argument(
        '--normalization_method', help='normalization method (none, maggie, minmax, std_freq)', default='none')

    features = 'stft',
    # default is false
    parser.add_argument('--should_plot', action='store_true',
                        help='enable to plot the histogram')

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print("ERROR: folder", args.input_folder, "does not exist!")
        exit(-1)

    durations = loop_over_all_files(args.input_folder, extension='ogg',
                                    should_plot=args.should_plot,
                                    features=args.features,
                                    normalization_method=args.normalization_method)
