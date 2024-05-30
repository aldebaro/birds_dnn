import sounddevice as sd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
# replace with your file path
filename = "./train_audio/asbfly/XC49755.ogg"
y, sr = librosa.load(filename)

# Play the audio file
sd.play(y, sr)
print("Sampling rate = ", sr)

# Use this to pause until the file is done playing
sd.wait()
