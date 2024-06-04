import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile
import librosa


def calculate_power(signal):
    '''Supports complex-valued signals.'''
    power = np.mean(np.sum(np.abs(signal) ** 2))
    return power


def design_bandpass_filter(lowcut, highcut, fs, order=5):
    # Design a bandpass IIR filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def apply_filter(signal, sos):
    # Apply the filter to the signal
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal


def extract_windows(signal, window_size, window_shift):
    # Number of windows
    num_windows = 1 + (len(signal) - window_size) // window_shift

    # Initialize an empty list to store the windows
    windows = []

    for i in range(num_windows):
        start = i * window_shift
        end = start + window_size
        window = signal[start:end]
        windows.append(window)

    return windows


def estimate_fundamental_frequency(y, Fs, minF0Frequency=0, maxF0Frequency=-1):
    '''
    y - input signal
    Fs - sampling frequency in Hz
    maxF0Frequency - maximum F0 frequency in Hz, use -1 to indicate Fs/2
    minF0Frequency - minimum F0 frequency in Hz
    '''
    if maxF0Frequency == -1:
        maxF0Frequency = Fs/2  # use Nyquist frequency

    Ts = 1.0/Fs  # sampling interval
    maxF0Period = 1.0/maxF0Frequency  # corresponding F0 (sec)
    Nbegin = round(maxF0Period/Ts)  # number of lags for max freq.
    if minF0Frequency > 0:
        minF0Period = 1.0/minF0Frequency  # corresponding F0 (sec)
        Nend = round(minF0Period/Ts)  # number of lags for min freq.
    else:
        Nend = len(y)

    # print("Nbegin and Nend", Nbegin, Nend)
    R = np.correlate(y, y, mode='full')
    lag0_index = (len(R)-1)//2  # find index of lag=0
    R = R[lag0_index:]  # only positive lags
    lags = np.arange(0, len(R), dtype=int)

    # if I use the abs in autocorrelation,
    # it gets twice the frequency because the
    # max abs is the negative peak
    # R_mag = np.abs(R[Nbegin:Nend+1])  # only part of interest
    R_mag = R[Nbegin:Nend+1]  # only part of interest
    Rmax = np.max(R_mag)
    relative_index_max = np.argmax(R_mag)

    # we used just part of R, so recalculate the index:
    index_max = Nbegin + relative_index_max
    lag_max = lags[index_max]  # get lag corresponding to index

    F0 = 1.0/((lag_max)*Ts)  # estimated F0 frequency (Hz)

    return F0, Rmax, lag_max, lags, R


def test_F0_estimation():
    Fs = 4000  # sampling frequency
    Ts = 1.0/Fs  # sampling interval

    # test with a sine
    y = np.sin(2*np.pi*534*np.arange(0, 2*Fs)*Ts)  # 300 Hz, duration 2 secs
    # test with simple signal
    # y = np.array([1, -2+1j, -5, 3+1j, -2-1j])

    plt.subplot(211)
    plt.plot(Ts*np.arange(0, len(y)), y)
    plt.xlabel('time (s)')
    plt.ylabel('Signal y(t)')

    # print(y)
    F0, Rmax, lag_max, lags, R = estimate_fundamental_frequency(y, Fs)

    plt.subplot(212)
    plt.plot(lags, R)
    plt.xlabel('lag (s)')
    plt.ylabel('Autocorrelation of y(t)')
    plt.plot(lag_max, Rmax, 'xr', markersize=20)

    print(
        f'Rmax (max of correlation)={Rmax} lag_max={lag_max} T0={lag_max*Ts} (s) Freq.={F0} Hz')

    t = np.arange(0, 2+Ts, Ts)
    sd.play(np.cos(2*np.pi*3*F0*t), Fs)  # play freq. 3*F0

    plt.show()


def test_filter():
    # Read the audio file
    wave_file = "../train_audio/ashdro1/XC37740.ogg"
    signal, fs = librosa.load(wave_file)

    # Design the filter
    lowcut = 500.0  # Low cut-off frequency
    highcut = 1500.0  # High cut-off frequency
    sos = design_bandpass_filter(lowcut, highcut, fs)

    # Apply the filter
    filtered_signal = apply_filter(signal, sos)

    # Write the filtered audio to a new file
    # wavfile.write('filtered_audio.wav', fs, filtered_signal.astype(np.int16))


if __name__ == "__main__":
    test_F0_estimation()
    test_filter()
