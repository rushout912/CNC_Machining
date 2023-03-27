import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
import time
from scipy.io import wavfile
from scipy.signal import find_peaks

# List of file paths for multiple wav files
filenames = ['healthy_fan.wav', 
             'healthy_fan2.wav', 
             'faulty_fan.wav']

for i, filename in enumerate(filenames):
    y, sr = librosa.load(filename)
    y = np.array(y, dtype=float)
    
    # Plot the raw audio signal
    plt.figure(figsize=(20, 5))
    plt.plot(y)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.title(f'Raw Audio Signal for {filename}')
    plt.show()
    time.sleep(3)

    # Compute the STFT
    n_fft = 2048
    hop_length = 512
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    mel_spectogram = librosa.feature.melspectrogram(y=y, sr=sr)
    
    librosa.display.specshow(librosa.power_to_db(mel_spectogram), sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-frequency Spectrogram for {filename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # set x-axis limit according to audio stream recording duration
    plt.xlim(0, librosa.get_duration(y=y, sr=sr))
    plt.show()
    time.sleep(3)

    # Plot the STFT applied Spectogram
    plt.figure(figsize=(20, 5))
    librosa.display.specshow(librosa.amplitude_to_db(stft), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Short-Time Fourier Transform Applied Spectogram for {filename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.xlim(0, librosa.get_duration(y=y, sr=sr))
    plt.show()
    time.sleep(3)

    # detect peak point
    peak_indices = find_peaks(stft.ravel())[0]
    
    # Plot peak points on top of STFT
    plt.title(f'Peak points STFT Graph for {filename}')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.scatter(peak_indices[1]*hop_length/sr, peak_indices[0]*sr/(2*n_fft), color='red', label='peak points')
    plt.legend()
    # set x-axis limit according to audio stream recording duration
    plt.xlim(0, librosa.get_duration(y=y, sr=sr))
    plt.show()
    time.sleep(3)


for filename in filenames:
    samplerate, data = wavfile.read(filename)
    t = np.arange(len(data)) / float(samplerate)

    print("data -> ", data)
    print("data length -> ", len(data))
    print("np.arange(len(data)) -> ", np.arange(len(data)))
    print("float(samplerate) -> ", float(samplerate))
    print("time -> ", t)

    data = data/max(data);  # Normalize Audio Data
    print("normalized data -> ", data)

    coeffs = pywt.wavedec(data, 'bior6.8', mode='sym', level=2)
    cA2, cD2, cD1 = coeffs
    y = pywt.waverec(coeffs, 'bior6.8', mode='sym')

    wavfile.write('sampley.wav', samplerate, y)
    wavfile.write('samplecA2.wav', samplerate, cA2)
    wavfile.write('samplecD2.wav', samplerate, cD2)
    wavfile.write('samplecD1.wav', samplerate, cD1)


    # Formatting for figure
    L = len(data)
    y = y[0:L];  # Matching length with input for plotting

    plt.figure(figsize=(30, 20))

    plt.subplot(5, 1, 1)
    plt.plot(t, data, color='k')
    plt.xlabel('Time')
    plt.ylabel('S')
    plt.title(f'Original Signal for {filename}')

    plt.subplot(5, 1, 2)
    plt.plot(cA2, color='r')
    plt.xlabel('Samples')
    plt.ylabel('cA2')
    plt.title(f'Approximation Coeff. for {filename} (cA2)')

    plt.subplot(5, 1, 3)
    plt.plot(cD2, color='g')
    plt.xlabel('Samples')
    plt.ylabel('cD2')
    plt.title(f'Detailed Coeff. for {filename} (cD1)')

    plt.subplot(5, 1, 4)
    plt.plot(cD1, color='m')
    plt.xlabel('Samples')
    plt.ylabel('cD1')
    plt.title(f'Detailed Coeff.for {filename} (cD2)')

    plt.subplot(5, 1, 5)
    plt.plot(t, y, color='b')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Reconstructed Signal for {filename}')

    # Saving plot as PNG image
    plt.savefig(f'plot_{filename}.png', dpi=100)

    plt.show()
