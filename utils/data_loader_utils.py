#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""





"""

""" 
# Copyright (c) 2021 Bosch Rexroth AG
# SPDX-License-Identifier: BSD-3-Clause

Author: Mohamed-Ali Tnani (Mohamed-ali.tnani@boschrexroth.de)
Date: November 24, 2021

This software is provided by the copyright holders and contributors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the copyright holder or contributors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pywt


def find_all_h5s_in_dir(s_dir):
    """
    list all .h5 files in a directory
    """

    fileslist = []
    for root, dirs, files in os.walk(s_dir):
        for file in files:
            if file.endswith(".h5"):
                fileslist.append(file)
    return fileslist


def load_tool_research_data(data_path, label, add_additional_label=True, verbose=True):
    """
    load data (good and bad) from the research data storages
    
    Keyword Arguments:
            data_path {str} -- [path to the directory] 
            label {str} -- ["good" or "bad"]
            add_additional_label {bool} -- [if true the labels will be in the form of "Mxx_Aug20xx_OPxx_sampleNr_label" otherwise "label"] (default: True)
            verbose {bool}

        Returns:
            datalist --  [list of the X samples]
            label --  [list of they labels ]
    """
    datalist = []
    data_label = []

    # list all .h5 files
    list_paths = find_all_h5s_in_dir(data_path)
    list_paths.sort()

    # read and append the samples with the corresponding labels
    if verbose:
        print(f"loading files from {data_path}... ")
    for element in list_paths:
        # check if additional label needed ("Mxx_Aug20xx_Tool,nrX") 
        if add_additional_label:
            add_label = element.split('/')[-1]
            additional_label = add_label[:-3] + "_" + label
        else:
            additional_label = label
        # extract data X and y 
        with h5py.File(os.path.join(data_path, element), 'r') as f:
            vibration_data = f['vibration_data'][:] # type: ignore
        datalist.append(vibration_data)
        data_label.append(additional_label)

    return datalist, data_label


def datafile_read(file, plotting=True, wavelet='db4', level=5):
    """loads and plots the data from the datafile

    Keyword Arguments:
        file {str} -- [path of the file] 

    Returns:
        ndarray --  [raw data original]
        ndarray -- [mel-spectrogram for x-axis]
        ndarray -- [mel-spectrogram for y-axis]
        ndarray -- [mel-spectrogram for z-axis]
        list -- [approximate coefficients for x-axis]
        list -- [detailed coefficients for x-axis]
        list -- [approximate coefficients for y-axis]
        list -- [detailed coefficients for y-axis]
        list -- [approximate coefficients for z-axis]
        list -- [detailed coefficients for z-axis]
        ndarray -- [STFT magnitude spectrum for x-axis]
        ndarray -- [STFT magnitude spectrum for y-axis]
        ndarray -- [STFT magnitude spectrum for z-axis]
    """
    with h5py.File(file, 'r') as f:
        vibration_data = f['vibration_data'][:] # type: ignore   
    # interpolation for x axis plot
    freq = 2000
    samples_s = len(vibration_data[:, 0]) / freq # type: ignore
    samples = np.linspace(0, samples_s, len(vibration_data[:, 0])) # type: ignore
    #mel_spectrogram analysis
    mel_spectrogram_x = librosa.feature.melspectrogram(y=vibration_data[:, 0], sr=freq, n_fft=2048, hop_length=512, n_mels=128) # type: ignore
    mel_spectrogram_y = librosa.feature.melspectrogram(y=vibration_data[:, 1], sr=freq, n_fft=2048, hop_length=512, n_mels=128) # type: ignore
    mel_spectrogram_z = librosa.feature.melspectrogram(y=vibration_data[:, 2], sr=freq, n_fft=2048, hop_length=512, n_mels=128) # type: ignore
    axes = ['X', 'Y', 'Z']

    # Discrete Wavelet Transform analysis
    dwt_x = pywt.wavedec(vibration_data[:, 0], wavelet, level=level) # type: ignore
    dwt_y = pywt.wavedec(vibration_data[:, 1], wavelet, level=level) # type: ignore
    dwt_z = pywt.wavedec(vibration_data[:, 2], wavelet, level=level) # type: ignore
    approx_coeffs_x = dwt_x[0]
    detail_coeffs_x = dwt_x[1:]
    approx_coeffs_y = dwt_y[0]
    detail_coeffs_y = dwt_y[1:]
    approx_coeffs_z = dwt_z[0]
    detail_coeffs_z = dwt_z[1:]
    '''approx_coeffs_x, detail_coeffs_x = [], []
    approx_coeffs_y, detail_coeffs_y = [], []
    approx_coeffs_z, detail_coeffs_z = [], []'''



    for i, ax in enumerate(axes):
        coeffs = pywt.wavedec(eval(f"vibration_data[:, i]"), wavelet, level=level)
        approx_coeffs, detail_coeffs = coeffs[0], coeffs[1:]
        approx_coeffs_x = np.concatenate([approx_coeffs_x, approx_coeffs], axis=0)
        approx_coeffs_y = np.concatenate([approx_coeffs_y, approx_coeffs], axis=0)
        approx_coeffs_z = np.concatenate([approx_coeffs_z, approx_coeffs], axis=0)
    stft_x = np.abs(librosa.stft(vibration_data[:, 0], n_fft=2048, hop_length=512, win_length=2048, window='hann', center=False)) # type: ignore
    stft_y = np.abs(librosa.stft(vibration_data[:, 1], n_fft=2048, hop_length=512, win_length=2048, window='hann', center=False)) # type: ignore
    stft_z = np.abs(librosa.stft(vibration_data[:, 2], n_fft=2048, hop_length=512, win_length=2048, window='hann', center=False)) # type: ignore
    # plotting
    if plotting:
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 0], 'b') # type: ignore
        plt.ylabel('X-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10)) # type: ignore
        plt.grid()
        plt.show()
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 1], 'b') # type: ignore
        plt.ylabel('Y-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10)) # type: ignore
        plt.grid()
        plt.show()
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 2], 'b') # type: ignore
        plt.ylabel('Z-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10)) # type: ignore
        plt.grid()
        plt.show()
        for i, ax in enumerate(axes):
            plt.figure(figsize=(20, 5))
            librosa.display.specshow(librosa.power_to_db(eval(f"mel_spectrogram_{ax.lower()}")), sr=freq, hop_length=512, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'{ax}-axis Vibration Data Mel-Spectrogram')
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10)) # type: ignore
            plt.xlim(0, librosa.get_duration(y=vibration_data[:, i], sr=freq)) # type: ignore
            plt.grid()
            plt.show()
        
        # plot DWT results
        """plt.figure(figsize=(10, 2))
        plt.plot(approx_coeffs_x, label='X-axis approx. coefficients')
        plt.plot(detail_coeffs_x[-1], label='X-axis detailed coefficients')
        plt.legend()
        plt.title('Discrete Wavelet Transform - X-axis')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 2))
        plt.plot(approx_coeffs_y, label='Y-axis approx. coefficients')
        plt.plot(detail_coeffs_y[-1], label='Y-axis detailed coefficients')
        plt.legend()
        plt.title('Discrete Wavelet Transform - Y-axis')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 2))
        plt.plot(approx_coeffs_z, label='Z-axis approx. coefficients')
        plt.plot(detail_coeffs_z[-1], label='Z-axis detailed coefficients')
        plt.legend()
        plt.title('Discrete Wavelet Transform - Z-axis')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()"""
        '''def plot_dwt_window(approx_coeffs, detail_coeffs, title, window_size=5000):
            current_start = 0
            current_end = window_size

            def plot_window(start, end):
                plt.clf()
                plt.plot(approx_coeffs[start:end], label='Approx. coefficients')
                plt.plot(detail_coeffs[-1][start:end], label='Detailed coefficients')
                plt.title(title)
                plt.xlabel('Samples')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid()
                plt.draw()
                plt.pause(0.001)

            plt.ion()  # Turn on interactive mode
            plot_window(current_start, current_end)

            while True:
                key = input("Enter 'left', 'right', or 'quit': ")

                if key == 'right':
                    current_start += window_size
                    current_end += window_size
                elif key == 'left':
                    current_start -= window_size
                    current_end -= window_size
                elif key == 'quit':
                    break
                else:
                    print("Invalid input. Try again.")
                    continue

                current_start = max(current_start, 0)
                current_end = min(current_end, len(approx_coeffs))
                plot_window(current_start, current_end)

            plt.ioff()  # Turn off interactive mode'''
        def plot_dwt_window(approx_coeffs, detail_coeffs, title, window_size=5000):
            n_samples = len(approx_coeffs)
            start_indices = range(0, n_samples, window_size)
            end_indices = [min(start_idx + window_size, n_samples) for start_idx in start_indices]
            
            plt.figure(figsize=(20,5))
            plt.plot(approx_coeffs, label='Approx. coefficients')
            plt.plot(detail_coeffs[-1], label='Detailed coefficients')
            plt.title(title)
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid()
            for start, end in zip(start_indices, end_indices):
                plt.axvspan(start, end, alpha=0.1, color='gray')
            plt.show()


        # Call the plot_dwt_window function for each axis
        window_size = 5000  # Adjust this value according to your needs
        plot_dwt_window(approx_coeffs_x, detail_coeffs_x, 'Discrete Wavelet Transform - X-axis', window_size)
        plot_dwt_window(approx_coeffs_y, detail_coeffs_y, 'Discrete Wavelet Transform - Y-axis', window_size)
        plot_dwt_window(approx_coeffs_z, detail_coeffs_z, 'Discrete Wavelet Transform - Z-axis', window_size) 
        for i, ax in enumerate(axes):
            plt.figure(figsize=(20, 5))
            D = np.abs(librosa.stft(eval(f"vibration_data[:, i]"), n_fft=2048, hop_length=512, win_length=2048, window='hann', center=False))
            D = librosa.amplitude_to_db(D, ref=np.max) # type: ignore
            mel_spec = librosa.feature.melspectrogram(y=eval(f"vibration_data[:, i]"), sr=freq, n_fft=2048, hop_length=512, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max) # type: ignore
            librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='linear', sr=freq, hop_length=512, vmin=-80, vmax=0)
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'{ax}-axis Vibration Data STFT Magnitude Spectrum')
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(10)) # type: ignore
            plt.grid()
            plt.show()

              

    return vibration_data, mel_spectrogram_x, mel_spectrogram_y, mel_spectrogram_z, approx_coeffs_x, detail_coeffs_x, approx_coeffs_y, detail_coeffs_y, approx_coeffs_z, detail_coeffs_z, stft_x, stft_y, stft_z