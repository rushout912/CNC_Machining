import csv
import datetime
import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import wave
i=0
f,ax = plt.subplots(2)

# Prepare the Plotting Environment with random starting values
x = 0
y = 0


# Plot 0 is for raw audio data
li, = ax[0].plot(x, y)

ax[0].set_xlim(0,500)
ax[0].set_ylim(-1000,5000)
ax[0].set_title("Raw Audio Signal")
# Plot 1 is for the FFT of the audio
li2, = ax[1].plot(x, y)
ax[1].set_xlim(0,5000)
ax[1].set_ylim(0,50)
ax[1].set_title("Short Time Fourier Transform")
      
# Show the plot, but without blocking updates
plt.pause(0.01)
plt.tight_layout()

FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 1
RATE = 16000
CHUNK = 1024 # 1024bytes of data read from a buffer
RECORD_SECONDS = 30
overlap= 0
WAVE_OUTPUT_FILENAME = "test.wav"

audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=1,frames_per_buffer=CHUNK)

# Open a WAV file for writing
wf = wave.open(WAVE_OUTPUT_FILENAME, "w")

# Set the WAV file parameters
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)

                    

global keep_going
keep_going = True

def plot_data(in_data):
    audio_data = np.frombuffer(in_data, np.int16)
    # Compute the frequencies of the FFT
    N = len(audio_data)
    sample_rate = RATE
    T = 1./sample_rate
    frequencies = fftpack.fftfreq(N, T)
    
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    dfft = 10*np.log10(abs(np.fft.rfft(audio_data)))
    
    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    print (len(audio_data[0:10]))
    print (dfft[0:10])
    
    # Adjust the buffer size using a pop function
    buffer_size = 500
    if len(audio_data) > buffer_size:
        audio_data = audio_data[-buffer_size:]
        #dfft = dfft[-buffer_size:]
        frequencies = frequencies[-buffer_size:]
    
    li.set_xdata(np.arange(len(audio_data)))
    li.set_ydata(audio_data)
    li2.set_xdata(np.arange(len(dfft))*10)
    li2.set_ydata(dfft)
    
    # Write the audio data to the WAV file
    wf.writeframes(in_data)        

    plt.pause(0.01)
    # Open a CSV file for writing
    with open("audio_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row
        writer.writerow(["frequency", "amplitude", "class"])
        class_value = 1 # assign class value here for labeling healthy and faulty fan sounds.

        # Write the data to the CSV file
        for i, (frequency, amplitude) in enumerate(zip(frequencies, dfft)):
            writer.writerow([frequency, amplitude, class_value])
           
    if keep_going:
        return True
    else:
        return False

# Open the connection and start streaming the data
stream.start_stream()
    


# Loop so program doesn't end while the stream callback's
# itself for new data
while keep_going:
    try:
        plot_data(stream.read(CHUNK, exception_on_overflow=False))

    except KeyboardInterrupt:
        keep_going=False
        stream.stop_stream()
        stream.close()
        audio.terminate()
        wf.close()


    except:
        pass             



