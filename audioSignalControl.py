import pygame
import numpy as np
import time
from ZMQcomms import paired_messenger
import sys
from random import gauss
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(signal_in, sampling_rate=30000.0, highpass_frequency=300.0, lowpass_frequency=6000.0, filt_order=4):
    b, a = butter_bandpass(highpass_frequency, lowpass_frequency, sampling_rate, order=filt_order)
    signal_out = lfilter(b, a, signal_in)
    return signal_out

def createWhiteNoiseSound(frequency, sampleRate, peak, frequency_band_width):
    # Create white noise of one second length
    arr = np.array([gauss(0.0, 1.0) for i in range(sampleRate)])
    # Filter signal to specified frequency band
    cut_off_low = frequency - frequency_band_width / 2.0
    cut_off_high = frequency + frequency_band_width / 2.0
    arr = butter_bandpass_filter(arr, sampleRate, cut_off_low, cut_off_high, filt_order=4)
    # Convert sound signal to correct range and type
    arr = arr / max(abs(arr)) * peak
    arr = arr.astype(np.int16)

    return arr

def createSineWaveSound(frequency, sampleRate, peak):
    arr = np.array([peak * np.sin(2.0 * np.pi * frequency * x / sampleRate) for x in range(0, sampleRate)])
    arr = arr.astype(np.int16)

    return arr

def createAudioSignal(frequency, frequency_band_width=1000, modulation_frequency=0):
    '''
    Calling this function must be preceded by calling the 
    following lines for pygame to recognise the mono sound
    pygame.mixer.pre_init(48000, -16, 2) # here 48000 needs to match the sampleRate in this function
    pygame.init()
    '''
    sampleRate = 48000 # Must be the same as in the line 
    peak = 4096 # 4096 : the peak ; volume ; loudness
    if int(round(frequency_band_width)) == 0:
        arr = createSineWaveSound(frequency, sampleRate, peak)
    else:
        arr = createWhiteNoiseSound(frequency, sampleRate, peak, frequency_band_width)
    # Modulate signal at specified frequency if requested
    if modulation_frequency > 0:
        modulation_frequency = int(modulation_frequency)
        marr = np.array([np.sin(2.0 * np.pi * modulation_frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.float64)
        marr = marr - min(marr)
        marr = marr / max(marr)
        arr = np.int16(arr.astype(np.float64) * marr)
    # Turn sound array into a two-channel pygame sound object
    arr = np.repeat(arr[:,None],2,axis=1)
    sound = pygame.sndarray.make_sound(arr)

    # Use sound.play(-1) to start sound. (-1) means it is in infinite loop
    # Use sound.stop() to stop the sound

    return sound

class audioControl(object):
    def __init__(self, frequency, frequency_band_width=1000, modulation_frequency=0):
        # Initialize pygame for playing sound
        pygame.mixer.pre_init(48000, -16, 2)
        pygame.mixer.init()
        pygame.init()
        # Get sound
        self.sound = createAudioSignal(frequency, frequency_band_width, modulation_frequency)
        # Set up ZMQ communications
        self.ControlMessages = paired_messenger(port=4186)
        time.sleep(1)
        self.ControlMessages.add_callback(self.command_parser)
        self.ControlMessages.sendMessage('initialization_successful')

    def command_parser(self, message):
        if message == 'play':
            self.sound.play(-1)
        elif message == 'stop':
            self.sound.stop()
        elif message == 'close':
            self.KeepListeningCommands = False
            self.ControlMessages.close()

    def runUntilCloseCommand(self):
        self.KeepListeningCommands = True
        while self.KeepListeningCommands:
            time.sleep(0.25)
        pygame.quit()

def main():
    controller = audioControl(int(sys.argv[1]), int(sys.argv[2]))
    controller.runUntilCloseCommand()

if __name__ == '__main__':
    main()
