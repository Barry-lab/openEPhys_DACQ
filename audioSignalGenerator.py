
import numpy as np
from random import gauss
from scipy.signal import butter, lfilter

import contextlib
with contextlib.redirect_stdout(None):
    import pygame

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(signal_in, sampling_rate, highpass_frequency, lowpass_frequency, filt_order):
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

def createAudioSignal(frequency, frequency_band_width, modulation_frequency, peak=4096):
    '''
    Creates a sound signal with specified parameters (in Hz), 
    that can be played in scope where pygame is initialized as:
        pygame.mixer.pre_init(48000, -16, 2)
        pygame.mixer.init()
        pygame.init()
    Returns pygame sound object that can be used with following commands:
        sound.play(n) play for n seconds
        sound.play(-1) play indefinitely
        sound.stop() stop playing the sound

        peak - int - default is 4096. Changes the amplitude of the raw sound data
    '''
    sampleRate = 48000 # Must be the same as in the line 
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

    return sound
