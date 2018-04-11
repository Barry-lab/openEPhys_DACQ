from pygame import init as pygame_init
from pygame import sndarray as pygame_sndarray
from pygame import mixer
import numpy as np
import time
from ZMQcomms import paired_messenger
import sys

def createSineWaveSound(frequency, modulation_frequency=0):
    # Calling this function must be preceded by calling the following lines
    # for pygame to recognise the mono sound
    # pygame.mixer.pre_init(48000, -16, 1) # here 48000 needs to match the sampleRate in this function
    # pygame.init()
    sampleRate = 48000 # Must be the same as in the line 
    peak = 4096 # 4096 : the peak ; volume ; loudness
    arr = np.array([peak * np.sin(2.0 * np.pi * frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
    if modulation_frequency > 0:
        # Modulate pure tone at specified frequency
        modulation_frequency = int(modulation_frequency)
        marr = np.array([np.sin(2.0 * np.pi * modulation_frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.float64)
        marr = marr - min(marr)
        marr = marr / max(marr)
        arr = np.int16(arr.astype(np.float64) * marr)
    arr = np.repeat(arr[:,None],2,axis=1)
    sound = pygame_sndarray.make_sound(arr)

    # Use sound.play(-1) to start sound. (-1) means it is in infinite loop
    # Use sound.stop() to stop the sound

    return sound

class audioControl(object):
    def __init__(self, frequency, modulation_frequency=0):
        # Initialize pygame for playing sound
        mixer.pre_init(48000, -16, 2)
        pygame_init()
        # Get sound
        self.sound = createSineWaveSound(frequency, modulation_frequency)
        # Set up ZMQ communications
        self.ControlMessages = paired_messenger(port=4186)
        self.ControlMessages.add_callback(self.command_parser)
        # Make audioControl responsive to commands
        self.KeepListeningCommands = True
        self.run()

    def command_parser(self, message):
        if message == 'play':
            self.sound.play(-1)
        elif message == 'stop':
            self.sound.stop()
        elif message == 'close':
            self.KeepListeningCommands = False
            self.ControlMessages.close()

    def run(self):
        while self.KeepListeningCommands:
            time.sleep(0.25)

def main():
    controller = audioControl(int(sys.argv[1]), int(sys.argv[2]))

if __name__ == '__main__':
    main()
