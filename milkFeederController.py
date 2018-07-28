import argparse
import piconzero as pz
from time import sleep
from ZMQcomms import paired_messenger
from threading import Thread
import pygame
import os
from audioSignalGenerator import createAudioSignal

class piconzero_state_controller(object):
    '''
    This class is used to control the state of piconzero
    where multiple processes may use it interchangably
    '''

    def __init__(self):
        self.piconzero_initialized = False

    def ensure_piconzero_is_initialized(self):
        if not self.piconzero_initialized:
            pz.init()
            self.piconzero_initialized = True

    def ensure_piconzero_is_closed(self):
        if self.piconzero_initialized:
            pz.cleanup()
            self.piconzero_initialized = False

# Instantiate piconzero_state_controller to be used by any function in this script
pz_state_controller = piconzero_state_controller()


class pinchValveController(object):
    '''
    This controls the current driven to pinch valve via Picon Zero
    '''
    def __init__(self, motor_pin=1, max_power=127):
        pz_state_controller.ensure_piconzero_is_initialized()
        self.motor_pin = motor_pin
        self.max_power = max_power

    def openTimer(self, duration):
        '''
        Fully opens pinch valve for specified time in seconds
        '''
        pz.setMotor(self.motor_pin, self.max_power)
        sleep(duration)
        pz.setMotor(self.motor_pin, 0)

    def close(self):
        pz_state_controller.ensure_piconzero_is_closed()


class LEDcontroller(object):
    '''
    This controls an LED light via Picon Zero.
    '''
    def __init__(self, intensity=100, pin=0):
        '''
        Itensity (0 - 100) of LED brightness and its pin on Picon Zero can be specified.
        '''
        pz_state_controller.ensure_piconzero_is_initialized()
        self.intensity = int(intensity)
        self.pin = pin
        if self.intensity == 100:
            pz.setOutputConfig(self.pin, 0)
        else:
            pz.setOutputConfig(self.pin, 1)

    def turnLEDon(self):
        pz.setOutput(self.pin, self.intensity)

    def turnLEDoff(self):
        pz.setOutput(self.pin, 0)

    def close(self):
        pz_state_controller.ensure_piconzero_is_closed()


class audioSignalController(object):
    '''
    This class creates a signal with specific parameters and can play it from a local speaker.
    '''
    def __init__(self, frequency, frequency_band_width=1000, modulation_frequency=0):
        '''
        The sound frequency band center, width and modulating frequency can be defiend in Hz
        '''
        # Initialize pygame for playing sound
        os.system('amixer sset PCM,0 99%')
        pygame.mixer.pre_init(48000, -16, 2)
        pygame.mixer.init()
        pygame.init()
        # Create sound
        self.sound = createAudioSignal(frequency, frequency_band_width, modulation_frequency)

    def playAudioSignal(self):
        self.sound.play(-1)

    def stopAudioSignal(self):
        self.sound.stop()

    def playAudioSignalForSpecificDuration(self, duration):
        self.sound.play(duration)

    def close(self):
        pygame.mixer.quit()
        pygame.quit()


class Controller(object):
    '''
    This class controls the operation of the pinch valve, speaker and an LED on the milkFeeder.
    '''

    def __init__(self, pinchValve=False, audioSignalParams=None, lightSignalIntensity=None, init_feedback=False):
        '''
        Initializes milkFeeder Controller with specified parameters.
        pinchValve - bool - makes pinch valve release available
        audioSignalParams - tuple(signal_frequency,frequency_band_width,modulation_frequency) -
                            Initializes audioSignal controller with specified parameters.
        lightSignalIntensity - int - Initializes lightSignal controller with specified intensity (0 - 100).
        init_feedback - bool - Sends ZMQ message to localhost if initialization successful: init_successful
                        When using this, ensure a ZMQ device is paired to the IP of the device running this script.
        '''
        self.parse_arguments(pinchValve, audioSignalParams, lightSignalIntensity, init_feedback)
        # Initialize ZMQcomms in a separate thread
        T_initialize_ZMQcomms = Thread(target=self.initialize_ZMQcomms)
        T_initialize_ZMQcomms.start()
        # Initialize all requested features
        if self.pinchValve:
            self.pinchValveController = pinchValveController()
        if not (self.lightSignalIntensity is None) and self.lightSignalIntensity > 0:
            self.LEDcontroller = LEDcontroller(lightSignalIntensity)
        if not (self.audioSignalParams is None):
            self.audioSignalController = audioSignalController(*audioSignalParams)
        # Ensure ZMQ comms have been initialized
        T_initialize_ZMQcomms.join()
        # Send feedback text if requested
        if self.init_feedback:
            self.ZMQmessenger.sendMessage('init_successful')

    def parse_arguments(self, pinchValve, audioSignalParams, lightSignalIntensity, init_feedback):
        '''
        Parses input arguments into class attributes in correct type
        '''
        self.pinchValve = bool(pinchValve)
        if not (audioSignalParams is None):
            self.audioSignalParams = audioSignalParams
        else:
            self.audioSignalParams = None
        if not (lightSignalIntensity is None):
            self.lightSignalIntensity = int(lightSignalIntensity)
        else:
            self.lightSignalIntensity = None
        self.init_feedback = bool(init_feedback)

    def initialize_ZMQcomms(self):
        '''
        Allows initializing ZMQ comms in a separate thread
        '''
        self.ZMQmessenger = paired_messenger(port=4186)
        self.ZMQmessenger.add_callback(self.command_parser)
        sleep(1) # This ensures all ZMQ protocols have been properly initated before finishing this process

    def command_parser(self, message):
        '''
        Parses incoming ZMQ message for function name, input arguments and calls that function.
        '''
        method_name = message.split(' ')[0]
        args = message.split(' ')[1:]
        getattr(self, method_name)(*args)

    def releaseMilk(self, duration, feedback=False):
        '''
        Fully opens pinch valve for specified time in seconds. Input duration is converted to float.
        '''
        if type(duration) != float:
            duration = float(duration)
        if self.pinchValve:
            self.pinchValveController.openTimer(duration)
        if feedback:
            self.ZMQmessenger.sendMessage('releaseMilk successful')

    def startPlayingAudioSignal(self):
        self.audioSignalController.playAudioSignal()

    def stopPlayingAudioSignal(self):
        self.audioSignalController.stopAudioSignal()

    def startLightSignal(self):
        self.LEDcontroller.turnLEDon()

    def stopLightSignal(self):
        self.LEDcontroller.turnLEDoff()

    def startAllSignals(self):
        if hasattr(self, 'LEDcontroller'):
            self.startPlayingAudioSignal()
        if hasattr(self, 'audioSignalController'):
            self.startLightSignal()

    def stopAllSignals(self):
        if hasattr(self, 'LEDcontroller'):
            self.stopPlayingAudioSignal()
        if hasattr(self, 'audioSignalController'):
            self.stopLightSignal()

    def keepProcessRunning(self):
        '''
        This method allows any process to wait until close method has been called for this class
        '''
        self.closeCommandReceived = False
        while not self.closeCommandReceived:
            try:
                time.sleep(0.1)
            except KeyboardInterrupt:
                self.close()

    def close(self):
        '''
        Closes all initiated processes gracefully.
        '''
        self.closeCommandReceived = True
        self.ZMQmessenger.close()
        if hasattr(self, 'pinchValveController'):
            self.pinchValveController.close()
        if hasattr(self, 'LEDcontroller'):
            self.LEDcontroller.close()
        if hasattr(self, 'audioSignalController'):
            self.audioSignalController.close()


if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Running this script initates Controller class.')
    parser.add_argument('--pinchValve', action='store_true', 
                        help='Initializes pinch valve.')
    parser.add_argument('--audioSignal', type=int, nargs = 3, 
                        help='Initializes audioSignal controller with specified parameters\n' + \
                        'signal_frequency frequency_band_width modulation_frequency.')
    parser.add_argument('--lightSignal', type=int, nargs = 1, 
                        help='Initializes lightSignal controller with specified intensity (0 - 100).')
    parser.add_argument('--init_feedback', action='store_true', 
                        help='Sends ZMQ message to localhost if initialization successful: init_successful\n' + \
                        'When using this, ensure a ZMQ device is paired to the IP of the device running this script.')
    parser.add_argument('--openValve', type=float, nargs = 1, 
                        help='Makes device opens pinch valve for specified duration (s) and does not activate Controller class.')
    args = parser.parse_args()
    # If releasePellet command given skip everything and just release the number of pellets specified
    if args.openValve:
        pvc = pinchValveController()
        pvc.openTimer(float(args.openValve[0]))
        pvc.close()
    else:
        # Load input arguments
        if args.lightSignal:
            pinchValve = True
        else:
            pinchValve = False
        if args.audioSignal:
            audioSignalParams = (args.audioSignal[0], args.audioSignal[1], args.audioSignal[2])
        else:
            audioSignalParams = None
        if args.lightSignal:
            lightSignalIntensity = int(args.lightSignal[0])
        else:
            lightSignalIntensity = None
        if args.init_feedback:
            init_feedback = True
        else:
            init_feedback = False
        # Initialize Controller
        active_Controller = Controller(pinchValve=pinchValve, audioSignalParams=audioSignalParams, 
                                       lightSignalIntensity=lightSignalIntensity, init_feedback=init_feedback)
        # Start an endless loop that can be cancelled with close method or ctrl+c
        active_Controller.keepProcessRunning()
