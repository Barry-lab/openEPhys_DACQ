import argparse
import piconzero as pz
from time import sleep
from ZMQcomms import paired_messenger
from threading import Thread
import os
from audioSignalGenerator import createAudioSignal

import contextlib
with contextlib.redirect_stdout(None):
    import pygame

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
    def __init__(self):
        '''
        The sound frequency band center, width and modulating frequency can be defiend in Hz
        '''
        # Initialize pygame for playing sound
        os.system('amixer sset PCM,0 99%')
        pygame.mixer.pre_init(48000, -16, 2)
        pygame.mixer.init()
        pygame.init()
        # Create sound
        self.sound = {}

    def createAudioSignal(self, name, frequency, frequency_band_width, modulation_frequency, loudness='default'):
        if loudness == 'loud':
            self.sound[name] = createAudioSignal(frequency, frequency_band_width, modulation_frequency, 200000)
        else:
            self.sound[name] = createAudioSignal(frequency, frequency_band_width, modulation_frequency)

    def playAudioSignal(self, name):
        self.sound[name].play(-1)

    def stopAudioSignal(self, name):
        self.sound[name].stop()

    def playAudioSignalForSpecificDuration(self, name, duration):
        self.sound[name].play(-1, maxtime=int(duration * 1000))

    def startAllSounds(self):
        for key in self.sound.keys():
            self.sound[key].play(-1)

    def stopAllSounds(self):
        for key in self.sound.keys():
            self.sound[key].stop()

    def close(self):
        self.stopAllSounds()
        pygame.mixer.quit()
        pygame.quit()


class Controller(object):
    '''
    This class controls the operation of the pinch valve, speaker and an LED on the milkFeeder.
    '''

    def __init__(self, pinchValve=False, trialAudioSignalParams=None, lightSignalIntensity=None, 
                 lightSignalPins=[0], negativeAudioSignal=None, init_feedback=False):
        '''
        pinchValve             - bool - makes pinch valve release available
        trialAudioSignalParams - tuple(signal_frequency,frequency_band_width,modulation_frequency) -
                                 Initializes audioSignal controller with specified parameters.
        lightSignalIntensity   - int - Initializes lightSignal controller with specified intensity (0 - 100).
        lightSignalPins        - list of int - number for picon zero pins to use for LED signalling.
        negativeAudioSignal    - float - allows playing a full-range white noise for specified duration if > 0
        init_feedback          - bool - Sends ZMQ message to localhost if initialization successful: init_successful
                                 When using this, ensure a ZMQ device is paired to the IP of the device running this script.
        '''
        self.pinchValve = bool(pinchValve)
        # Initialize ZMQcomms in a separate thread
        T_initialize_ZMQcomms = Thread(target=self.initialize_ZMQcomms)
        T_initialize_ZMQcomms.start()
        # Initialize all requested features
        if self.pinchValve:
            self.pinchValveController = pinchValveController()
        if not (lightSignalIntensity is None) and lightSignalIntensity > 0:
            self.LEDcontrollers = []
            for pin in lightSignalPins:
                self.LEDcontrollers.append(LEDcontroller(lightSignalIntensity, pin))
        if not (trialAudioSignalParams is None) or not (negativeAudioSignal is None):
            self.audioSignalController = audioSignalController()
            if not (trialAudioSignalParams is None):
                self.audioSignalController.createAudioSignal('trialAudioSignal', *trialAudioSignalParams)
            if not (negativeAudioSignal is None):
                self.audioSignalController.createAudioSignal('negativeAudioSignal', 12000, 10000, 0, 'loud')
                self.negativeAudioSignalDuration = negativeAudioSignal
        # Ensure ZMQ comms have been initialized
        T_initialize_ZMQcomms.join()
        # Send feedback text if requested
        if init_feedback:
            self.ZMQmessenger.sendMessage('init_successful')

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

    def startLightSignal(self):
        if hasattr(self, 'LEDcontrollers'):
            for LEDcontroller in self.LEDcontrollers:
                LEDcontroller.turnLEDon()

    def stopLightSignal(self):
        if hasattr(self, 'LEDcontrollers'):
            for LEDcontroller in self.LEDcontrollers:
                LEDcontroller.turnLEDoff()

    def startTrialAudioSignal(self):
        self.audioSignalController.playAudioSignal('trialAudioSignal')

    def stopTrialAudioSignal(self):
        self.audioSignalController.stopAudioSignal('trialAudioSignal')

    def startNegativeAudioSignal(self):
        self.audioSignalController.playAudioSignal('negativeAudioSignal')

    def playNegativeAudioSignal(self):
        self.audioSignalController.playAudioSignalForSpecificDuration('negativeAudioSignal', self.negativeAudioSignalDuration)

    def stopNegativeAudioSignal(self):
        self.audioSignalController.stopAudioSignal('negativeAudioSignal')

    def startAllSignals(self):
        if hasattr(self, 'LEDcontrollers'):
            self.startLightSignal()
        if hasattr(self, 'audioSignalController'):
            self.audioSignalController.startAllSounds()

    def stopAllSignals(self):
        if hasattr(self, 'LEDcontrollers'):
            self.stopLightSignal()
        if hasattr(self, 'audioSignalController'):
            self.audioSignalController.stopAllSounds()

    def keepProcessRunning(self):
        '''
        This method allows any process to wait until close method has been called for this class
        '''
        self.closeCommandReceived = False
        while not self.closeCommandReceived:
            try:
                sleep(0.1)
            except KeyboardInterrupt:
                self.close()

    def close(self):
        '''
        Closes all initiated processes gracefully.
        '''
        self.closeCommandReceived = True
        self.stopAllSignals()
        self.ZMQmessenger.close()
        if hasattr(self, 'pinchValveController'):
            self.pinchValveController.close()
        if hasattr(self, 'LEDcontrollers'):
            for LEDcontroller in self.LEDcontrollers:
                LEDcontroller.close()
        if hasattr(self, 'audioSignalController'):
            self.audioSignalController.close()


if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Running this script initates Controller class.')
    parser.add_argument('--pinchValve', action='store_true', 
                        help='Initializes pinch valve.')
    parser.add_argument('--lightSignalIntensity', type=int, nargs = 1, 
                        help='Initializes lightSignal controller with specified intensity (0 - 100).')
    parser.add_argument('--lightSignalPins', type=int, nargs='*', 
                        help='Specifies which pins to use for light signal.')
    parser.add_argument('--trialAudioSignal', type=int, nargs = 3, 
                        help='Initializes audioSignalController with specified parameters\n' + \
                        'signal_frequency frequency_band_width modulation_frequency.')
    parser.add_argument('--negativeAudioSignal', type=float, nargs = 1, 
                        help='Initializes audioSignalController also for wide-band noise burst of specified duration(s).')
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
        if args.lightSignalIntensity:
            pinchValve = True
        else:
            pinchValve = False
        if args.lightSignalIntensity:
            lightSignalIntensity = int(args.lightSignalIntensity[0])
        else:
            lightSignalIntensity = None
        if args.lightSignalPins:
            lightSignalPins = args.lightSignalPins
        else:
            lightSignalPins = [1]
        if args.trialAudioSignal:
            trialAudioSignalParams = (args.trialAudioSignal[0], args.trialAudioSignal[1], args.trialAudioSignal[2])
        else:
            trialAudioSignalParams = None
        if args.negativeAudioSignal:
            negativeAudioSignal = float(args.negativeAudioSignal[0])
            if negativeAudioSignal == 0:
                negativeAudioSignal = None
        else:
            negativeAudioSignal = None
        if args.init_feedback:
            init_feedback = True
        else:
            init_feedback = False
        # Initialize Controller
        active_Controller = Controller(pinchValve=pinchValve, trialAudioSignalParams=trialAudioSignalParams, 
                                       lightSignalIntensity=lightSignalIntensity, lightSignalPins=lightSignalPins, 
                                       negativeAudioSignal=negativeAudioSignal, init_feedback=init_feedback)
        # Start an endless loop that can be cancelled with close method or ctrl+c
        active_Controller.keepProcessRunning()
