from ZMQcomms import paired_messenger
import sys
import piconzero as pz
import time

class lightControl(object):
    '''
    This controls an LED light in response to commands received over ZMQcomms.
    NOTE! Using piconzero in other processes (e.g. openPinchValve.py) shuts off the light.
            To avoid this, all such processes should be integrated into a single script.
    '''
    def __init__(self, intensity=100, pin=0):
        self.intensity = int(intensity)
        self.pin = pin
        # Set up ZMQ communications
        self.ControlMessages = paired_messenger(port=4187)
        time.sleep(1)
        self.ControlMessages.add_callback(self.command_parser)
        self.KeepListeningCommands = True

    def lightStart(self):
        '''
        Initializes piconzero and uses it to direct power to an LED
        '''
        pz.init()
        if self.intensity == 100:
            pz.setOutputConfig(self.pin,0)
        else:
            pz.setOutputConfig(self.pin,1)
        pz.setOutput(self.pin, self.intensity)

    def lightStop(self):
        '''
        Turns off the LED
        '''
        pz.setOutput(self.pin, 0)

    def command_parser(self, message):
        if message == 'start':
            self.lightStart()
        elif message == 'stop':
            self.lightStop()
        elif message == 'close':
            self.KeepListeningCommands = False
            self.lightStop()
            self.ControlMessages.close()
            pz.cleanup()

    def runUntilCloseCommand(self):
        '''
        This ensures the class exists until self.KeepListeningCommands is set to False
        '''
        self.KeepListeningCommands = True
        while self.KeepListeningCommands:
            time.sleep(0.25)

def main():
    if len(sys.argv) == 2:
        controller = lightControl(int(sys.argv[1]))
    else:
        controller = lightControl()
    controller.runUntilCloseCommand()

if __name__ == '__main__':
    main()
