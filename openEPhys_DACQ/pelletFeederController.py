from RPi import GPIO
from time import time, sleep
from copy import copy
from ZMQcomms import paired_messenger
import argparse

GPIO.setmode(GPIO.BOARD) # Use board numbering for TTL pin

class detect_pellet(object):
    '''
    This class signals when IR break beam has been interrupted.
    It can call a callback function and/or pause process until pellet detection.
    At pellet detection, the class closes automaticall, unless single_detection is set to False.
    '''

    def __init__(self, callback=None, single_detection=True, beamPin=15):
        '''
        callback - optional - function to call when IR break beam has been interrupted
        single_detection - bool - True by default, 
                           which closes detect_pellet class after first time IR beam is interrupted.
        beamPin - the GPIO.BOARD pin numbering
        '''
        self.callback = callback
        self.single_detection = single_detection
        self.beamPin = beamPin
        self.pellet_detected = False
        GPIO.setup(self.beamPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.add_event_detect(self.beamPin, GPIO.FALLING, callback=self.beam_interrupted)

    def beam_interrupted(self, channel):
        '''
        This method is called when IR break beam is interrupted.
        It closes the detect_pellet bindings, if single_dection=True.
        It changes the attribute pellet_detected to True.
        It calls the callback function in __init__ input arguments.
        '''
        if self.single_detection:
            self.close()
        self.pellet_detected = True
        if not (self.callback is None):
            self.callback()

    def wait_for_pellet_detection(self, max_wait_time=1.0):
        '''
        This method waits for pellet detection for the maximum time stated.
        It exits as soon as pellet is detected or timer runs out, closing detect_pellet class.
        It returns True or False, depending if pellet was detected.
        '''
        starting_time = time()
        while not self.pellet_detected and (time() - starting_time) < max_wait_time:
            sleep(0.05)
        self.close()

        return self.pellet_detected

    def close(self):
        GPIO.cleanup(self.beamPin)


class servo_controller(object):
    '''
    This class controls servo SG-5010 with PWM signal
    '''    
    def __init__(self, servoPin=12, startingAngle=90):
        self.servoPin = servoPin # Specify the TTL pin to be used for signal on RPi board
        self.PWM_cycle_frequency = 50.0 # in Hz
        self.time_for_180_turn = 1.0 # in seconds
        GPIO.setup(self.servoPin, GPIO.OUT)
        # Initialize PWM signalling
        self.pwm = GPIO.PWM(self.servoPin, self.PWM_cycle_frequency)
        GPIO.output(self.servoPin, True)
        # Set motor to starting angle position
        self.current_angle = startingAngle
        self.pwm.start(self.compute_duty_cycle_for_angle(self.current_angle))
        sleep(self.time_for_180_turn / 2.0)

    def compute_duty_cycle_for_angle(self, angle):
        return angle / 19.5 + 2.0

    def setAngle(self, angle):
        '''
        Instructs the stepper motor to move to specific angle by changing the duty cycle of the PWM signal.
        '''
        self.pwm.ChangeDutyCycle(self.compute_duty_cycle_for_angle(angle))
        self.current_angle = angle

    def computeMoveTimeToAngle(self, angle):
        '''
        Returns an estimated time of seconds required to turn to given angle from current angle.
        '''
        angular_difference = abs(self.current_angle - angle)
        proportion_of_180_turn = float(angular_difference) / 180.0

        return proportion_of_180_turn * self.time_for_180_turn

    def moveToAngle(self, angle):
        '''
        Moves stepper motor to specified angle.
        '''
        time_required = self.computeMoveTimeToAngle(angle)
        self.setAngle(angle)
        sleep(time_required)

    def wiggle(self, angular_range=15, n_repeats=1):
        '''
        Wiggles stepper motor back and forth the amount of degrees specified in angular_range and n_repeats times.
        '''
        original_angle = copy(self.current_angle)
        time_required = self.computeMoveTimeToAngle(original_angle + angular_range)
        for n in range(n_repeats):
            self.setAngle(original_angle + angular_range)
            sleep(time_required)
            self.setAngle(original_angle - angular_range)
            sleep(2 * time_required)
            self.setAngle(original_angle)
            sleep(time_required)
    
    def close(self):
        self.pwm.stop()
        sleep(0.1) # Ensures PWM stops properly before signal is terminated.
        GPIO.output(self.servoPin, False)
        GPIO.cleanup(self.servoPin)


def release_pellet(LoadingAngle=30, ReleaseAngle=90, wiggle_angular_range=15, wiggle_repeats=4):
    '''
    This function attempts to release a single pellet with stepper motor
    '''
    sc = servo_controller()
    sc.moveToAngle(LoadingAngle)
    sc.wiggle(wiggle_angular_range, wiggle_repeats)
    sc.moveToAngle(ReleaseAngle)
    sc.close()


def release_exactly_one_pellet(LoadingAngle=30, ReleaseAngle=90, wiggle_angular_range=15, wiggle_repeats=4, max_attempts=10):
    '''
    This function attempts to release a pellet until one is detected by the IR break beam.
    It waits until completed and returns either True or False depending if successful.
    '''

    max_wait_time_for_pellet=1.0
    for n in range(max_attempts):
        dp = detect_pellet()
        release_pellet(LoadingAngle, ReleaseAngle, wiggle_angular_range, wiggle_repeats)
        pellet_released = dp.wait_for_pellet_detection(max_wait_time=max_wait_time_for_pellet)
        if pellet_released:
            break

    return pellet_released


class Controller(object):
    '''
    This class controls the operation of the stepper motor and IR break beam 
    for releasing controlled number of pellets.
    '''

    def __init__(self, init_feedback=False):
        '''
        init_feedback - bool - Sends ZMQ message to localhost if initialization successful: init_successful
                        When using this, ensure a ZMQ device is paired to the IP of the device running this script.
        '''
        # Initialize ZMQcomms
        self.initialize_ZMQcomms()
        # Send feedback text if requested
        if init_feedback:
            self.ZMQmessenger.sendMessage('init_successful')

    def initialize_ZMQcomms(self):
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

    def release_pellets(self, quantity=1, feedback=False):
        '''
        Releases the number of pellets requested. If feedback requested, sends bool in str format after each pellet.
        If an attempt of a single pellet fails in max_attempts, the rest are aborted.
        '''
        # Parse input arguments, converting from string format if necessary
        if type(quantity) != int:
            quantity = int(quantity)
        if type(feedback) != bool:
            if type(feedback) == str and feedback == 'True':
                feedback = True
            else:
                feedback = False
        # Attempt releasing all required pellets
        for n_pellet in range(quantity):
            self.ZMQmessenger.sendMessage('release_pellets in_progress')
            successful_release = release_exactly_one_pellet(LoadingAngle=30, ReleaseAngle=90, 
                                                            wiggle_angular_range=15, wiggle_repeats=4, 
                                                            max_attempts=10)
            # If release has failed, the rest of the pellets are aborted.
            if not successful_release:
                break
        # Return feedback over ZMQ if requested.
        if feedback:
            if successful_release:
                self.ZMQmessenger.sendMessage('release_pellets successful')
            else:
                self.ZMQmessenger.sendMessage('release_pellets failed')

    def keepProcessRunning(self):
        '''
        This method allows any process to wait until close method has been called for this class.
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
        self.ZMQmessenger.close()


if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Running this script initates Controller class.')
    parser.add_argument('--init_feedback', action='store_true', 
                        help='Sends ZMQ message to localhost if initialization successful: init_successful\n' + \
                        'When using this, ensure a ZMQ device is paired to the IP of the device running this script.')
    parser.add_argument('--releasePellet', type=int, nargs = 1, 
                        help='Makes device release a specified number of pellets and does not activate Controller class.')
    args = parser.parse_args()
    # If releasePellet command given skip everything and just release the number of pellets specified
    if args.releasePellet:
        for n_pellet in range(int(args.releasePellet[0])):
            successful_release = release_exactly_one_pellet(LoadingAngle=30, ReleaseAngle=90, 
                                                            wiggle_angular_range=15, wiggle_repeats=4, 
                                                            max_attempts=10)
            if not successful_release:
                break
    else:
        # Load input arguments
        if args.init_feedback:
            init_feedback = True
        else:
            init_feedback = False
        # Initialize Controller
        active_Controller = Controller(init_feedback=init_feedback)
        # Start an endless loop that can be cancelled with close method or ctrl+c
        active_Controller.keepProcessRunning()
