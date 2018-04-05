# This script operates a servo to drop a pellet or multiple.
# Use input argument to specify the number of pellets
# e.g. depositPellet.py 4
# which would drop 4 pellets. No argument means 1 pellet.

# To adjust the Loading and Release positions of the servo,
# edit script variables LoadingAngle and ReleaseAngle

from RPi import GPIO
import time
import sys
import threading
from ZMQcomms import sendMessagesPAIR
import os

GPIO.setmode(GPIO.BOARD) # Use board numbering for TTL pin

class detect_pellet(object):
    # This class keeps checking if a pellet has been released
    # as soon as it is instantiated
    # Use the following to check if pellet was detected
    # pellet_checker = detect_pellet()
    # with pellet_checker.pelletDetectedLock:
    #     print(pellet_checker.pelletDetected)
    # Use pellet_checker.close() to stop

    def __init__(self,beamPin=15):
        self.beamPin = beamPin
        GPIO.setup(self.beamPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.pelletDetected = False
        self.pelletDetectedLock = threading.Lock()
        threading.Thread(target=self.wait_for_pellet).start()

    def wait_for_pellet(self):
        while not self.pelletDetected:
            val = GPIO.input(self.beamPin)
            if val == 0:
                with self.pelletDetectedLock:
                    self.pelletDetected = True

    def close(self):
        with self.pelletDetectedLock:
            self.pelletDetected = True
        GPIO.cleanup(self.beamPin)

class servo_controller(object):
    # This class activates servo SG-5010 with PWM signal
    # setAngle function allows setting the servo to a specific angle    
    def __init__(self,servoPin=12):
        frequency = 50.0 # PWM cycle frequency
        self.servoPin = servoPin # Specify the TTL pin to be used for signal on RPi board
        GPIO.setup(self.servoPin, GPIO.OUT)
        # Initialize PWM signalling
        self.pwm = GPIO.PWM(self.servoPin, frequency)
        GPIO.output(self.servoPin, False) # Keep signal off until first movement command to avoid jitter
        self.pwm.start(0)

    def setAngle(self, angle):
        # This sets to signal servo to a specific position
        duty = angle / 19.5 + 2.0
        self.pwm.ChangeDutyCycle(duty)
        # This ensures the TTL pin is active and sending signal
        GPIO.output(self.servoPin, True)
    
    def close(self):
        # Stops sending the signal to the servo
        GPIO.output(self.servoPin, False)
        time.sleep(0.1)
        self.pwm.stop()
        GPIO.cleanup(self.servoPin)

def update_pelletUseCount(n_pellets):
    '''
    Rewrites local file that counts the number of pellets used.
    If no file 'pelletUseCount' exists, a file is created and count resets to zero.
    '''
    pelletUseCountFileName = 'pelletUseCount'
    if os.path.isfile(pelletUseCountFileName):
        with open(pelletUseCountFileName, 'r') as file:
            pelletUseCount = int(file.read())
    else:
        pelletUseCount = 0
    pelletUseCount += n_pellets
    with open(pelletUseCountFileName, 'w') as file:
        file.write(str(pelletUseCount))

# Set these angles to work with your feeder
LoadingAngle = 30
ReleaseAngle = 90
# Get input argument for number of pellets
if len(sys.argv) < 2:
    print('Incorrect input.')
else:
    n_pellets = int(sys.argv[1])
    # Initialise messaging pipe
    if len(sys.argv) == 3 and str(sys.argv[2]) == 'feedback':
        publisher = sendMessagesPAIR()
    # Initialise signalling to servo
    sc = servo_controller()
    # Drop as many pellets as requested
    for n_pellet  in range(n_pellets):
        release_successful = False
        while not release_successful:
            pellet_checker = detect_pellet()
            # Moves the servo to the loading position
            sc.setAngle(LoadingAngle)
            time.sleep(0.4) # Allows time for movement
            # Jitter in the loading position to make sure a pellet falls in
            for jitter in [-15, 15, -15, 15, -15, 15]:
                sc.setAngle(LoadingAngle + jitter)
                time.sleep(0.1)
            # Move the servo to the pellet releasing position
            sc.setAngle(ReleaseAngle)
            time.sleep(0.4) # Allow time for movement
            # Check if pellet was dropped.
            waiting_started = time.time()
            max_wait_time = 1
            while not release_successful and time.time() - waiting_started < max_wait_time:
                time.sleep(0.05)
                with pellet_checker.pelletDetectedLock:
                    release_successful = pellet_checker.pelletDetected
            pellet_checker.close()
            # Send message of outcome
            if len(sys.argv) == 3 and str(sys.argv[2]) == 'feedback':
                if release_successful:
                    publisher.sendMessage('successful')
                else:
                    publisher.sendMessage('failed')
    # Stop signalling to the servo
    sc.close()
    # Update pelletUseCount file
    update_pelletUseCount(n_pellets)
    # Close messaging pipe
    if len(sys.argv) == 3 and str(sys.argv[2]) == 'feedback':
        publisher.close()
