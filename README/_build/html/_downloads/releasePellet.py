# This script operates a servo to drop a pellet or multiple.
# Use input argument to specify the number of pellets
# e.g. depositPellet.py 4
# which would drop 4 pellets. No argument means 1 pellet.

# To adjust the Loading and Release positions of the servo,
# edit script variables LoadingAngle and ReleaseAngle

from RPi import GPIO
from time import sleep
import sys

class servo_controller(object):
    # This class activates servo SG-5010 with PWM signal
    # setAngle function allows setting the servo to a specific angle    
    def __init__(self,ttlPin=12):
        frequency = 50.0 # PWM cycle frequency
        self.ttlPin = ttlPin # Specify the TTL pin to be used for signal on RPi board
        GPIO.setmode(GPIO.BOARD) # Use board numbering for TTL pin
        GPIO.setup(self.ttlPin, GPIO.OUT)
        # Initialize PWM signalling
        self.pwm = GPIO.PWM(self.ttlPin, frequency)
        GPIO.output(self.ttlPin, False) # Keep signal off until first movement command to avoid jitter
        self.pwm.start(0)

    def setAngle(self, angle):
        # This sets to signal servo to a specific position
        duty = angle / 19.5 + 2.0
        self.pwm.ChangeDutyCycle(duty)
        # This ensures the TTL pin is active and sending signal
        GPIO.output(self.ttlPin, True)
    
    def close(self):
        # Stops sending the signal to the servo
        GPIO.output(self.ttlPin, False)
        sleep(0.1)
        self.pwm.stop()
        GPIO.cleanup(self.ttlPin)

# Set these angles to work with your feeder
LoadingAngle = 30
ReleaseAngle = 90
# Get input argument for number of pellets
if len(sys.argv) < 2:
    n_pellets = 1
else:
    n_pellets = int(sys.argv[1])
# Initialise signalling to servo
sc = servo_controller()
# Drop as many pellets as requested
for n_pellet  in range(n_pellets):
    # Moves the servo to the loading position
    sc.setAngle(LoadingAngle)
    sleep(0.4) # Allows time for movement
    # Jitter in the loading position to make sure a pellet falls in
    for jitter in [-15, 15, -15, 15, -15, 15]:
        sc.setAngle(LoadingAngle + jitter)
        sleep(0.1)
    # Move the servo to the pellet releasing position
    sc.setAngle(ReleaseAngle)
    sleep(0.4) # Allow time for movement
# Stop signalling to the servo
sc.close()
