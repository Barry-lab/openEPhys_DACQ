# This script operates Picon Zero attached to Raspberry Pi Zero
# to send a constant current which is intended to open a pinch valve.

# Call the function by specifying in the argument the duration
# of the valve opening period.

# Example: python openPinchValve.py 2.5
# This would open the valve for two and a half seconds.

# By Sander Tanni, UCL, July 2017

import sys
import piconzero as pz
from time import sleep
from ZMQcomms import paired_messenger
from threading import Thread

publisher = None

def initialize_publisher():
    global publisher
    publisher = paired_messenger(port=1232)
    sleep(1)

if len(sys.argv) < 2:
    print('Incorrect input.')
else:
    # Prepare feedback messaging connection in a separate thread
    if len(sys.argv) == 3 and str(sys.argv[2]) == 'feedback':
        T_initialize_publisher = Thread(target=initialize_publisher)
        T_initialize_publisher.start()
    # Perform pinch valve operation
    duration = float(sys.argv[1])
    pz.init()
    pz.setMotor(1,127)
    sleep(duration)
    pz.setMotor(1,0)
    pz.cleanup()
    # Send message that action was completed
    if len(sys.argv) == 3 and str(sys.argv[2]) == 'feedback':
        T_initialize_publisher.join() # Make sure publisher initialization thread is finished
        publisher.sendMessage('successful')
        publisher.close()
