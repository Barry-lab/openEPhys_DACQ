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

if len(sys.argv) != 2:
    print('Incorrect input.')
else:
    duration = float(sys.argv[1])
    pz.init()
    pz.setMotor(1,127)
    sleep(duration)
    pz.setMotor(1,0)
    pz.cleanup()
