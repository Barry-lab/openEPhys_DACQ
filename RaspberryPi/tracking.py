### This script finds the location of the peak luminance in images captured by RPi camera
### and saves as well as sends this info across network with ZeroMQ along with timestamps.

# By Sander Tanni, May 2017, UCL

import numpy as np
import picamera
import picamera.array
import time
import cv2
import csv
import zmq
import json
from RPi import GPIO
import os
import cPickle as pickle
from scipy.spatial.distance import euclidean

# Get assigned RPi number
RPi_number = int(open('RPiNumber','r').read().splitlines()[0]) # The number to identify logs and messages from this RPi

# Get OpenEphys configuration details for this RPi
with open('RPiSettings.p','rb') as file:
    RPiSettings = pickle.load(file)
if RPiSettings['LEDmode'] == 'double':
    doubleLED = True
elif RPiSettings['LEDmode'] == 'single':
    doubleLED = False
save_frames = RPiSettings['save_frames']
camera_iso = RPiSettings['camera_iso']
shutter_speed = RPiSettings['shutter_speed']
exposure_setting = RPiSettings['exposure_setting']
smoothradius = RPiSettings['smoothing_radius']
imageres = RPiSettings['resolution']
CentralIP = RPiSettings['centralIP']
PosPort = RPiSettings['pos_port']
StopPort = RPiSettings['stop_port']
RPiIP = RPiSettings['RPiIP'][RPi_number]
LED_max_distance = RPiSettings['LED_max_distance']

# Set up ZeroMQ
# Set IP addresses and ports for sending position data and receiving stop command
posIP = 'tcp://' + RPiIP + ':' + PosPort
stopIP = 'tcp://' + CentralIP + ':' + StopPort
# For sending position data
contextpub = zmq.Context()
sockpub = contextpub.socket(zmq.PUB)
sockpub.bind(posIP)
# For receiving start/stop commands
contextrec = zmq.Context()
sockrec = contextrec.socket(zmq.SUB)
sockrec.setsockopt(zmq.SUBSCRIBE, '')
sockrec.connect(stopIP)
time.sleep(0.1) # Give time to establish sockets for ZeroMQ

# Set up GPIO system for TTL pulse signalling
ttlPin = 11 # The GPIO pin used for output on RPi (check https://pinout.xyz/pinout/pin11_gpio17#)
GPIO.setmode(GPIO.BOARD) # Use the mapping based on physical bin numbering
GPIO.setup(ttlPin, GPIO.OUT, initial=GPIO.LOW)
GPIO.output(ttlPin, False)

# Prepare log file first line. X1 and Y1 are for the dimmer LED on the animal's head, if two are used.
linedata = ['RPinumber', 'TTLtime', 'Frametime', 'X1', 'Y1', 'X2', 'Y2', 'Luminance_1', 'Luminance_2'] # Set header for the file
if os.path.isfile('./logfile.csv'):
    os.remove('./logfile.csv')
csvfile = open('logfile.csv', 'a')
logwriter = csv.writer(csvfile)
logwriter.writerow(linedata)
csvfile.close()

def piStim():
    # This function is called each time a TTL pulse is sent from RPi
    GPIO.output(ttlPin, True)
    time.sleep(0.001)
    GPIO.output(ttlPin, False)

class Tracking(picamera.array.PiRGBAnalysis):
    # This class is the target output of frames captures with the camera
    # It is based on picamera.array.PiRGBAnalysis class, where many other functions are defined
    def __init__(self, parent=None):
        super(Tracking, self).__init__(parent)
        # Get calibration data
        with open('calibrationTmatrix.p', 'rb') as file:
            self.calibrationTmatrix = pickle.load(file)

    def analyse(self, frame):
        # Each time a frame is catpured, this function is called on that frame
        piStim() # Send TTL pulse
        currenttime = self.camera.timestamp # Get timestamp of the RPi camera
        frametime = self.camera.frame.timestamp # Get timestamp of frame capture (PTS) in RPi camera time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert BGR data to grayscale
        gray = cv2.GaussianBlur(gray, (smoothradius, smoothradius), 0) # Smooth the image
        (minVal, maxVal_1, minLoc, maxLoc_1) = cv2.minMaxLoc(gray) # Find coordinates of pixel with highest value
        maxLoc_1 = np.reshape(np.array([[maxLoc_1[0], maxLoc_1[1]]],dtype=np.float32),(1,1,2))
        XYcoord_1 = cv2.perspectiveTransform(maxLoc_1, self.calibrationTmatrix) # Use transformation matrix to map pixel values to position in real world
        if doubleLED:
            # Find the location of second brightest point
            gray = cv2.circle(gray, (maxLoc_1[0,0,0], maxLoc_1[0,0,1]), 2 * smoothradius, 0, -1) # Make are at bright LED dark
            (minVal, maxVal_2, minLoc, maxLoc_2) = cv2.minMaxLoc(gray) # Find coordinates of pixel with highest value
            maxLoc_2 = np.reshape(np.array([[maxLoc_2[0], maxLoc_2[1]]],dtype=np.float32),(1,1,2))
            XYcoord_2 = cv2.perspectiveTransform(maxLoc_2, self.calibrationTmatrix) # Use transformation matrix to map pixel values to position in real world
            distance = euclidean(np.array(XYcoord_1[0,0,0].astype('float'), XYcoord_1[0,0,1].astype('float')), \
                                 np.array(XYcoord_2[0,0,0].astype('float'), XYcoord_2[0,0,1].astype('float')))
            if distance < LED_max_distance: # If 2nd LED is detected close enough
                # Set data into compact format
                linedata = [RPi_number, currenttime, frametime, XYcoord_2[0,0,0].astype('float'), \
                            XYcoord_2[0,0,1].astype('float'), XYcoord_1[0,0,0].astype('float'), \
                            XYcoord_1[0,0,1].astype('float'), maxVal_2, maxVal_1]
            else: # If second LED is too far away to be real, move bright LED to primary position
                linedata = [RPi_number, currenttime, frametime, XYcoord_1[0,0,0].astype('float'), \
                            XYcoord_1[0,0,1].astype('float'), None, None, maxVal_1, None]
        else:
            # Set data into compact format
            linedata = [RPi_number, currenttime, frametime, XYcoord_1[0,0,0].astype('float'), \
                        XYcoord_1[0,0,1].astype('float'), None, None, maxVal_1, None]
        # Send data over ethernet with ZeroMQ
        message = json.dumps(linedata) # Convert data to string format
        sockpub.send(message) # Send the message using ZeroMQ
        # Write data into CSV file
        csvfile = open('logfile.csv', 'a')
        logwriter = csv.writer(csvfile)
        logwriter.writerow(linedata)
        csvfile.close()
        # Save frames as jpg images if requested. Slows framerate.
        if save_frames:
            # Paint circles at locations of detected LEDs with radius of the st.dev of smoothing step
            frame = cv2.circle(frame, (maxLoc_1[0,0,0], maxLoc_1[0,0,1]), smoothradius, (255, 0, 0), 2)
            if doubleLED:
                frame = cv2.circle(frame, (maxLoc_2[0,0,0], maxLoc_2[0,0,1]), smoothradius, (0, 255, 0), 2)
            cv2.imwrite('frame{}.jpg'.format(currenttime),frame)

# Here is the actual core of the script
with picamera.PiCamera() as camera: # Initializes the RPi camera module
    with Tracking(camera) as output: # Initializes the Tracking class
        camera.resolution = (imageres[0], imageres[1]) # Set frame capture resolution
        camera.exposure_mode = exposure_setting
        camera.iso = camera_iso # Set Camera ISO value (sensitivity to light)
        camera.shutter_speed = shutter_speed # Set camera shutter speed
        setattr(picamera.array.PiRGBAnalysis, 'camera', camera) # Makes camera module accessible in Tracking class (necessary for getting timestamps from camera module)
        # Start Recording Process
        camera.start_recording(output, format='bgr') # Initializes the camera
        print('Starting recording')
        # Wait for stop command
        stopmessage = ''
        while stopmessage != 'stop': # Wait for start command over network
            stopmessage = sockrec.recv() # Receive message
        print('Stopped recording')
        camera.stop_recording() # Stop recording

GPIO.cleanup(ttlPin) # Close GPIO system
# Close zeroMQ system
sockpub.close()
sockrec.close()