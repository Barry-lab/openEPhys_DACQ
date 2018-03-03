### This script finds captures a series of images with RPi camera and saves the last one as 'frame.jpg'
### It is used by Camera Settings GUI to capture current image.

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
from scipy.spatial.distance import euclidean
import cPickle as pickle
import copy
import sys

# Get assigned RPi number
RPi_number = int(open('RPiNumber','r').read().splitlines()[0]) # The number to identify logs and messages from this RPi

# Get OpenEphys configuration details for this RPi
with open('TrackingSettings.p','rb') as file:
    TrackingSettings = pickle.load(file)
camera_iso = TrackingSettings['camera_iso']
shutter_speed = TrackingSettings['shutter_speed']
exposure_setting = TrackingSettings['exposure_setting']
imageres = TrackingSettings['resolution']

class Tracking(picamera.array.PiRGBAnalysis):
    # This class is the target output of frames captures with the camera
    # It is based on picamera.array.PiRGBAnalysis class, where many other functions are defined
    def analyse(self, frame):
        if not hasattr(self, 'frames'):
            self.frames = []
        self.frames.append(frame)

    def output(self):
        return self.frames

# Here is the actual core of the script
# Aquire a series of images with same script as used for tracking
frames = []
with picamera.PiCamera() as camera: # Initializes the RPi camera module
    with Tracking(camera) as output: # Initializes the Tracking class
        camera.resolution = (imageres[0], imageres[1]) # Set frame capture resolution
        camera.exposure_mode = exposure_setting
        camera.iso = camera_iso # Set Camera ISO value (sensitivity to light)
        camera.shutter_speed = shutter_speed # Set camera shutter speed
        camera.start_recording(output, format='bgr') # Initializes the camera
        time.sleep(2)
        camera.stop_recording() # Stop recording
        frames = output.output()

# Only keep the last frame
frame = frames[-1]
# Save the frame to disk
cv2.imwrite('frame.jpg', frame)
