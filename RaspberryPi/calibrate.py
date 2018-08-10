### This script captures a series of images with the RPi camera and tries to
### find corners between squares of a chessboard pattern.
### A transformation matrix is generated that can be used to map pixel values
### to positions on the chessobard pattern.

### The script has an extra option to load data saved during previous calibration
### and 'overlay' the pattern of chessboard corners on the current image from camera.

# By Sander Tanni, April 2017, UCL

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

class Tracking(picamera.array.PiRGBAnalysis):
    # This class is the target output of frames captures with the camera
    # It is based on picamera.array.PiRGBAnalysis class, where many other functions are defined
    def analyse(self, frame):
        if not hasattr(self, 'frames'):
            self.frames = []
        self.frames.append(frame)

    def output(self):
        return self.frames

def processFrames(RPi_number,frames, ndots_x, ndots_y, spacing, overlay=None):
    if overlay is None: # Do this unless only overlay was requested
        # Find pattern in each image
        patterns = []
        for frame in frames:
            img = np.uint8(copy.copy(frame))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
            ret, crns = cv2.findCirclesGrid(gray,(ndots_x, ndots_y),flags=flags)
            if ret == True:
                # Append to pattern list
                patterns.append(crns)
            else:
                patterns.append([])
        # Find average pattern
        good_patterns = [x for x in patterns if x != []]
        n_successful_patterns = len(good_patterns)
        print('RPi Nr ' + str(RPi_number + 1) + ': ' + \
              str(n_successful_patterns) + ' of ' + str(len(frames)) + ' images used.')
        if n_successful_patterns >= 1:
            good_patterns = np.stack(good_patterns, axis=3)
            pattern_mean = np.mean(good_patterns, axis=3)
        else:
            pattern_mean = None
    # Find average image
    image = np.stack(frames, axis=3)
    image = np.uint8(np.mean(image, axis=3))
    # Draw image with corners
    if not (overlay is None): # Use pre-existing corners, if overlay requested
        pattern_mean = overlay
    if not (pattern_mean is None):
        image = cv2.drawChessboardCorners(image, (ndots_x, ndots_y), pattern_mean, True)

    return image, pattern_mean

def getTransformMatrix(pattern, ndots_x, ndots_y, spacing, corner_offset):
    # Generate object point values corresponding to the pattern
    objp = np.mgrid[0:ndots_x,0:ndots_y].T.reshape(-1,2).astype(np.float32)
    objp[:,1] = objp[:,1] / 2
    shiftrows = np.arange(1,ndots_y,2)
    for row in shiftrows:
        tmpidx = np.arange(row * ndots_x, (row + 1) * ndots_x)
        objp[tmpidx,0] = objp[tmpidx,0] + 0.5
    # Stretch the object point values to scale with the real pattern
    objp = objp * spacing
    # Add offset from arena corner to get circle locations in the arena
    objp[:,0] = objp[:,0] + corner_offset[0]
    objp[:,1] = objp[:,1] + corner_offset[1]
    # Add the zeros to force pattern onto the plane in 3D world
    objp = np.concatenate((objp, np.zeros((objp.shape[0],1))), 1)
    # Compute transformation matrix
    calibrationTmatrix, mask = cv2.findHomography(pattern, objp, cv2.RANSAC,5.0)
    
    return calibrationTmatrix

def main():
    # Get assigned RPi number
    RPi_number = int(open('RPiNumber','r').read().splitlines()[0]) # The number to identify logs and messages from this RPi

    # Get OpenEphys configuration details for this RPi
    with open('TrackingSettings.p','rb') as file:
        TrackingSettings = pickle.load(file)
    ndots_x = TrackingSettings['calibration_n_dots'][0]
    ndots_y = TrackingSettings['calibration_n_dots'][1]
    spacing = TrackingSettings['calibration_spacing']
    corner_offset = [TrackingSettings['corner_offset'][0], TrackingSettings['corner_offset'][1]]
    camera_iso = TrackingSettings['camera_iso']
    shutter_speed = TrackingSettings['shutter_speed']
    exposure_setting = TrackingSettings['exposure_setting']
    imageres = TrackingSettings['resolution']

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
            frames = output.output() # Save recorded frames to a variable available for rest of script
    # Only keep the last 5 frames to maximise quality and reduce processing time
    frames = frames[-5:]
    # Check if overlaying existing calibrationData was requested
    overlay = None
    if len(sys.argv) > 1:
        if sys.argv[1] == 'overlay':
            if str(RPi_number) in TrackingSettings['calibrationData'].keys():
                overlay = TrackingSettings['calibrationData'][str(RPi_number)]['pattern']
    # Use images to find pattern of the chessboard image
    # or if requested, overlay previous pattern on current image
    image, pattern = processFrames(RPi_number, frames, ndots_x, ndots_y, spacing, overlay)
    if overlay is None:
        # Compute transformation matrix and save it as well as calibration data
        if not (pattern is None):
            calibrationTmatrix = getTransformMatrix(pattern, ndots_x, ndots_y, spacing, corner_offset)
        else:
            calibrationTmatrix = None
        if calibrationTmatrix is None or pattern is None:
            calibrationData = {'image': image, 'ndots_x': ndots_x, 'ndots_y': ndots_y, 'spacing': spacing}
        else:
            calibrationData = {'calibrationTmatrix': calibrationTmatrix, 'image': image, 
                               'pattern': pattern, 'ndots_x': ndots_x, 
                               'ndots_y': ndots_y, 'spacing': spacing}
        with open('calibrationData.p', 'wb') as file:
            pickle.dump(calibrationData, file)
    else: # If requested, simply save the current image with overlay of previous pattern
        cv2.imwrite('overlay.jpg', image)

if __name__ == '__main__':
    main()
