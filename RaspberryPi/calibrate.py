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

# Get assigned RPi number
RPi_number = int(open('RPiNumber','r').read().splitlines()[0]) # The number to identify logs and messages from this RPi

# Get OpenEphys configuration details for this RPi
with open('RPiSettings.p','rb') as file:
    RPiSettings = pickle.load(file)
nsquares_x = RPiSettings['calibration_n_squares'][0]
nsquares_y = RPiSettings['calibration_n_squares'][1]
squaresize = RPiSettings['calibration_square']
camera_iso = RPiSettings['camera_iso']
shutter_speed = RPiSettings['shutter_speed']
exposure_setting = RPiSettings['exposure_setting']
smoothradius = RPiSettings['smoothing_radius']
imageres = RPiSettings['resolution']

class Tracking(picamera.array.PiRGBAnalysis):
    # This class is the target output of frames captures with the camera
    # It is based on picamera.array.PiRGBAnalysis class, where many other functions are defined
    def analyse(self, frame):
        if not hasattr(self, 'frames'):
            self.frames = []
        self.frames.append(frame)

    def output(self):
        return self.frames

def processFrames(frames, nsquares_y, nsquares_x, squaresize, overlay=False):
    if not overlay: # Do this unless only overlay was requested
        # Set criteria for cv2.cornerSubPix accuracy improvement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Find corners in each image
        corners = []
        for nframe in range(len(frames)):
            img = np.uint8(copy.copy(frames[nframe]))
            # print('processing frame nr ' + str(nframe + 1) + ' of ' + str(len(frames)))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, crns = cv2.findChessboardCorners(gray, (nsquares_x - 1, nsquares_y - 1), None)
            print(ret)
            if ret == True:
                # Compute search radius for when using cv2.cornerSubPix
                pixeldistance = euclidean(np.squeeze(crns[0,:,:]),np.squeeze(crns[-1,:,:]))
                objectdistance = euclidean(np.float32([0,0]),np.float32([nsquares_x - 1, nsquares_y - 1]))
                sradius = int(np.round(pixeldistance / objectdistance / 2))
                # Increase corner pixel accuracy
                crns = cv2.cornerSubPix(gray,crns,(sradius,sradius),(-1,-1),criteria)
                # Correct image flip such that 0,0 would be closest to the brightest spot
                gray = cv2.GaussianBlur(gray, (smoothradius, smoothradius), 0) # Smooth the image
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray) # Find coordinates of pixel with highest value
                crnsFirstDist = euclidean(np.squeeze(crns[0,:,:]),np.array(maxLoc))
                crnsLastDist = euclidean(np.squeeze(crns[-1,:,:]),np.array(maxLoc))
                print(crnsFirstDist)
                print(crnsLastDist)
                if crnsLastDist < crnsFirstDist:
                    print('Flipped')
                    crns = np.flipud(crns)
                # Append to corners list
                corners.append(crns)
            else:
                corners.append([])
        # Find average corners
        tmpcorners = [x for x in corners if x != []]
        print(str(len(tmpcorners)) + ' of ' + str(len(frames)) + ' images used.')
        tmp = np.concatenate(tmpcorners, axis=1)
        corners_mean = np.mean(tmp, axis=1)
        corners_mean = corners_mean.reshape((corners_mean.shape[0],1,2), order='C')
    # Find average image
    image = np.concatenate(frames, axis=2)
    image = image.reshape((frames[0].shape[0],frames[0].shape[1],frames[0].shape[2],len(frames)), order='F')
    image = np.uint8(np.mean(image, axis=3))
    # Draw image with corners
    if overlay: # Use pre-existing corners, if overlay requested
        with open('calibrationData.p', 'rb') as file:
            calibrationData = pickle.load(file)
        corners_mean = calibrationData['corners']
    image = cv2.drawChessboardCorners(image, (nsquares_x - 1,nsquares_y - 1), corners_mean, True)

    return image, corners_mean

def getTransformMatrix(corners_mean, nsquares_y, nsquares_x, squaresize):
    # Generate object point values corresponding to corners
    objp = np.zeros(((nsquares_x - 1) * (nsquares_y - 1),3), dtype=np.float32)
    objp[:,:2] = np.mgrid[0:(nsquares_x - 1),0:(nsquares_y - 1)].T.reshape(-1,2)
    objp = objp * squaresize
    # Compute transformation matrix
    transformMatrix, mask = cv2.findHomography(corners_mean, objp, cv2.RANSAC,5.0)

    return transformMatrix

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
overlay = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'overlay':
        overlay = 'True'
# Use images to find corners of the chessboard image
# or if requested, overlay previous corners on current image
image, corners_mean = processFrames(frames, nsquares_y, nsquares_x, squaresize, overlay)
if not overlay:
    # Compute transformation matrix and save it as well as calibration data
    transformMatrix = getTransformMatrix(corners_mean, nsquares_y, nsquares_x, squaresize)
    with open('calibrationTmatrix.p', 'wb') as file:
        pickle.dump(transformMatrix, file)
    calibrationData = {'image': image, 'corners': corners_mean, 'nsquares_y': nsquares_y, 'nsquares_x': nsquares_x, 'squaresize': squaresize}
    with open('calibrationData.p', 'wb') as file:
        pickle.dump(calibrationData, file)
else: # If requested, simply save the current image with overlay of previous corners
    cv2.imwrite('overlay.jpg', image)
