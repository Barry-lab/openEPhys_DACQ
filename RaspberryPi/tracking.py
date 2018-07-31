import numpy as np
import picamera
import picamera.array
from time import sleep, time
import cv2
import csv
import zmq
import json
from RPi import GPIO
import os
import cPickle as pickle
from scipy.spatial.distance import euclidean
import sys
from multiprocessing import Process, RawArray, Lock
from ctypes import c_uint8, c_bool
from copy import copy
from Queue import Queue
from threading import Thread
from ZMQcomms import paired_messenger
import argparse
from shutil import rmtree

# # Write output to a log file
# orig_stdout = sys.stdout
# sys.stdout = open('stdout', 'w')
# sys.stderr = open('stderr','w')

def save_incoming_frames(path_to_frames, frame_RawArray, frame_shape, 
                         P_save_incoming_frames_Stopper_Lock, 
                         P_save_incoming_frames_Processing_Start_Lock, 
                         P_save_incoming_frames_Processing_End_Lock):
    frame_number = 0
    while not P_save_incoming_frames_Stopper_Lock.acquire(block=True, timeout=0.0005):
        while P_save_incoming_frames_Processing_Start_Lock.acquire(block=False):
            frame_number += 1
            frame = np.frombuffer(frame_RawArray, dtype=c_uint8).reshape(frame_shape)
            filename = os.path.join(path_to_frames, str(frame_number) + '.jpg')
            cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            P_save_incoming_frames_Processing_End_Lock.release()

def process_this_frame(frame, currenttime, frametime, params, write_to_logfile, send_data_with_ZMQpublisher):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert BGR data to grayscale
    gray = cv2.GaussianBlur(gray, (params['smoothing_radius'], params['smoothing_radius']), 0) # Smooth the image
    (_1, maxVal_1, _2, maxLoc_1) = cv2.minMaxLoc(gray) # Find coordinates of pixel with highest value
    maxLoc_1 = np.reshape(np.array([[maxLoc_1[0], maxLoc_1[1]]]),(1,1,2))
    XYcoord_1 = cv2.perspectiveTransform(maxLoc_1.astype(np.float32), params['calibrationTmatrix']).astype('float') # Use transformation matrix to map pixel values to position in real world
    if params['doubleLED']:
        # Find the location of second brightest point
        gray = cv2.circle(gray, (maxLoc_1[0,0,0], maxLoc_1[0,0,1]), params['LED_radius_pix'], 0, -1) # Make are at bright LED dark
        (_1, maxVal_2, _2, maxLoc_2) = cv2.minMaxLoc(gray) # Find coordinates of pixel with highest value
        maxLoc_2 = np.reshape(np.array([[maxLoc_2[0], maxLoc_2[1]]]),(1,1,2))
        XYcoord_2 = cv2.perspectiveTransform(maxLoc_2.astype(np.float32), params['calibrationTmatrix']).astype('float') # Use transformation matrix to map pixel values to position in real world
        distance = euclidean(np.array([XYcoord_1[0,0,0], XYcoord_1[0,0,1]]), 
                             np.array([XYcoord_2[0,0,0], XYcoord_2[0,0,1]]))
        if distance < params['LED_max_distance']: # If 2nd LED is detected close enough
            # Set data into compact format
            linedata = [params['RPi_number'], currenttime, frametime, XYcoord_1[0,0,0], XYcoord_1[0,0,1], 
                        XYcoord_2[0,0,0], XYcoord_2[0,0,1], maxVal_1, maxVal_2]
        else: # If second LED is too far away to be real, move bright LED to primary position
            linedata = [params['RPi_number'], currenttime, frametime, XYcoord_1[0,0,0], XYcoord_1[0,0,1], 
                        None, None, maxVal_1, None]
    else:
        # Set data into compact format
        linedata = [params['RPi_number'], currenttime, frametime, XYcoord_1[0,0,0], XYcoord_1[0,0,1], 
                    None, None, maxVal_1, None]
    # Send data over ethernet with ZeroMQ
    send_data_with_ZMQpublisher(linedata)
    # Write data into CSV file
    write_to_logfile(linedata)

class Frame_Handler(object):
    def __init__(self, params):
        self.parse_params(params)
        self.init_queue_processor()
        self.init_logfile_writer()
        self.init_ZMQpublisher()
        if self.save_frames:
            self.init_save_incoming_frames()

    def parse_params(self, params):
        self.params = params
        self.frame_shape = (params['resolution'][1], params['resolution'][0], 3)
        self.save_frames = params['save_frames']
        self.RPiIP = params['RPiIP']
        self.pos_port = params['pos_port']

    def init_queue_processor(self):
        self.processing_queue = Queue()
        self.T_queue_processor = Thread(target=self.queue_processor)
        self.T_queue_processor.setDaemon(True)
        self.T_queue_processor.start()

    def init_logfile_writer(self):
            if os.path.exists('logfile.csv'):
                os.remove('logfile.csv')
            self.logfile = open('logfile.csv', 'a')
            self.logwriter = csv.writer(self.logfile)

    def init_ZMQpublisher(self):
        posIP = 'tcp://' + self.RPiIP + ':' + self.pos_port
        # For sending position data
        contextpub = zmq.Context()
        self.ZMQpublisher = contextpub.socket(zmq.PUB)
        self.ZMQpublisher.bind(posIP)
        sleep(0.1) # Give time to establish sockets for ZeroMQ

    def init_save_incoming_frames(self):
        self.path_to_frames = 'frames'
        if os.path.exists(self.path_to_frames):
            rmtree(self.path_to_frames)
        os.mkdir(self.path_to_frames)
        # Initialize save_incoming_frames process
        numel = int(self.frame_shape[0] * self.frame_shape[1] * self.frame_shape[2])
        self.frame_RawArray = RawArray(c_uint8, numel)
        self.frame_RawArray_Numpy_Wrapper = np.frombuffer(self.frame_RawArray, dtype=c_uint8).reshape(self.frame_shape)
        # This lock starts locked. When this lock is released, save_incoming_frames process stops running
        self.P_save_incoming_frames_Stopper_Lock = Lock()
        self.P_save_incoming_frames_Stopper_Lock.acquire()
        # This lock starts locked. When released, save_incoming_frames processes frame_RawArray and locks this lock.
        self.P_save_incoming_frames_Processing_Start_Lock = Lock()
        self.P_save_incoming_frames_Processing_Start_Lock.acquire()
        # This lock starts locked. When save_incoming_frames has finished processing frame_RawArray, it releases this lock.
        # The parent process waits for this to happend so it can lock this lock again.
        self.P_save_incoming_frames_Processing_End_Lock = Lock()
        self.P_save_incoming_frames_Processing_End_Lock.acquire()
        self.P_save_incoming_frames = Process(target=save_incoming_frames, 
                                              args=(self.path_to_frames, self.frame_RawArray, self.frame_shape, 
                                                    self.P_save_incoming_frames_Stopper_Lock, 
                                                    self.P_save_incoming_frames_Processing_Start_Lock, 
                                                    self.P_save_incoming_frames_Processing_End_Lock))
        self.P_save_incoming_frames.start()

    def write_to_logfile(self, linedata):
        self.logwriter.writerow(linedata)

    def send_data_with_ZMQpublisher(self, data):
        message = json.dumps(data) # Convert data to string format
        self.ZMQpublisher.send(message) # Send the message using ZeroMQ

    def processor(self, frame, currenttime, frametime):
        # If frames are to be saved, share frame data with the separate process for saving frames.
        if self.save_frames:
            np.copyto(self.frame_RawArray_Numpy_Wrapper, frame)
            self.P_save_incoming_frames_Processing_Start_Lock.release()
        # Process the frame, store and broadcast the results
        process_this_frame(frame, currenttime, frametime, 
                           self.params, self.write_to_logfile, 
                           self.send_data_with_ZMQpublisher)
        # If frames are being saved, wait until this frame is saved.
        if self.save_frames:
            self.P_save_incoming_frames_Processing_End_Lock.acquire()

    def queue_processor(self):
        while True:
            args = self.processing_queue.get()
            self.processor(*args)
            self.processing_queue.task_done()

    def add_frame_to_processing_queue(self, frame, currenttime, frametime):
        self.processing_queue.put((frame, currenttime, frametime))

    def close(self):
        self.processing_queue.join()
        if hasattr(self, 'P_save_incoming_frames_Stopper_Lock'):
            self.P_save_incoming_frames_Stopper_Lock.release()
        self.logfile.close()
        self.ZMQpublisher.close()


class PiCameraOutput(picamera.array.PiRGBAnalysis):

    def __init__(self, camera, Frame_Handler_Params, **kwargs):
        super(PiCameraOutput, self).__init__(camera, **kwargs)
        self.frame_handler = Frame_Handler(Frame_Handler_Params)
        self.active_camera = camera # This is necessary for grabbing timestamps from camera
        self.init_TTL_signalling()
        self.Use_PiCamera_Stream = False
        self.analyse_method_active = False

    def init_TTL_signalling(self, ttlPin=11):
        self.ttlPin = ttlPin
        # Set up GPIO system for TTL pulse signalling
        GPIO.setmode(GPIO.BOARD) # Use the mapping based on physical bin numbering
        GPIO.setup(self.ttlPin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.output(self.ttlPin, False)

    def send_TTL_signal(self):
        GPIO.output(self.ttlPin, True)
        sleep(0.001)
        GPIO.output(self.ttlPin, False)

    def analyse(self, frame):
        if self.Use_PiCamera_Stream:
            self.analyse_method_active = True
            # Each time a frame is catpured, this function is called on that frame
            self.send_TTL_signal() # Send TTL pulse
            currenttime = copy(self.active_camera.timestamp) # Get timestamp of the RPi camera
            frametime = copy(self.active_camera.frame.timestamp) # Get timestamp of frame capture (PTS) in RPi camera time
            self.frame_handler.add_frame_to_processing_queue(frame, currenttime, frametime)
            self.analyse_method_active = False

    def start(self):
        self.Use_PiCamera_Stream = True

    def close(self, *args, **kwargs):
        super(PiCameraOutput, self).close(*args, **kwargs)
        self.Use_PiCamera_Stream = False # Ensures no further frames are sent to analyse
        while self.analyse_method_active: # Ensures all frames in analyse have been passed to self.frame_handler
            sleep(0.01)
        self.frame_handler.close()
        GPIO.cleanup(self.ttlPin)


def RPiNumber():
    with open('RPiNumber','r') as file:
        RPi_number = int(file.read().splitlines()[0])

    return RPi_number

def distance_between_adjacent_pixels(calibrationTmatrix, resolution):
    '''
    Approximation of distance in centimeters correspond to single pixel difference at the center of field of view.
    '''
    distance_in_pixels = 10
    # Pick two points at the center of the field of view
    tmp_loc1 = np.reshape(np.array([int(resolution[0] / 2), int(resolution[1] / 2)],dtype=np.float32),(1,1,2))
    tmp_loc2 = np.reshape(np.array([int(resolution[0] / 2), int(resolution[1] / 2) + distance_in_pixels],dtype=np.float32),(1,1,2))
    # Use transformation matrix to map pixel values to position in real world in centimeters
    tmp_loc1 = cv2.perspectiveTransform(tmp_loc1, calibrationTmatrix)
    tmp_loc2 = cv2.perspectiveTransform(tmp_loc2, calibrationTmatrix)
    # Compute the distance between the two points in real world centimeters
    distance_in_cm = euclidean(np.array([tmp_loc1[0,0,0].astype('float'), tmp_loc1[0,0,1].astype('float')]), 
                               np.array([tmp_loc2[0,0,0].astype('float'), tmp_loc2[0,0,1].astype('float')]))
    # Compute distance in centimeters between adjacent pixels
    distance = float(distance_in_cm) / float(distance_in_pixels)

    return distance

def convert_centimeters_to_pixel_distance(distance_in_cm, calibrationTmatrix, resolution):
    return int(np.round(float(distance_in_cm) / distance_between_adjacent_pixels(calibrationTmatrix, resolution)))

class CameraController(object):
    def __init__(self, TrackingSettings=None, remoteControl=False):
        if TrackingSettings is None:
            with open('TrackingSettings.p','rb') as file:
                TrackingSettings = pickle.load(file)
        self.remoteControl = remoteControl
        self.RPiNumber = RPiNumber()
        # Set start and end triggers
        if self.remoteControl:
            self.start_image_acquisition = False
            self.stop_image_acquisition = False
        # Initialize ZMQ communications
        if self.remoteControl:
            self.initialize_ZMQcomms(int(TrackingSettings['stop_port']))
        # Parse recording settings
        self.camera_iso = TrackingSettings['camera_iso']
        self.shutter_speed = TrackingSettings['shutter_speed']
        self.exposure_setting = TrackingSettings['exposure_setting']
        self.resolution = TrackingSettings['resolution']
        self.Frame_Handler_Params = self.prepare_Frame_Handler_Params(TrackingSettings)
        # Specify framerate based on how fast Frame_Handler processes a single frame
        if self.Frame_Handler_Params['save_frames']:
            self.framerate = 25
        else:
            self.framerate = 30

    def initialize_ZMQcomms(self, port):
        self.ZMQmessenger = paired_messenger(port=port)
        self.ZMQmessenger.add_callback(self.command_parser)
        sleep(1) # This ensures all ZMQ protocols have been properly initated before finishing this process

    def command_parser(self, message):
        '''
        Parses incoming ZMQ message for function name, input arguments and calls that function.
        '''
        method_name = message.split(' ')[0]
        args = message.split(' ')[1:]
        getattr(self, method_name)(*args)

    def prepare_Frame_Handler_Params(self, TrackingSettings):
        Frame_Handler_Params = {}
        Frame_Handler_Params['RPi_number'] = self.RPiNumber
        # Load the Calibration Matrix
        if str(self.RPiNumber) in TrackingSettings['calibrationData'].keys():
            calibrationTmatrix = TrackingSettings['calibrationData'][str(self.RPiNumber)]['calibrationTmatrix']
        else:
            raise ValueError('Calibration data does not exist for this RPi.')
        Frame_Handler_Params['calibrationTmatrix'] = calibrationTmatrix
        # Identify if doubleLED tracking is enabled
        if TrackingSettings['LEDmode'] == 'double':
            Frame_Handler_Params['doubleLED'] = True
        elif TrackingSettings['LEDmode'] == 'single':
            Frame_Handler_Params['doubleLED'] = False
        # Find values for distances between LED
        LED_separation = TrackingSettings['LED_separation']
        Frame_Handler_Params['LED_max_distance'] = LED_separation * 2
        LED_radius_pix = convert_centimeters_to_pixel_distance(LED_separation / 2.0, 
                                                               Frame_Handler_Params['calibrationTmatrix'], 
                                                               self.resolution)
        Frame_Handler_Params['LED_radius_pix'] = LED_radius_pix
        # Other parameters
        Frame_Handler_Params['resolution'] = TrackingSettings['resolution']
        Frame_Handler_Params['save_frames'] = TrackingSettings['save_frames']
        Frame_Handler_Params['smoothing_radius'] = TrackingSettings['smoothing_radius']
        Frame_Handler_Params['pos_port'] = TrackingSettings['pos_port']
        Frame_Handler_Params['RPiIP'] = TrackingSettings['RPiInfo'][str(self.RPiNumber)]['IP']

        return Frame_Handler_Params

    def start(self):
        self.start_image_acquisition = True

    def stop(self):
        self.stop_image_acquisition = True

    def run_Camera(self):
        # Here is the actual core of the script
        with picamera.PiCamera(clock_mode='raw', framerate=self.framerate) as camera: # Initializes the RPi camera module
            camera.resolution = (self.resolution[0], self.resolution[1]) # Set frame capture resolution
            camera.exposure_mode = self.exposure_setting
            camera.iso = self.camera_iso # Set Camera ISO value (sensitivity to light)
            camera.shutter_speed = self.shutter_speed # Set camera shutter speed
            with PiCameraOutput(camera, self.Frame_Handler_Params) as output: # Initializes the PiCameraOutput class
                camera.start_recording(output, format='bgr') # Initializes the camera
                # Wait for starting command
                if self.remoteControl:
                    self.ZMQmessenger.sendMessage('init_successful')
                    while not self.start_image_acquisition:
                        sleep(0.01)
                else:
                    _ = raw_input('Press enter to start image acquisition: ')
                output.start()
                # Wait for stop command
                if self.remoteControl:
                    while not self.stop_image_acquisition:
                        sleep(0.1)
                else:
                    _ = raw_input('Press enter to stop image acquisition: ')
                camera.stop_recording() # Stop recording
        # Close ZMQ messenger
        if self.remoteControl:
            self.ZMQmessenger.close()

if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Running this script initates CameraController class.')
    parser.add_argument('--remote', action='store_true', 
                        help='Expects start and stop commands over ZMQ. Default is keyboard input.')
    args = parser.parse_args()
    # If releasePellet command given skip everything and just release the number of pellets specified
    if args.remote:
        Controller = CameraController(remoteControl=True)
        Controller.run_Camera()
    else:
        Controller = CameraController()
        Controller.run_Camera()
