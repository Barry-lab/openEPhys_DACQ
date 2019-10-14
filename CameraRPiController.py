from __future__ import division
import numpy as np
import picamera
import picamera.array
from picamera import mmal
from time import sleep, time
import cv2
import csv
import zmq
import json
import pigpio
import os
from scipy.spatial.distance import euclidean
from multiprocessing import Manager, Process, RawArray, Value
from threading import Thread
from copy import copy
from ZMQcomms import remote_controlled_class
import argparse
from ctypes import c_uint8, c_uint16, c_bool
import warnings
import io
import socket
import struct
from PIL import Image
from subprocess import PIPE, Popen
from functools import reduce
import psutil


class csv_writer(object):
    def __init__(self, filename):
        '''
        Note! Any file at filename is overwritten.
        '''
        if os.path.exists(filename):
            os.remove(filename)
        self.logfile = open(filename, 'a')
        self.logwriter = csv.writer(self.logfile)

    def write(self, data):
        '''
        data - list - written to CSV file as a row
        '''
        self.logwriter.writerow(data)

    def close(self):
        self.logfile.close()


def distance_between_adjacent_pixels(calibrationTmatrix, frame_shape):
    '''
    Approximation of distance in centimeters correspond to single pixel difference at the center of field of view.
    '''
    distance_in_pixels = 10
    # Pick two points at the center of the field of view
    tmp_loc1 = np.reshape(np.array([int(frame_shape[1] / 2), int(frame_shape[0] / 2)],dtype=np.float32),(1,1,2))
    tmp_loc2 = np.reshape(np.array([int(frame_shape[1] / 2), int(frame_shape[0] / 2) + distance_in_pixels],dtype=np.float32),(1,1,2))
    # Use transformation matrix to map pixel values to position in real world in centimeters
    tmp_loc1 = cv2.perspectiveTransform(tmp_loc1, calibrationTmatrix)
    tmp_loc2 = cv2.perspectiveTransform(tmp_loc2, calibrationTmatrix)
    # Compute the distance between the two points in real world centimeters
    distance_in_cm = euclidean(np.array([tmp_loc1[0,0,0].astype('float'), tmp_loc1[0,0,1].astype('float')]), 
                               np.array([tmp_loc2[0,0,0].astype('float'), tmp_loc2[0,0,1].astype('float')]))
    # Compute distance in centimeters between adjacent pixels
    distance = float(distance_in_cm) / float(distance_in_pixels)

    return distance

def convert_centimeters_to_pixel_distance(distance_in_cm, calibrationTmatrix, frame_shape):
    return int(np.round(float(distance_in_cm) / distance_between_adjacent_pixels(calibrationTmatrix, frame_shape)))


class RPiMonitorLogger(object):

    def __init__(self, frequency=2, queue_length_method=None):
        self._frequency = frequency
        self._queue_length_method = queue_length_method
        self._continue_bool = Value(c_bool)
        self._continue_bool.value = True
        self._queue_length = Value(c_uint16)
        self._P_logger = Process(target=RPiMonitorLogger.logger, 
                                 args=(self._continue_bool, 
                                       self._frequency, 
                                       self._queue_length))
        self._P_logger.start()
        if not (queue_length_method is None):
            self._keep_updating_queue_length = True
            self._T_update_queue_length = Thread(target=self._update_queue_length)
            self._T_update_queue_length.start()

    def _update_queue_length(self):
        while self._keep_updating_queue_length:
            self._queue_length.value = int(self._queue_length_method())
            sleep(1.0 / self._frequency)

    @staticmethod
    def get_cpu_temperature():
        process = Popen(['vcgencmd', 'measure_temp'], stdout=PIPE)
        output, _error = process.communicate()
        output = output.decode()
        return float(output[output.index('=') + 1:output.rindex("'")])

    @staticmethod
    def get_cpu_usage():
        return psutil.cpu_percent()

    @staticmethod
    def get_memory_usage():
        return psutil.virtual_memory().percent

    @staticmethod
    def get_disk_usage():
        return psutil.disk_usage('/').percent

    @staticmethod
    def logger(continue_bool, frequency, queue_length):
        writer = csv_writer('RPiMonitorLog.csv')
        # Write header
        writer.write(['time_from_start', 
                      'cpu_temperature', 
                      'cpu_usage', 
                      'memory_usage', 
                      'disk_usage', 
                      'queue_length'])
        # Get start time
        start_time = time()
        # Keep writing lines at specified rate
        while continue_bool.value:
            writer.write([time() - start_time, 
                          RPiMonitorLogger.get_cpu_temperature(), 
                          RPiMonitorLogger.get_cpu_usage(), 
                          RPiMonitorLogger.get_memory_usage(), 
                          RPiMonitorLogger.get_disk_usage(), 
                          queue_length.value])
            sleep(1.0 / float(frequency))
        # Close CSV file before finishing process
        writer.close()

    def close(self):
        self._continue_bool.value = False
        self._P_logger.join()
        if hasattr(self, '_T_update_queue_length'):
            self._keep_updating_queue_length = False
            self._T_update_queue_length.join()


class OnlineTracker(object):
    '''
    Inherits process_with_queue functionality and instantiates multiprocess_function class
    for processing each frame in the queue with detect_leds() method.
    The output from save_frame is published with ZMQ and stored in a csv file.

    The nature of online tracking requires frames to be processed in correct sequence.
    It is therefore essential that detect_leds() method would run faster than frame rate.
    '''

    def __init__(self, params):
        '''
        Inherits __init__() from process_with_queue and adds initialization of ZMQpublisher, 
        csv file writer and multiprocess_function class for running detect_leds() method.

        params - dict - must contain all the parameters used by the class:
            'RPiIP' - str
            'OnlineTracker_port' - str
            'smoothing_box' - int
            'calibrationTmatrix' - array
            'tracking_mode' - str
            'LED_separation' - float
            'frame_shape' - tuple - (height, width, channels) of incoming frames
        '''
        params['LED_separation_pix'] = convert_centimeters_to_pixel_distance(params['LED_separation'], 
                                                                             params['calibrationTmatrix'], 
                                                                             params['frame_shape'])
        self.init_ZMQpublisher(params['RPiIP'], params['OnlineTracker_port'])
        self.csv_writer = csv_writer('OnlineTrackerData.csv')
        if params['tracking_mode'] == 'dual_led' or params['tracking_mode'] == 'single_led':
            params['cutout'] = self.create_circular_cutout(params)
        elif params['tracking_mode'] == 'motion':
            self.last_frame = None
        else:
            raise ValueError('tracking_mode not recognized in params.')
        self.params = params

    def init_ZMQpublisher(self, RPiIP, OnlineTracker_port):
        '''
        Sets up publishing messages with ZMQ at 'localhost'.
        '''
        posIP = 'tcp://' + RPiIP + ':' + str(OnlineTracker_port)
        # For sending position data
        contextpub = zmq.Context()
        self.ZMQpublisher = contextpub.socket(zmq.PUB)
        self.ZMQpublisher.bind(posIP)
        sleep(0.5) # Give time to establish sockets for ZeroMQ

    @staticmethod
    def transform_pix_to_cm(pos_pix, calibrationTmatrix):
        '''
        Transforms position on image from pixels to centimeters in real world coordinates,
        based on transformation matrix.
        Coordinate values in C-major order relative to the grayscale image.
        '''
        # Use transformation matrix to map pixel values to position in real world
        led_pix_4PT = np.reshape(np.array(pos_pix[::-1]),(1,1,2)).astype(np.float32)
        pos_cm_fromPT = cv2.perspectiveTransform(led_pix_4PT, calibrationTmatrix).astype('float')
        pos_cm = pos_cm_fromPT.squeeze()

        return pos_cm

    @staticmethod
    def detect_first_led(gray, calibrationTmatrix):
        '''
        Finds the highest luminance in the grayscale image and transforms pix location to real world coordinates.
        Coordinate values in C-major order relative to the grayscale image.
        '''
        (_1, lum, _2, led_pix) = cv2.minMaxLoc(gray) # Find coordinates of pixel with highest value
        led_pix = led_pix[::-1]
        xy_cm = OnlineTracker.transform_pix_to_cm(led_pix, calibrationTmatrix)

        return led_pix, xy_cm, lum

    @staticmethod
    def create_circular_cutout(params):
        '''
        Creates a blank array and indexing arrays to fill it with values from image
        '''
        # Extract necessary values from params
        cutout_radius = params['LED_separation_pix'] * 2
        img_shape = params['frame_shape'][:2]
        # Create array blank array for search area
        cutout_shape = (cutout_radius * 2 + 1, cutout_radius * 2 + 1)
        blank = np.zeros(cutout_shape, dtype=np.uint8)
        # Find indices in array at correct distance from center
        all_indices = np.unravel_index(list(range(blank.size)), blank.shape, order='F')
        center_ind = [cutout_radius, cutout_radius]
        ind_1 = np.array([], dtype=np.int16)
        ind_2 = np.array([], dtype=np.int16)
        for a,b in zip(*all_indices):
            if euclidean(center_ind, [a, b]) <= cutout_radius:
                ind_1 = np.append(ind_1, a)
                ind_2 = np.append(ind_2, b)
        # Create extraction indices
        ind_1_extr = ind_1 - cutout_radius
        ind_2_extr = ind_2 - cutout_radius
        # Combine info into a dictionary
        cutout = {'blank': blank, 'ind_1': ind_1, 'ind_2': ind_2, 
                  'ind_1_extr': ind_1_extr, 'ind_2_extr': ind_2_extr, 
                  'img_shape': img_shape, 'cutout_radius': cutout_radius}

        return cutout

    @staticmethod
    def center_and_crop_cutout(cutout, center_pix):
        '''
        Shifts the cutout indexing arrays to center_pix and removes indices outside frame.
        '''
        new_cutout = copy(cutout)
        # Create extraction indices centered at center_pix
        new_cutout['ind_1_extr'] = new_cutout['ind_1_extr'] + center_pix[0]
        new_cutout['ind_2_extr'] = new_cutout['ind_2_extr'] + center_pix[1]
        # Keep only indices that are in img_shape
        ind_1_inside = np.logical_and(0 <= new_cutout['ind_1_extr'], 
                                      new_cutout['ind_1_extr'] < new_cutout['img_shape'][0])
        ind_2_inside = np.logical_and(0 <= new_cutout['ind_2_extr'], 
                                      new_cutout['ind_2_extr'] < new_cutout['img_shape'][1])
        ind_inside = np.logical_and(ind_1_inside, ind_2_inside)
        if np.sum(ind_inside) < new_cutout['ind_1_extr'].size:
            new_cutout['ind_1_extr'] = new_cutout['ind_1_extr'][ind_inside]
            new_cutout['ind_2_extr'] = new_cutout['ind_2_extr'][ind_inside]
            new_cutout['ind_1'] = new_cutout['ind_1'][ind_inside]
            new_cutout['ind_2'] = new_cutout['ind_2'][ind_inside]

        return new_cutout

    @staticmethod
    def detect_max_luminance_in_circular_area(img, center_pix, cutout):
        '''
        Return coordinates and value of maximum pixel in range of center_pix.
        Coordinate values in C-major order relative to the grayscale image.
        '''
        shifted_cutout = OnlineTracker.center_and_crop_cutout(cutout, center_pix)
        # Extract pixel values from img and put into blank cutout array
        pix_vals = img[shifted_cutout['ind_1_extr'], shifted_cutout['ind_2_extr']]
        img_cutout = shifted_cutout['blank']
        img_cutout[shifted_cutout['ind_1'], shifted_cutout['ind_2']] = pix_vals
        # Find maximum value in img_cutout and convert to correct coordinates in img
        (_1, lum, _2, led_pix_cutout) = cv2.minMaxLoc(img_cutout)
        led_pix_cutout = led_pix_cutout[::-1]
        led_pix = np.array(led_pix_cutout) - shifted_cutout['cutout_radius'] + center_pix
        led_pix = tuple(led_pix)

        return led_pix, lum

    @staticmethod
    def detect_second_led(gray, calibrationTmatrix, led_pix_1, LED_separation_pix, cutout):
        '''
        Returns coordinates of second LED in centimeters.
        led_pix_1 location in image is masked in range of LED_separation_pix / 2.0 and
        second LED is the brightest luminance within LED_separation_pix * 2 of led_pix_1.
        Coordinate values in C-major order relative to the grayscale image.
        '''
        gray = cv2.circle(gray, tuple(led_pix_1[::-1]), int(round(LED_separation_pix / 2.0)), 0, -1)
        led_pix, lum = OnlineTracker.detect_max_luminance_in_circular_area(gray, led_pix_1, cutout)
        xy_cm = OnlineTracker.transform_pix_to_cm(led_pix, calibrationTmatrix)

        return led_pix, xy_cm, lum

    @staticmethod
    def detect_leds(gray, params):
        '''
        Detects brightest point on grayscaled data after smoothing.
        If two LEDs in use, tries finding the second brithest spot in proximity of the first.
        Returns a list with values:
            xcoord of first LED
            ycoord of first LED
            xcoord of second LED
            ycoord of second LED
            luminance of first LED
            luminance of second LED

            Values are None for second LED if not requested or not found.
        '''
        gray = cv2.blur(gray, ksize=(params['smoothing_box'], params['smoothing_box']))
        led_pix_1, led_cm_1, lum_1 = OnlineTracker.detect_first_led(gray, params['calibrationTmatrix'])
        if params['tracking_mode'] == 'dual_led':
            led_pix_2, led_cm_2, lum_2 = OnlineTracker.detect_second_led(gray, params['calibrationTmatrix'], 
                                                                          led_pix_1, params['LED_separation_pix'], 
                                                                          params['cutout'])
            linedata = [led_cm_1[0], led_cm_1[1], led_cm_2[0], led_cm_2[1], lum_1, lum_2]
        elif params['tracking_mode'] =='single_led':
            linedata = [led_cm_1[0], led_cm_1[1], None, None, lum_1, None]

        return linedata

    @staticmethod
    def detect_motion(last_frame, current_frame, params):
        '''
        Detects position of a moving object.

        last_frame - grayscale image as (height x width) uint8 numpy array
        current_frame - grayscale image as (height x width) uint8 numpy array
        params - dict - {'smoothing box': int, 
                         'motion_threshold': int, 
                         'motion_size': int, 
                         'calibrationTmatrix': 3 x 3 numpy array}

        Returns a list with values (compatible with detect_leds() method):
            xcoord of moving object
            ycoord of moving object
            None
            None
            number of pixels above 'motion_threshold'
            None

            Values are None if not enough motion was detected.

        The absolute difference between two frames is smoothed with box kernel, 
        with size 'smoothing_box' x 'smoothing_box'. The resulting difference map is
        thresholded with 'motion_threshold'. If the resulting boolean array has
        more than 'motion_size' True values, it is used to compute the center of mass
        (Otherwise, None values are reported).
        The center of mass is the location of reported after converting from pixel space 
        to real space using 'calibrationTmatrix'.
        '''
        if last_frame is None:
            linedata = [None, None, None, None, None, None]
        else:
            frame_diff = cv2.absdiff(last_frame, current_frame)
            frame_diff = cv2.blur(frame_diff, ksize=(params['smoothing_box'], params['smoothing_box']))
            motion_idx = frame_diff > params['motion_threshold']
            y_idx, x_idx = np.where(motion_idx)
            num_motion_pix = len(y_idx)
            if num_motion_pix > params['motion_size']:
                pos_pix = (int(np.mean(x_idx)), int(np.mean(y_idx)))
                pos_cm = OnlineTracker.transform_pix_to_cm(pos_pix[::-1], params['calibrationTmatrix'])
                linedata = [pos_cm[0], pos_cm[1], None, None, num_motion_pix, None]
            else:
                linedata = [None, None, None, None, None, None]

        return linedata

    def send_data_with_ZMQpublisher(self, linedata):
        '''
        Publishes data with ZMQ.
        '''
        message = json.dumps(linedata)  # Convert data to string format
        message = message.encode()  # Convert data into bytes format
        self.ZMQpublisher.send(message) # Send the message using ZeroMQ

    def write_to_logfile(self, linedata):
        '''
        Writes a list of values into previously opened CSV file.
        '''
        self.logwriter.writerow(linedata)

    def process_motion(self, frame):
        '''
        Detects motion as difference between input frame and last frame.
        Overwrites self.last_frame variable to be used at next method call.

        frame - grayscale numpy array
        '''
        linedata = OnlineTracker.detect_motion(self.last_frame, frame, self.params)
        self.last_frame = copy(frame)

        return linedata

    def process(self, frame):
        '''
        Processes grayscale frame using detect_leds() method or process_motion() method.
        Passes output from detect_leds() to be sent via ZMQ and writtend to CSV file.
        '''
        if self.params['tracking_mode'] == 'dual_led' or self.params['tracking_mode'] == 'single_led':
            linedata = OnlineTracker.detect_leds(frame, self.params)
        elif self.params['tracking_mode'] == 'motion':
            linedata = self.process_motion(frame)
        else:
            raise ValueError('tracking_mode not recognized in params.')
        self.send_data_with_ZMQpublisher(linedata)
        self.csv_writer.write(linedata)

    def close(self):
        '''
        Also closes CSV file and ZMQ publisher.
        '''
        self.csv_writer.close()
        self.ZMQpublisher.close()


class TTLpulse_CameraTime_Writer(object):
    '''
    Writes camera current timestamps to csv file whenever TLL pulse rising edge detected.
    '''
    def __init__(self, camera, ttlPin=18):
        '''
        camera - picamera.PiCamera instance
        ttlPin - BCM numbering pin for detecting TTL pulses
        filename - CSV logfile name
        '''
        self.camera = camera
        self.ttlPin = ttlPin
        self.csv_writer = csv_writer('TTLpulseTimestamps.csv')
        # Initialize TTL edge detection
        self.piGPIO = pigpio.pi()
        self.piGPIOCallback = self.piGPIO.callback(self.ttlPin, pigpio.RISING_EDGE, self.write_time)

    def write_time(self, gpio, level, tick):
        '''
        Retrieves camera timestamp and writes it to the file.
        The latency between TTL pulse tick and system tick after querying camera timestamp is subtracted.
        '''
        currenttime = self.camera.timestamp
        tickDiff = pigpio.tickDiff(tick, self.piGPIO.get_current_tick())
        currenttime = currenttime - tickDiff
        self.csv_writer.write([currenttime])

    def close(self):
        self.piGPIOCallback.cancel()
        self.piGPIO.stop()
        self.csv_writer.close()


class SharedArrayQueue(object):
    '''
    Replicates Queue functionality for specified size and ctype of numpy arrays,
    for fast transfer of data between processes using pre-allocated shared memory.
    '''
    def __init__(self, ctype, array_shape, max_queue_length):
        '''
        ctype - ctypes of the arrays e.g. c_uint8.
        array_shape - tuple - dimensions of the array, e.g. (480, 720).
        max_queue_length - int - number of pre-allocated shared arrays.
            Note! Too high value for large arrays could cause Out Of Memory errors.
        '''
        self.ctype = ctype
        self.array_shape = array_shape
        self.manager = Manager()
        self.data_indices = list(range(max_queue_length))
        self.occupied_data_indices = self.manager.list()
        self.shared_arrays, self.shared_array_wrappers = SharedArrayQueue.create_shared_arrays(ctype, array_shape, max_queue_length)

    @staticmethod
    def create_shared_array(ctype, array_shape):
        '''
        Returns a multiprocessing.RawArray and its Numpy wrapper.
        '''
        numel = int(reduce(lambda x, y: x*y, array_shape))
        shared_array = RawArray(ctype, numel)
        shared_array_wrapper = np.frombuffer(shared_array, dtype=ctype).reshape(array_shape)

        return shared_array, shared_array_wrapper

    @staticmethod
    def create_shared_arrays(ctype, array_shape, max_queue_length):
        '''
        Arranges shared_array and shared_array_wrapper from create_shared_array 
        into lists of length max_queue_length.
        '''
        shared_arrays = []
        shared_array_wrappers = []
        for _ in range(max_queue_length):
            shared_array, shared_array_wrapper = SharedArrayQueue.create_shared_array(ctype, array_shape)
            shared_arrays.append(shared_array)
            shared_array_wrappers.append(shared_array_wrapper)

        return shared_arrays, shared_array_wrappers

    def put(self, data, block=False):
        '''
        Stores data into available location in shared memory if data is numpy.ndarray.
        If data is not numpy.ndarray, using get on this item in the queue will return string 'IncorrectItem'.
        If max_queue_length has been reached and block=False (default), an Exception is raised.
        If max_queue_length has been reached and block=True, method waits until space in queue is available.
        '''
        if isinstance(data, np.ndarray):
            # Find available data indices
            unavailable_data_indices = list(self.occupied_data_indices)
            available_data_indices = list(set(self.data_indices) - set(unavailable_data_indices))
            # Find single available index to use or if none available, raise Exception
            if len(available_data_indices) > 0:
                array_position = available_data_indices.pop(0)
                # Place array into shared array
                np.copyto(self.shared_array_wrappers[array_position], data)
                # Update available 
                self.occupied_data_indices.append(array_position)
            else:
                if block:
                    sleep(0.001)
                    self.put(data, block)
                else:
                    raise Exception('SharedArrayQueue pre-allocated memory full!')
        else:
            self.occupied_data_indices.append(-1)

    def get(self, timeout=None):
        '''
        Returns the next numpy.ndarray in the queue and waits until available.
        If anything else has been put into queue, get() returns 'IncorrectItem'.
        If timeout is specified and item is not available, get()
        waits until timeout seconds and returns 'TimeoutReached'.
        '''
        if not (timeout is None):
            timeout_start_time = time()
        while True:
            if len(self.occupied_data_indices) == 0:
                sleep(0.001)
            else:
                array_position = self.occupied_data_indices.pop(0)
                if array_position >= 0:
                    data = np.frombuffer(self.shared_arrays[array_position], dtype=self.ctype)
                    return data.reshape(self.array_shape)
                else:
                    return 'IncorrectItem'
            # Return None if timeout has been reached
            if not (timeout is None):
                if time() - timeout_start_time > timeout:
                    return 'TimeoutReached'

    def qsize(self):
        '''
        Returns the number of items currently in the queue.
        '''
        return len(self.occupied_data_indices)

    def join(self):
        '''
        Waits until all items have been acquired from the queue.
        '''
        while len(self.occupied_data_indices) > 0:
            sleep(0.001)


class RawYUV_Processor(object):
    '''
    Used by RawYUV_Output to process each incoming frame.
    Uses SharedArrayQueue to pass items into OnlineTracker in a separate process.
    '''
    def __init__(self, OnlineTrackerParams, frame_shape, monitor=True):
        '''
        OnlineTrackerParams - dict - input to prepare_OnlineTracker_params
        frame_shape - tuple - (height, width, n_channels) of incoming frames
        '''
        OnlineTrackerParams['frame_shape'] = frame_shape
        self.queue = SharedArrayQueue(c_uint8, frame_shape, 50)
        self.P_OnlineTracker_Process = Process(target=RawYUV_Processor.OnlineTracker_Process, 
                                                args=(OnlineTrackerParams, self.queue))
        self.P_OnlineTracker_Process.start()
        if monitor:
            self._MonitorLogger = RPiMonitorLogger(4, self.queue.qsize)

    @staticmethod
    def OnlineTracker_Process(params, queue):
        '''
        Uses OnlineTracker.process() on each item in the queue,
        until 'IncorrectItem' is received from queue.get().

        params - dict - parameters required by OnlineTracker.
        '''
        OT = OnlineTracker(params)
        while True:
            item = queue.get(timeout=0.01)
            if isinstance(item, np.ndarray):
                OT.process(item)
            elif isinstance(item, str) and item == 'IncorrectItem':
                break
        OT.close()

    def write(self, frame):
        '''
        Called by RawYUV_Output for each frame.
        '''
        self.queue.put(frame)

    def close(self):
        self.queue.put('STOP')
        self.queue.join()
        self.P_OnlineTracker_Process.join()
        if hasattr(self, '_MonitorLogger'):
            self._MonitorLogger.close()


class Calibrator(object):
    '''
    Performs operations using input frame and calibration_parameters.
    See description of methods:
        get_pattern()
        get_frame_with_pattern()
        get_calibrationTmatrix()
        get_calibration_data()
    '''
    def __init__(self, frame, calibration_parameters, pattern=None, calibrationTmatrix=None):
        '''
        frame - uint8 numpy array - cv2 RGB image
        calibration_parameters - dict - {'ndots_xy': tuple of ints - nr of dots along frame (height, width)
                                         'spacing': float - spacing of dots in centimeters
                                         'offset_xy_xy': tuple of floats - (offset on x axis, offset on 6 axis)}
        Optional to save computation time for some methods
            pattern - as output from get_pattern()
            calibrationTmatrix - as output from get_calibrationTmatrix()
        '''
        self.frame = frame
        self.ndots_xy = calibration_parameters['ndots_xy']
        self.spacing = calibration_parameters['spacing']
        self.offset_xy = calibration_parameters['offset_xy']
        self.calibrationTmatrix = calibrationTmatrix
        self.pattern = pattern

    def _detect_pattern(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        ret, pattern = cv2.findCirclesGrid(gray,(self.ndots_xy[0], self.ndots_xy[1]), flags=flags)
        if ret:
            self.pattern = pattern

        return ret

    def get_pattern(self):
        '''
        Returns dot pattern array if available or successfully detect.
        Returns None otherwise.
        '''
        if not (self.pattern is None):
            return self.pattern
        elif self._detect_pattern():
            return self.pattern

    def get_frame_with_pattern(self):
        if not (self.pattern is None):
            return cv2.drawChessboardCorners(self.frame, (self.ndots_xy[0], self.ndots_xy[1]), self.pattern, True)

    def _compute_calibrationTmatrix(self):
        # Generate object point values corresponding to the pattern
        objp = np.mgrid[0:self.ndots_xy[0],0:self.ndots_xy[1]].T.reshape(-1,2).astype(np.float32)
        objp[:,1] = objp[:,1] / 2
        shiftrows = np.arange(1,self.ndots_xy[1],2)
        for row in shiftrows:
            tmpidx = np.arange(row * self.ndots_xy[0], (row + 1) * self.ndots_xy[0])
            objp[tmpidx,0] = objp[tmpidx,0] + 0.5
        # Stretch the object point values to scale with the real pattern
        objp = objp * self.spacing
        # Add offset_xy from arena corner to get circle locations in the arena
        objp[:,0] = objp[:,0] + self.offset_xy[0]
        objp[:,1] = objp[:,1] + self.offset_xy[1]
        # Add the zeros to force pattern onto the plane in 3D world
        objp = np.concatenate((objp, np.zeros((objp.shape[0],1))), 1)
        # Compute transformation matrix
        self.calibrationTmatrix, mask = cv2.findHomography(self.pattern, objp, cv2.RANSAC,5.0)

    def get_calibrationTmatrix(self):
        '''
        Returns calibrationTmatrix array if available or pattern availalbe for computation.
        Returns None otherwise.
        '''
        if not (self.calibrationTmatrix is None):
            return self.calibrationTmatrix
        elif not (self.pattern is None):
            self._compute_calibrationTmatrix()
            return self.calibrationTmatrix

    def get_calibration_data(self):
        '''
        Returns a dictionary if data available or can be computed. Returns None otherwise.
        {'calibrationTmatrix' - as output from get_calibrationTmatrix()
        'pattern' - as output from get_pattern()
        'frame' - input frame}
        '''
        pattern = self.get_pattern()
        if not (pattern is None):
            calibrationTmatrix = self.get_calibrationTmatrix()
            return {'calibrationTmatrix': calibrationTmatrix, 'pattern': pattern, 'frame': self.frame}


class PiVideoEncoder_with_timestamps(picamera.PiVideoEncoder):
    '''
    picamera.PiVideoEncoder subclass that writes camera timestamp
    of each frame to file VideoEncoderTimestamps.csv
    '''
    def __init__(self, *args, **kwargs):
        super(PiVideoEncoder_with_timestamps, self).__init__(*args, **kwargs)
        self.csv_writer = csv_writer('VideoEncoderTimestamps.csv')

    def _callback_write(self, buf, **kwargs):

        if isinstance(buf, picamera.mmalobj.MMALBuffer):
            # for firmware >= 4.4.8
            flags = buf.flags
        else:
            # for firmware < 4.4.8
            flags = buf[0].flags

        if not (flags & mmal.MMAL_BUFFER_HEADER_FLAG_CONFIG):

            if flags & mmal.MMAL_BUFFER_HEADER_FLAG_FRAME_END:

                if buf.pts < 0:
                    # this usually happens if the video quality is set to
                    # a low value (= high quality). Try something in the range
                    # 20 to 25.
                    print("invalid time time stamp (buf.pts < 0):", buf.pts)

                self.csv_writer.write([buf.pts])

        return super(PiVideoEncoder_with_timestamps, self)._callback_write(buf, **kwargs)

    def close(self, *args, **kwargs):
        super(PiVideoEncoder_with_timestamps, self).close(*args, **kwargs)
        self.csv_writer.close()


class PiRawVideoEncoder_with_timestamps(picamera.PiRawVideoEncoder):
    '''
    picamera.PiRawVideoEncoder subclass that writes camera timestamp
    of each frame to file RawVideoEncoderTimestamps.csv
    '''
    def __init__(self, *args, **kwargs):
        super(PiRawVideoEncoder_with_timestamps, self).__init__(*args, **kwargs)
        self.csv_writer = csv_writer('RawVideoEncoderTimestamps.csv')

    def _callback_write(self, buf, **kwargs):

        if isinstance(buf, picamera.mmalobj.MMALBuffer):
            # for firmware >= 4.4.8
            flags = buf.flags
        else:
            # for firmware < 4.4.8
            flags = buf[0].flags

        if not (flags & mmal.MMAL_BUFFER_HEADER_FLAG_CONFIG):

            if flags & mmal.MMAL_BUFFER_HEADER_FLAG_FRAME_END:

                if buf.pts < 0:
                    # this usually happens if the video quality is set to
                    # a low value (= high quality). Try something in the range
                    # 20 to 25.
                    print("invalid time time stamp (buf.pts < 0):", buf.pts)

                self.csv_writer.write([buf.pts])

        return super(PiRawVideoEncoder_with_timestamps, self)._callback_write(buf, **kwargs)

    def close(self, *args, **kwargs):
        super(PiRawVideoEncoder_with_timestamps, self).close(*args, **kwargs)
        self.csv_writer.close()


class PiCamera_with_timestamps(picamera.PiCamera):
    '''
    This is a subclass of picamera.PiCamera to provide accurate timestamps for each frame.
    '''
    def __init__(self, *args, **kwargs):
        '''
        If TTLpulse_CameraTime_Writer=True, TTLpulse_CameraTime_Writer class
        is used to write the camera timestamp at each detected TTL pulse.
        '''
        Start_TTLpulse_CameraTime_Writer = kwargs.pop('TTLpulse_CameraTime_Writer', False)
        super(PiCamera_with_timestamps, self).__init__(*args, **kwargs)
        if Start_TTLpulse_CameraTime_Writer:
            self.TTLpulse_CameraTime_Writer = TTLpulse_CameraTime_Writer(self)
        self.ThisThingIsAlive = True

    def _get_video_encoder(self, *args, **kwargs):
        '''
        Provides an encoder class with timestamps for encoder_formats 'h264' and 'yuv'.
        In case of format 'mjpeg', picamera.PiCookedVideoEncoder is returned.
        Other formats raise ValueError.
        '''
        encoder_format = args[2]
        if encoder_format == 'h264':
            return PiVideoEncoder_with_timestamps(self, *args, **kwargs)
        elif encoder_format == 'yuv':
            return PiRawVideoEncoder_with_timestamps(self, *args, **kwargs)
        elif encoder_format == 'mjpeg':
            return picamera.PiCookedVideoEncoder(self, *args, **kwargs)
        else:
            raise ValueError('Incorrect encoder format requested from PiCamera_with_timestamps.')

    def stop_recording(self, *args, **kwargs):
        if hasattr(self, 'TTLpulse_CameraTime_Writer'):
            self.TTLpulse_CameraTime_Writer.close()
        super(PiCamera_with_timestamps, self).stop_recording(*args, **kwargs)

    def close(self, *args, **kwargs):
        super(PiCamera_with_timestamps, self).close(*args, **kwargs)
        self.ThisThingIsAlive = False


class RawYUV_Output(picamera.array.PiYUVAnalysis):
    '''
    Output for picamera.PiCamera.start_recording(format='yuv').
    Uses 
    '''

    def __init__(self, *args, **kwargs):
        '''
        RawYUV_Output_Processor keyword argument write() method is called for each frame.
        If no RawYUV_Output_Processor provided, frames are not processed.
        '''
        OnlineTrackerParams = kwargs.pop('OnlineTrackerParams', None)
        if not (OnlineTrackerParams is None):
            frame_shape = (kwargs['size'][1], kwargs['size'][0])
            self.RawYUV_Output_Processor = RawYUV_Processor(OnlineTrackerParams, frame_shape)
        super(RawYUV_Output, self).__init__(*args, **kwargs)

    def analyse(self, frame):
        if hasattr(self, 'RawYUV_Output_Processor'):
            grayscale_frame = frame[:,:,0]
            self.RawYUV_Output_Processor.write(grayscale_frame)

    def close(self, *args, **kwargs):
        super(RawYUV_Output, self).close(*args, **kwargs)
        if hasattr(self, 'RawYUV_Output_Processor'):
            self.RawYUV_Output_Processor.close()


class Stream_MJPEG_Output(object):
    '''
    Streams MJPEG data to an IP address
    '''
    def __init__(self, address, port):
        '''
        address - str - IP address where to send data.
        port - int - port number to use.
        '''
        self.client_socket = socket.socket()
        self.client_socket.connect((address, port))
        self.connection = self.client_socket.makefile('wb')
        self.stream = io.BytesIO()
        # Set variables for single frame grab
        self.grabbing_single_frame = False
        self.start_grabbing_single_frame = False

    def write(self, buf):
        if self.grabbing_single_frame:
            self.write_for_grab_frame(buf)
        if not self.grabbing_single_frame:
            if buf.startswith(b'\xff\xd8'):
                # Start of new frame; send the old one's length
                # then the data
                size = self.stream.tell()
                if size > 0:
                    self.connection.write(struct.pack('<L', size))
                    self.connection.flush()
                    self.stream.seek(0)
                    self.connection.write(self.stream.read(size))
                    self.stream.seek(0)
                    if self.start_grabbing_single_frame:
                        self.write_for_grab_frame(buf)
            self.stream.write(buf)

    def write_for_grab_frame(self, buf):
        if self.start_grabbing_single_frame:
            self.grabbing_single_frame = True
            self.start_grabbing_single_frame = False
            self.single_frame_stream.write(buf)
        else:
            if buf.startswith(b'\xff\xd8'):
                self.grabbing_single_frame = False
            else:
                self.single_frame_stream.write(buf)
        
    def grab_frame(self):
        '''
        Returns next full frame.
        '''
        self.single_frame_stream = io.BytesIO()
        self.start_grabbing_single_frame = True
        while self.start_grabbing_single_frame or self.grabbing_single_frame:
            sleep(0.05)
        self.single_frame_stream.seek(0)
        image = Image.open(self.single_frame_stream)
        frame = np.array(image)
        self.grabbing_single_frame = False

        return frame

    def close(self):
        self.connection.write(struct.pack('<L', 0))
        self.connection.flush()
        self.connection.close()
        self.client_socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


class Controller(object):
    '''
    Initializes PiCamera and allows control of recording.
    '''
    resolutions = {'low': (800, 608), 'high': (1600, 1216)}

    def __init__(self, resolution_option=None, framerate=30, OnlineTrackerParams=None):
        '''
        resolution - str - 'high' for (1600, 1216). Otherwise (800, 608) is used.
        OnlineTrackerParams - dict - see OnlineTracker input arguments.
        '''
        self.framerate = int(framerate)
        self._delete_old_files()
        self._init_camera(resolution_option)
        self.init_processing(OnlineTrackerParams)
        self.isRecording = False
        self.isProcessing = False
        self.isStreaming = False

    def _init_camera(self, resolution_option=None, warmup=2):
        '''
        Initializes camera with specififed settings and fixes gains.

        resolution - str - 'high' for (1600, 1216). Otherwise (800, 608) is used.
        '''
        self.camera = PiCamera_with_timestamps(clock_mode='raw', sensor_mode=4, framerate=self.framerate, 
                                               resolution=self._get_resolution(resolution_option), 
                                               TTLpulse_CameraTime_Writer=True)
        self.camera.awb_mode = 'auto'
        self.camera.exposure_mode = 'sports'
        self.camera.iso = 800
        self.camera.start_preview()
        sleep(warmup)
        gains = self.camera.awb_gains
        self.camera.exposure_mode = 'off'
        self.camera.awb_mode = 'off'
        self.camera.awb_gains = gains
        self.camera.shutter_speed = self.camera.exposure_speed
        self.camera.image_denoise = False
        self.camera.video_denoise = False
        self.camera.stop_preview()

    def _delete_old_files(self):
        if os.path.exists('video.h264'):
            os.remove('video.h264')

    def _get_resolution(self, setting):
        return self.resolutions[setting] if setting in list(self.resolutions.keys()) else self.resolutions['low']

    def init_processing(self, OnlineTrackerParams=None):
        if not (OnlineTrackerParams is None):
            self.RawYUV_Output = RawYUV_Output(self.camera, size=self.resolutions['low'], 
                                               OnlineTrackerParams=OnlineTrackerParams)
        else:
            self.RawYUV_Output = RawYUV_Output(self.camera, size=self.resolutions['low'])

    def grab_frame_with_capture(self, resolution=None):
        '''
        Returns frame requested resolution option or with video capture resolution as RGB numpy array.

        Note! May fail during streaming or recordings.
        '''
        resolution = resolution or self.camera.resolution
        frame_shape = (resolution[1], resolution[0], 3)
        output = np.empty((frame_shape[0] * frame_shape[1] * frame_shape[2],), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.camera.capture(output, format='rgb', use_video_port=True, resize=resolution)
        frame = output.reshape(frame_shape)

        return frame

    def grab_frame_from_MJPEG_stream(self, resolution=None):
        '''
        Returns frame requested resolution option or with video capture resolution as RGB numpy array.

        Is more reliable during streaming.
        '''
        resolution = resolution or self.camera.resolution
        frame_shape = (resolution[1], resolution[0], 3)
        frame = self.Stream_MJPEG_Output.grab_frame()
        if frame_shape != frame.shape:
            frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))

        return frame

    def grab_frame(self, resolution=None):
        if self.isStreaming:
            return self.grab_frame_from_MJPEG_stream(resolution)
        else:
            return self.grab_frame_with_capture(resolution)

    def calibrate(self, calibration_parameters, keep_frames={'low': True}):
        '''
        Grabs frames low' and 'high' if current camera resolution is 'high'. Else just 'low'.
        Attempts to calibrate using Calibrator and these frames and returns output if successful.
        Returns None if calibration unsuccessful.

        calibration_parameters - dict - as required by Calibrator
        keep_frames - dict - keys corresponding to output with bool values to specify if to return frames.
                    By default only 'low' resolution frame is returned.

        output = {'low':  Calibrator.get_calibration_data(), 
                  'high': Calibrator.get_calibration_data()}
        '''
        # Identify if calibration is necessary at both resolutions
        calibration_resolutions = self.resolutions
        # Get frames at calibration_resolutions
        frames = {}
        for key in calibration_resolutions.keys():
            frames[key] = self.grab_frame(calibration_resolutions[key])
            sleep(0.2) # Ensures there is not too long block on MJPEG capture
        # Calibrate each frame
        calibration = {}
        for key in frames.keys():
            calibrator = Calibrator(frames[key], calibration_parameters)
            calibration[key] = calibrator.get_calibration_data()
        # Return calibration if successful, otherwise None
        if not (None in calibration.values()):
            # Remove unwanted frames
            for key in calibration:
                if not (key in keep_frames and keep_frames[key] == True):
                    del calibration[key]['frame']

            return calibration

    def start_recording_video(self):
        self.isRecording = True
        if self.camera.resolution == self.resolutions['high']:
            quality = 27
            bitrate = 15000000
        else:
            quality = 23
            bitrate = 15000000
        self.camera.start_recording('video.h264', format='h264', splitter_port=1, 
                                    quality=quality, bitrate=bitrate)

    def start_processing(self):
        self.isProcessing = True
        if self.camera.resolution == self.resolutions['low']:
            self.camera.start_recording(self.RawYUV_Output, format='yuv', 
                                        splitter_port=2)
        else:
            self.camera.start_recording(self.RawYUV_Output, format='yuv', 
                                        splitter_port=2, 
                                        resize=self.resolutions['low'])

    def start_streaming(self, address='192.168.0.10', port=8000, resolution_option='low'):
        '''
        Starts MJPEG stream to Recording PC.
        '''
        self.isStreaming = True
        self.Stream_MJPEG_Output = Stream_MJPEG_Output(address=address, port=port)
        if self.camera.resolution == self.resolutions[resolution_option]:
            self.camera.start_recording(self.Stream_MJPEG_Output, format='mjpeg', 
                                        splitter_port=3)
        else:
            self.camera.start_recording(self.Stream_MJPEG_Output, format='mjpeg', 
                                        splitter_port=3, resize=self.resolutions[resolution_option])

    def stop_recording_video(self):
        self.camera.stop_recording(splitter_port=1)
        self.isRecording = False

    def stop_processing(self):
        self.camera.stop_recording(splitter_port=2)
        self.isProcessing = False

    def stop_streaming(self):
        self.camera.stop_recording(splitter_port=3)
        self.isStreaming = False

    def stop(self):
        if self.isRecording:
            self.stop_recording_video()
        if self.isProcessing:
            self.stop_processing()
            self.RawYUV_Output.close()
        if self.isStreaming:
            self.stop_streaming()
            self.Stream_MJPEG_Output.close()
            del self.Stream_MJPEG_Output

    def close(self):
        self.stop()
        self.camera.close()
        while self.camera.ThisThingIsAlive:
            sleep(0.05)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


def StartStop_Controller():
    with Controller() as controller:
        _ = input('Press enter to start video acquisition: ')
        controller.start_recording_video()
        _ = input('Press enter to stop video acquisition: ')
        controller.stop()


def main(args):
    if args.remote:
        if args.port:
            remote_controlled_class(Controller, block=True, port=args.port[0])
        else:
            raise ValueError('Port required for remote control.')
    else:
        StartStop_Controller()


if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Running this script initates Controller class.')
    parser.add_argument('--remote', action='store_true', 
                        help='Expects start and stop commands over ZMQ. Default is keyboard input.')
    parser.add_argument('--port', type=int, nargs=1, 
                        help='The port to use for ZMQ paired_messenger with Recording PC.')
    args = parser.parse_args()
    main(args)
