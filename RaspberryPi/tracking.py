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
import cPickle as pickle
from scipy.spatial.distance import euclidean
import sys
from multiprocessing import Manager, Process, RawArray
from copy import copy, deepcopy
from threading import Thread
from ZMQcomms import paired_messenger
import argparse
from shutil import rmtree
from Queue import Empty as QueueIsEmpty
from ctypes import c_uint8
import warnings
import io
import socket
import struct


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
            'RPi_number' - int
            'RPiIP' - str
            'pos_port' - str
            'smoothing_radius' - int
            'calibrationTmatrix' - array
            'doubleLED' - bool
            'LED_separation' - float
            'frame_shape' - tuple - (height, width, channels) of incoming frames
        '''
        self.RPi_number = params['RPi_number']
        params['LED_separation_pix'] = convert_centimeters_to_pixel_distance(params['LED_separation'], 
                                                                             params['calibrationTmatrix'], 
                                                                             params['frame_shape'])
        self.init_ZMQpublisher(params['RPiIP'], params['pos_port'])
        self.csv_writer = csv_writer('OnlineTrackerData.csv')
        params['cutout'] = self.create_circular_cutout(params)
        self.params = params

    def init_ZMQpublisher(self, RPiIP, pos_port):
        '''
        Sets up publishing messages with ZMQ at 'localhost'.
        '''
        posIP = 'tcp://' + RPiIP + ':' + pos_port
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
        all_indices = np.unravel_index(range(blank.size),blank.shape, order='F')
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
    def detect_leds(frame, params):
        '''
        Detects brightest point on grayscaled data after smoothing.
        If multiple LEDs in use, tries finding the second brithest spot in proximity of the first.
        Returns a list with values:
            xcoord of first LED
            ycoord of first LED
            xcoord of second LED
            ycoord of second LED
            luminance of first LED
            luminance of second LED

            Values are None for second LED if not requested or not found.
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert BGR data to grayscale
        gray = cv2.GaussianBlur(gray, (params['smoothing_radius'], params['smoothing_radius']), 0) # Smooth the image
        led_pix_1, led_cm_1, lum_1 = OnlineTracker.detect_first_led(gray, params['calibrationTmatrix'])
        if params['doubleLED']:
            led_pix_2, led_cm_2, lum_2 = OnlineTracker.detect_second_led(gray, params['calibrationTmatrix'], 
                                                                          led_pix_1, params['LED_separation_pix'], 
                                                                          params['cutout'])
            linedata = [led_cm_1[0], led_cm_1[1], led_cm_2[0], led_cm_2[1], lum_1, lum_2]
        else:
            linedata = [led_cm_1[0], led_cm_1[1], None, None, lum_1, None]

        return linedata

    def send_data_with_ZMQpublisher(self, linedata):
        '''
        Publishes data with ZMQ.
        '''
        message = json.dumps(linedata) # Convert data to string format
        self.ZMQpublisher.send(message) # Send the message using ZeroMQ

    def write_to_logfile(self, linedata):
        '''
        Writes a list of values into previously opened CSV file.
        '''
        self.logwriter.writerow(linedata)

    def process(self, frame):
        '''
        Processes YUV frame using detect_leds() method.
        Passes output from detect_leds() to be sent via ZMQ and writtend to CSV file.
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        linedata = OnlineTracker.detect_leds(frame, self.params)
        self.send_data_with_ZMQpublisher([self.RPi_number] + linedata)
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
    def __init__(self, camera, ttlPin=18, filename='TTLpulse_CameraTime.csv'):
        '''
        camera - picamera.PiCamera instance
        ttlPin - BCM numbering pin for detecting TTL pulses
        filename - CSV logfile name
        '''
        self.camera = camera
        self.ttlPin = ttlPin
        self.csv_writer = csv_writer(filename)
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


def RPiNumber():
    '''
    Returns RPiNumber from a file in the current working directory.
    '''
    with open('RPiNumber','r') as file:
        RPi_number = int(file.read().splitlines()[0])

    return RPi_number


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
        self.data_indices = range(max_queue_length)
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
        for n in range(max_queue_length):
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
    def __init__(self, OnlineTrackerParams, frame_shape):
        '''
        OnlineTrackerParams - dict - input to prepare_OnlineTracker_params
        frame_shape - tuple - (height, width, n_channels) of incoming frames
        '''
        OnlineTrackerParams['frame_shape'] = frame_shape
        self.queue = SharedArrayQueue(c_uint8, frame_shape, 50)
        self.P_OnlineTracker_Process = Process(target=RawYUV_Processor.OnlineTracker_Process, 
                                                args=(OnlineTrackerParams, self.queue))
        self.P_OnlineTracker_Process.start()

    @staticmethod
    def OnlineTracker_Process(params, queue):
        '''
        Uses OnlineTracker.process() on each item in the queue,
        until 'IncorrectItem' is received from queue.get().

        params - dict - parameters required by OnlineTracker.
        '''
        OT = OnlineTracker(params)
        while True:
            s_t = time()
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
        online_tracking = kwargs.pop('online_tracking', False)
        OnlineTrackerParams = kwargs.pop('OnlineTrackerParams', None)
        if online_tracking and not (OnlineTrackerParams is None):
            frame_shape = (kwargs['size'][1], kwargs['size'][0], 3)
            self.RawYUV_Output_Processor = RawYUV_Processor(OnlineTrackerParams, frame_shape)
        super(RawYUV_Output, self).__init__(*args, **kwargs)

    def analyse(self, frame):
        if hasattr(self, 'RawYUV_Output_Processor'):
            self.RawYUV_Output_Processor.write(frame)

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

    def write(self, buf):
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
        self.stream.write(buf)

    def close(self):
        self.connection.write(struct.pack('<L', 0))
        self.connection.close()
        self.client_socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


class CameraController(object):
    '''
    Initializes PiCamera and allows control of recording.
    '''
    resolutions = {'low': (800, 608), 'high': (1600, 1216)}

    def __init__(self, resolution_option=None, online_tracking=False, OnlineTrackerParams=None):
        '''
        resolution - str - 'high' for (1600, 1216). Otherwise (800, 608) is used.
        RawYUV_Processor - object - write() method is called for each YUV frame to be processed.
        '''
        self._init_camera(resolution_option)
        if online_tracking and not (OnlineTrackerParams is None):
            self.RawYUV_Output = RawYUV_Output(self.camera, size=self.resolutions['low'], 
                                               online_tracking=online_tracking, 
                                               OnlineTrackerParams=OnlineTrackerParams)
        else:
            self.RawYUV_Output = RawYUV_Output(self.camera, size=self.resolutions['low'])
        self._reset_available_ports()
        self.ThisThingIsAlive = True

    def _init_camera(self, resolution_option=None, warmup=2):
        '''
        Initializes camera with specififed settings and fixes gains.

        resolution - str - 'high' for (1600, 1216). Otherwise (800, 608) is used.
        '''
        self.camera = PiCamera_with_timestamps(clock_mode='raw', sensor_mode=4, framerate=30, 
                                               resolution=self._get_resolution(resolution_option), 
                                               TTLpulse_CameraTime_Writer=True)
        self.camera.awb_mode = 'auto'
        self.camera.exposure_mode = 'auto'
        self.camera.start_preview()
        sleep(warmup)
        gains = self.camera.awb_gains
        self.camera.awb_mode = 'off'
        self.camera.awb_gains = gains
        self.camera.shutter_speed = self.camera.exposure_speed
        self.camera.exposure_mode = 'off'

    def _get_resolution(self, setting):
        return self.resolutions[setting] if setting in self.resolutions.keys() else self.resolutions['low']

    def _reset_available_ports(self):
        self._available_ports = [1, 2, 3]

    def _get_available_port(self):
        return self._available_ports.pop(0)

    def start_processing(self):
        if self.camera.resolution == self.resolutions['low']:
            self.camera.start_recording(self.RawYUV_Output, format='yuv', 
                                        splitter_port=self._get_available_port())
        else:
            self.camera.start_recording(self.RawYUV_Output, format='yuv', 
                                        splitter_port=self._get_available_port(), 
                                        resize=self.resolutions['low'])

    def start_recording_video(self):
        self.camera.start_recording('video.h264', format='h264', 
                                    splitter_port=self._get_available_port(), quality=23)

    def start(self):
        '''
        Starts recording video and processing frames.
        '''
        self.start_recording_video()
        self.start_processing()

    def stop(self):
        self.camera.stop_recording()
        if hasattr(self, 'Stream_MJPEG_Output'):
            self.Stream_MJPEG_Output.close()
            del self.Stream_MJPEG_Output
        self._reset_available_ports()

    def start_streaming(self, address='192.168.0.10', port=8000):
        '''
        Starts MJPEG stream to Recording PC.
        '''
        self.Stream_MJPEG_Output = Stream_MJPEG_Output(address=address, port=port)
        if self.camera.resolution == self.resolutions['low']:
            self.camera.start_recording(self.Stream_MJPEG_Output, format='mjpeg', 
                                        splitter_port=self._get_available_port())
        else:
            self.camera.start_recording(self.Stream_MJPEG_Output, format='mjpeg', 
                                        splitter_port=self._get_available_port(), 
                                        resize=self.resolutions['low'])


    def grab_frame(self, resolution_option=None):
        '''
        Returns frame requested resolution option or with video capture resolution as BGR numpy array.
        '''
        resolution = resolution_option or self.camera.resolution
        frame_shape = (resolution[1], resolution[0], 3)
        output = np.empty((frame_shape[0] * frame_shape[1] * frame_shape[2],), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.camera.capture(output, format='bgr', use_video_port=True, resize=resolution)
        frame = output.reshape(frame_shape)

        return frame

    def close(self):
        self.camera.close()
        while self.camera.ThisThingIsAlive:
            sleep(0.05)
        self.RawYUV_Output.close()
        self.ThisThingIsAlive = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


class CameraRemoteController(object):
    '''
    Initializes ZMQ, CameraController and waits for commands to control CameraController.
    '''
    def __init__(self, port, **kwargs):
        '''
        port - int - ZMQ paired_messenger port.
        '''
        T_initialize_ZMQcomms = Thread(target=self.initialize_ZMQcomms, 
                                       args=(port,))
        T_initialize_ZMQcomms.start()
        with CameraController(**kwargs) as self.Controller:
            T_initialize_ZMQcomms.join()
            self.ZMQmessenger.sendMessage('init_successful')
            self.KeepCameraControllerAlive = True
            while self.KeepCameraControllerAlive:
                sleep(0.1)

    def initialize_ZMQcomms(self, port,  warmup=2):
        '''
        Initializes ZMQ communication with Recording PC.
        '''
        self.ZMQmessenger = paired_messenger(port=port)
        sleep(warmup) # This allows enough time for ZMQ protocols to be initiated
        self.command_parsing_in_progress = False
        self.ZMQmessenger.add_callback(self.command_parser)

    def command_parser(self, message):
        '''
        Parses incoming ZMQ message for function name, input arguments and calls that function.
        '''
        method_name = message.split(' ')[0]
        args = message.split(' ')[1:]
        getattr(self, method_name)(*args)
        if message != 'close':
            self.ZMQmessenger.sendMessage('Action Completed: ' + message)

    def start(self):
        self.Controller.start()

    def start_streaming(self, address, port):
        address = str(address)
        port = int(port)
        self.Controller.start_streaming(address, port)

    def stop(self):
        self.Controller.stop()

    def close(self):
        self.KeepCameraControllerAlive = False
        while self.Controller.ThisThingIsAlive:
            sleep(0.05)
        self.ZMQmessenger.sendMessage('Action Completed: close')
        self.ZMQmessenger.close()


def StartStop_CameraController(**kwargs):
    with CameraController(**kwargs) as Controller:
        _ = raw_input('Press enter to start image acquisition: ')
        Controller.start()
        _ = raw_input('Press enter to stop image acquisition: ')
        Controller.stop()

def load_settings():
    if os.path.isfile('TrackingSettings.p'):
        with open('TrackingSettings.p','rb') as file:
            TrackingSettings = pickle.load(file)
        # Get RPi Number
        params = {}
        params['RPi_number'] = RPiNumber()
        # Load the Calibration Matrix
        if str(params['RPi_number']) in TrackingSettings['calibrationData'].keys():
            calibrationTmatrix = TrackingSettings['calibrationData'][str(params['RPi_number'])]['calibrationTmatrix']
        else:
            raise ValueError('Calibration data does not exist for this RPi.')
        params['calibrationTmatrix'] = calibrationTmatrix
        # Identify if doubleLED tracking is enabled
        if TrackingSettings['LEDmode'] == 'double':
            params['doubleLED'] = True
        elif TrackingSettings['LEDmode'] == 'single':
            params['doubleLED'] = False
        # Get rest of the params
        params['smoothing_radius'] = TrackingSettings['smoothing_radius']
        params['pos_port'] = TrackingSettings['pos_port']
        params['RPiIP'] = TrackingSettings['RPiInfo'][str(params['RPi_number'])]['IP']
        params['LED_separation'] = TrackingSettings['LED_separation']

        return params


def main(args):
    kwargs = {}
    if args.tracking:
        OnlineTrackerParams = load_settings()
        if not (OnlineTrackerParams is None):
            kwargs['online_tracking'] = True
            kwargs['OnlineTrackerParams'] = OnlineTrackerParams
    if args.video_resolution:
        kwargs['resolution_option'] = args.video_resolution[0]
    if args.remote:
        if args.port:
            CameraRemoteController(args.port[0], **kwargs)
        else:
            raise ValueError('Port required for remote control.')
    else:
        StartStop_CameraController(**kwargs)


if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Running this script initates CameraController class.')
    parser.add_argument('--remote', action='store_true', 
                        help='Expects start and stop commands over ZMQ. Default is keyboard input.')
    parser.add_argument('--port', type=int, nargs=1, 
                        help='The port to use for ZMQ paired_messenger with Recording PC.')
    parser.add_argument('--tracking', action='store_true', 
                        help='Initializes RawYUV_Processor with TrackingSettings.p')
    parser.add_argument('--video_resolution', type=str, nargs=1, 
                        help='[high] for (1600, 1216). Otherwise (800, 608) is used.')
    args = parser.parse_args()
    main(args)
