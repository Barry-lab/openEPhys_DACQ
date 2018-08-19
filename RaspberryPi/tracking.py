import numpy as np
import picamera
import picamera.array
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
from multiprocessing import Process, RawArray, RawValue, Lock, Pipe
from ctypes import c_uint8, c_uint64
from copy import copy, deepcopy
from Queue import Queue
from threading import Thread
from ZMQcomms import paired_messenger
import argparse
from shutil import rmtree

# # Write output to a log file
# orig_stdout = sys.stdout
# sys.stdout = open('stdout', 'w')
# sys.stderr = open('stderr','w')

class process_with_queue(object):
    '''
    Base class for handling incoming data with a queue.

    When creating sub-classes from this base-class:
        Overwrite process() method, which otherwise raises NotImplementedError.
        When changing __init__() or close() methods, use:
            super(SubClassName, self).__init__()
            super(SubClassName, self).close()
    '''
    def __init__(self):
        self.init_queue()

    def process(self, *args, **kwargs):
        '''
        This method is called on any arguments passed to add_to_queue().
        '''
        raise NotImplementedError

    def queue_processor(self):
        '''
        Runs in a separate daemon thread and calls process() if anything in queue.
        '''
        while True:
            args, kwargs = self.queue.get()
            self.process(*args, **kwargs)
            self.queue.task_done()

    def init_queue(self):
        self.queue = Queue()
        T_queue_processor = Thread(target=self.queue_processor)
        T_queue_processor.setDaemon(True)
        T_queue_processor.start()

    def add_to_queue(self, *args, **kwargs):
        '''
        This adds input to processing queue, which 
        '''
        self.queue.put((args, kwargs))

    def close(self):
        '''
        Waits until all tasks added to the queue have been completed.
        '''
        self.queue.join()


class multiprocess_function(object):
    '''
    This class starts a process on a separate CPU core using multiprocessing module.
    It allows running a function on a separate process rapidly without spending time on
    creating the process or sending data to the process each time it is run.

    Each time run() method is called, the provided function is run just once.
    Run blocks until function call has been completed.
    Run returns the output from the function if it is not None.

    The function is called with the same arguments as used to instantiate multiprocess_function.

    Input arguments must be possible to pass with multiprocessing.Process() method as args.
    This means that only arguments that can be pickled, can be used.

    Data used by the function that can change any time it is run can be passed as
    shared shared memory or server processes, such as multiprocessing.Array().
    '''
    def __init__(self, function, *args, **kwargs):
        '''
        Input argument function is to be called with any input arguments that follow it.
        '''
        # Set up connection with the process in case data is returned
        self.processor_listener, processor_sender = Pipe(duplex=False)
        # This lock starts locked. When this lock is released, processor stops running.
        self.processor_Stopper_Lock = Lock()
        self.processor_Stopper_Lock.acquire()
        # This lock starts locked. When released, function is run and this lock is locked.
        self.processor_Start_Lock = Lock()
        self.processor_Start_Lock.acquire()
        # This lock starts locked. It is released as a run of function is finished and locked again by parent process.
        self.processor_End_Lock = Lock()
        self.processor_End_Lock.acquire()
        self.P_processor = Process(target=self.processor, 
                                       args=(function, args, kwargs, processor_sender, 
                                             self.processor_Stopper_Lock, 
                                             self.processor_Start_Lock, 
                                             self.processor_End_Lock))
        self.P_processor.start()

    @staticmethod
    def processor(function, args, kwargs, result_sender, Stopper_Lock, Start_Lock, End_Lock):
        '''
        This method is run in a separate process and controlled with run() and close() methods.
        '''
        while not Stopper_Lock.acquire(block=False):
            if Start_Lock.acquire(block=True, timeout=0.01):
                ret = function(*args, **kwargs)
                End_Lock.release()
                result_sender.send(ret)

    def run(self):
        '''
        Instructs processor() to call the function.
        Returns function output if not None.
        '''
        self.processor_Start_Lock.release()
        self.processor_End_Lock.acquire()
        ret = self.processor_listener.recv()
        if not (ret is None):
            return ret

    def close(self):
        '''
        Stops processor() method running in a seprarte process after any ongoing function call.
        '''
        self.processor_Stopper_Lock.release()
        self.P_processor.join()


class frame_writer(process_with_queue):
    '''
    Inherits process_with_queue functionality and instantiates multiprocess_function classes
    to store frames from the queue.

    Note that storing the images is an intensive process limited by disk writing speed.
    Increasing the image_quality (and thereby amount of data per frame) can reduce the maximum
    frames per second possible before long periods of data write gaps. These can cause issues
    with queues and parent processes.
    '''

    def __init__(self, resolution, n_processes=2, path_to_frames='frames', image_quality=85):
        '''
        Inherits __init__() from process_with_queue and adds initialization of frame writing.

        resolution - tuple - (frame_width,frame_height).
        n_processes - int - number of multiprocessing processes to use for writing frames (default is 2).
        path_to_frames - path for writing frames, default is 'frames' sub-folder in working directory.
        image_quality - int - between 1 and 100, higher being larger file size and higher quality (default is 85).
        '''
        super(frame_writer, self).__init__()
        self.clear_path_to_frames(path_to_frames)
        self.init_multiprocess_function(resolution, path_to_frames, image_quality)

    def clear_path_to_frames(self, path_to_frames):
        '''
        Ensures path_to_frames is an empty folder.
        '''
        path_to_frames = path_to_frames
        if os.path.exists(path_to_frames):
            rmtree(path_to_frames)
        os.mkdir(path_to_frames)

    @staticmethod
    def save_frame(path_to_frames, image_quality, frametime_RawArray, frame_RawArray, frame_shape):
        '''
        Stores frame as JPEG image file with frametime_RawArray as name.
        The limiting factor for the speed of this function is the writing speed of the device.
        Lower image_quality results in smaller file sizes, which allows for storing more frames per second.
        '''
        frame = np.frombuffer(frame_RawArray, dtype=c_uint8).reshape(frame_shape)
        frametime = int(frametime_RawArray.value)
        filename = os.path.join(path_to_frames, str(frametime) + '.jpg')
        cv2.imwrite(filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), image_quality])

    def init_multiprocess_function(self, resolution, path_to_frames, image_quality):
        '''
        Initializes multiprocess_function class with save_frame() method and shared states.
        '''
        # Create a shared memory array
        self.frametime_RawValue = RawValue(c_uint64)
        frame_shape = (resolution[1], resolution[0], 3)
        numel = int(frame_shape[0] * frame_shape[1] * frame_shape[2])
        frame_RawArray = RawArray(c_uint8, numel)
        self.frame_RawArray_Numpy_Wrapper = np.frombuffer(frame_RawArray, dtype=c_uint8).reshape(frame_shape)
        # Instantiate multiprocess_function
        self.multiprocess_function = multiprocess_function(self.save_frame, path_to_frames, image_quality, 
                                                           self.frametime_RawValue, 
                                                           frame_RawArray, frame_shape)

    def process(self, frame, frametime):
        '''
        Passes data to multiprocess_function class which runs save_frame.
        '''
        np.copyto(self.frame_RawArray_Numpy_Wrapper, frame)
        self.frametime_RawValue.value = int(frametime)
        self.multiprocess_function.run()

    def close(self):
        '''
        Inherits close() from process_with_queue and closes multiprocess_function class.
        '''
        super(frame_writer, self).close()
        self.multiprocess_function.close()


class online_tracker(process_with_queue):
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
            'LED_separation_pix' - int
            'resolution' - tuple - (frame_width,frame_height)
        '''
        super(online_tracker, self).__init__()
        self.RPi_number = params['RPi_number']
        self.init_ZMQpublisher(params['RPiIP'], params['pos_port'])
        self.init_logfile_writer()
        params['cutout'] = self.create_circular_cutout(params)
        self.init_multiprocess_function(params)

    def init_ZMQpublisher(self, RPiIP, pos_port):
        '''
        Sets up publishing messages with ZMQ at 'localhost'.
        '''
        posIP = 'tcp://' + RPiIP + ':' + pos_port
        # For sending position data
        contextpub = zmq.Context()
        self.ZMQpublisher = contextpub.socket(zmq.PUB)
        self.ZMQpublisher.bind(posIP)
        sleep(0.1) # Give time to establish sockets for ZeroMQ

    def init_logfile_writer(self, filename='logfile.csv'):
        '''
        Sets up CSV file writer.
        '''
        if os.path.exists(filename):
            os.remove(filename)
        self.logfile = open(filename, 'a')
        self.logwriter = csv.writer(self.logfile)

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
        xy_cm = online_tracker.transform_pix_to_cm(led_pix, calibrationTmatrix)

        return led_pix, xy_cm, lum

    @staticmethod
    def create_circular_cutout(params):
        '''
        Creates a blank array and indexing arrays to fill it with values from image
        '''
        # Extract necessary values from params
        cutout_radius = params['LED_separation_pix'] * 2
        img_shape = (params['resolution'][1], params['resolution'][0])
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
        shifted_cutout = online_tracker.center_and_crop_cutout(cutout, center_pix)
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
        led_pix, lum = online_tracker.detect_max_luminance_in_circular_area(gray, led_pix_1, cutout)
        xy_cm = online_tracker.transform_pix_to_cm(led_pix, calibrationTmatrix)

        return led_pix, xy_cm, lum

    @staticmethod
    def detect_leds(frame_RawArray, frame_shape, params):
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
        frame = np.frombuffer(frame_RawArray, dtype=c_uint8).reshape(frame_shape)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert BGR data to grayscale
        gray = cv2.GaussianBlur(gray, (params['smoothing_radius'], params['smoothing_radius']), 0) # Smooth the image
        led_pix_1, led_cm_1, lum_1 = online_tracker.detect_first_led(gray, params['calibrationTmatrix'])
        if params['doubleLED']:
            led_pix_2, led_cm_2, lum_2 = online_tracker.detect_second_led(gray, params['calibrationTmatrix'], 
                                                                          led_pix_1, params['LED_separation_pix'], 
                                                                          params['cutout'])
            linedata = [led_cm_1[0], led_cm_1[1], led_cm_2[0], led_cm_2[1], lum_1, lum_2]
        else:
            linedata = [led_cm_1[0], led_cm_1[1], None, None, lum_1, None]

        return linedata

    def init_multiprocess_function(self, params):
        '''
        Initializes multiprocess_function class with detect_leds() method and shared state.
        '''
        # Create a shared memory array
        frame_shape = (params['resolution'][1], params['resolution'][0], 3)
        numel = int(frame_shape[0] * frame_shape[1] * frame_shape[2])
        frame_RawArray = RawArray(c_uint8, numel)
        self.frame_RawArray_Numpy_Wrapper = np.frombuffer(frame_RawArray, dtype=c_uint8).reshape(frame_shape)
        # Instantiate multiprocess_function
        self.multiprocess_function = multiprocess_function(self.detect_leds, frame_RawArray, 
                                                           frame_shape, params)

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

    def process(self, frame, frametime):
        '''
        Passes data to multiprocess_function for running detect_leds() method.
        Passes output from detect_leds() to be sent via ZMQ and writtend to CSV file.
        '''
        np.copyto(self.frame_RawArray_Numpy_Wrapper, frame)
        linedata = self.multiprocess_function.run()
        linedata = [self.RPi_number, frametime] + linedata
        self.send_data_with_ZMQpublisher(linedata)
        self.write_to_logfile(linedata)

    def close(self):
        '''
        Inherits close() from process_with_queue and closes multiprocess_function class.
        Also closes CSV file and ZMQ publisher.
        '''
        super(online_tracker, self).close()
        self.multiprocess_function.close()
        self.logfile.close()
        self.ZMQpublisher.close()


class Frame_Handler(process_with_queue):
    '''
    Inherits process_with_queue functionality and instantiates online_tracker class and
    frame_writer if save_frames is True. Frame_Handler makes copies of any incoming data
    and passes them on to next queues in online_tracker and frame_writer.
    '''

    def __init__(self, params):
        '''
        Inherits __init__() from process_with_queue and adds initialization of online_tracker and frame_writer.
        params - dict - must contain all parameters necessary for online_tracker class and:
            'save_frames' - bool - indicates if frame_writer is used.
        '''
        super(Frame_Handler, self).__init__()
        self.save_frames = params['save_frames']
        self.online_tracker = online_tracker(params)
        if self.save_frames:
            self.frame_writer = frame_writer(params['resolution'])

    def process(self, frame, frametime):
        '''
        Makes copies of incoming data and passes them to online_tracker and frame_writer.
        '''
        self.online_tracker.add_to_queue(copy(frame), copy(frametime))
        if self.save_frames:
            self.frame_writer.add_to_queue(copy(frame), copy(frametime))

    def close(self):
        '''
        Inherits close() from process_with_queue and closes online_tracker and frame_writer.
        '''
        super(Frame_Handler, self).close()
        self.online_tracker.close()
        if self.save_frames:
            self.frame_writer.close()


class PiCameraOutput(picamera.array.PiRGBAnalysis):
    '''
    This class can be used as output for picamera.PiCamera.start_recording().
    It grabs GPU clock times for synchronization and passes them with each frame to a queue in
    Frame_Handler class, which takes care of the handling and processing of frames.

    Incoming frames are ignored until start() method is called.
    '''

    def __init__(self, Frame_Handler_Params, *args, **kwargs):
        '''
        Initializes TTL signalling and Frame_Handler.

        Frame_Handler_Params - dict - must contain all parameters necessary for Frame_Handler.
        '''
        super(PiCameraOutput, self).__init__(*args, **kwargs)
        self.frame_handler = Frame_Handler(Frame_Handler_Params)
        self.Use_PiCamera_Stream = False
        self.analyse_method_active = False

    def analyse(self, frame):
        '''
        Grabs camera GPU timestamps and passes each frame to Frame_Handler queue.
        '''
        if self.Use_PiCamera_Stream:
            self.analyse_method_active = True
            # Each time a frame is catpured, this function is called on that frame
            frametime = copy(self.camera.frame.timestamp) # Get timestamp of frame capture (PTS) in RPi camera time
            self.frame_handler.add_to_queue(frame, frametime)
            self.analyse_method_active = False

    def start(self):
        '''
        Makes analyse() process incoming frames.
        '''
        self.Use_PiCamera_Stream = True

    def close(self, *args, **kwargs):
        '''
        Closes all child classes and processes in correct order.
        '''
        super(PiCameraOutput, self).close(*args, **kwargs)
        self.Use_PiCamera_Stream = False # Ensures no further frames are sent to analyse
        while self.analyse_method_active: # Ensures all frames in analyse have been passed to self.frame_handler
            sleep(0.01)
        self.frame_handler.close()


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
        self.init_logfile_writer(filename)
        # Initialize TTL edge detection
        self.piGPIO = pigpio.pi()
        self.piGPIOCallback = self.piGPIO.callback(self.ttlPin, pigpio.RISING_EDGE, self.write_time)

    def init_logfile_writer(self, filename):
        '''
        Sets up CSV file writer.
        '''
        if os.path.exists(filename):
            os.remove(filename)
        self.logfile = open(filename, 'a')
        self.logwriter = csv.writer(self.logfile)

    def write_time(self, gpio, level, tick):
        '''
        Retrieves camera timestamp and writes it to the file.
        '''
        currenttime = copy(self.camera.timestamp)
        self.logwriter.writerow([currenttime])

    def close(self):
        self.piGPIOCallback.cancel()
        self.piGPIO.stop()
        self.logfile.close()


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

    def stop_recording(self, *args, **kwargs):
        if hasattr(self, 'TTLpulse_CameraTime_Writer'):
            self.TTLpulse_CameraTime_Writer.close()
        super(PiCamera_with_timestamps, self).stop_recording(*args, **kwargs)


def RPiNumber():
    '''
    Returns RPiNumber from a file in the current working directory.
    '''
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
    '''
    Controls RPi camera and frame processing.
    '''

    def __init__(self, TrackingSettings=None, remoteControl=False):
        '''
        TrackingSettings - dict - as created by CameraSettingsGUI.
        remoteControl - bool - if True, start and stop commands are expected via ZMQ. Otherwise, raw_input.
        '''
        if TrackingSettings is None:
            with open('TrackingSettings.p','rb') as file:
                TrackingSettings = pickle.load(file)
        self.remoteControl = remoteControl
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
        self.Frame_Handler_Params = self.prepare_Frame_Handler_Params(RPiNumber(), TrackingSettings)
        # Specify framerate based on how fast Frame_Handler processes a single frame
        if self.Frame_Handler_Params['save_frames']:
            self.framerate = 25
        else:
            self.framerate = 30

    def initialize_ZMQcomms(self, port):
        '''
        Initializes ZMQ communication with Recording PC.
        '''
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

    @staticmethod
    def prepare_Frame_Handler_Params(RPi_Nr, TrackingSettings):
        '''
        Parses TrackingSettings to produce specific parameters required by Frame_Handler.
        '''
        Frame_Handler_Params = {}
        Frame_Handler_Params['RPi_number'] = RPi_Nr
        # Load the Calibration Matrix
        if str(RPi_Nr) in TrackingSettings['calibrationData'].keys():
            calibrationTmatrix = TrackingSettings['calibrationData'][str(RPi_Nr)]['calibrationTmatrix']
        else:
            raise ValueError('Calibration data does not exist for this RPi.')
        Frame_Handler_Params['calibrationTmatrix'] = calibrationTmatrix
        # Identify if doubleLED tracking is enabled
        if TrackingSettings['LEDmode'] == 'double':
            Frame_Handler_Params['doubleLED'] = True
        elif TrackingSettings['LEDmode'] == 'single':
            Frame_Handler_Params['doubleLED'] = False
        # Find values for distances between LED
        LED_separation_pix = convert_centimeters_to_pixel_distance(TrackingSettings['LED_separation'], 
                                                                   Frame_Handler_Params['calibrationTmatrix'], 
                                                                   TrackingSettings['resolution'])
        Frame_Handler_Params['LED_separation_pix'] = LED_separation_pix
        # Other parameters
        Frame_Handler_Params['resolution'] = TrackingSettings['resolution']
        Frame_Handler_Params['save_frames'] = TrackingSettings['save_frames']
        Frame_Handler_Params['smoothing_radius'] = TrackingSettings['smoothing_radius']
        Frame_Handler_Params['pos_port'] = TrackingSettings['pos_port']
        Frame_Handler_Params['RPiIP'] = TrackingSettings['RPiInfo'][str(RPi_Nr)]['IP']

        return Frame_Handler_Params

    def start(self):
        self.start_image_acquisition = True

    def stop(self):
        self.stop_image_acquisition = True

    def run_Camera(self):
        '''
        Uses picamera.PiCamera.camera and PiCameraOutput to capture and process camera feed, respectively.
        If remoteControl is True, camera feed capture and processing is started and stopped with ZMQ messages.
        Otherwise, with user pressing Enter.
        '''
        with PiCamera_with_timestamps(clock_mode='raw', framerate=self.framerate, TTLpulse_CameraTime_Writer=True) as camera: # Initializes the RPi camera module
            camera.resolution = tuple(self.resolution) # Set frame capture resolution
            camera.exposure_mode = self.exposure_setting
            camera.iso = self.camera_iso # Set Camera ISO value (sensitivity to light)
            camera.shutter_speed = self.shutter_speed # Set camera shutter speed
            camera.awb_mode = 'auto'
            with PiCameraOutput(self.Frame_Handler_Params, camera) as output: # Initializes the PiCameraOutput class
                camera.start_recording(output, format='bgr') # Initializes the camera
                sleep(2) # Let the camera warm up
                # Fix gains
                gains = camera.awb_gains
                camera.awb_mode = 'off'
                camera.awb_gains = gains
                camera.shutter_speed = camera.exposure_speed
                camera.exposure_mode = 'off'
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
                # Stop recording
                camera.stop_recording()
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
