### These functions allow interfacing with the Raspberry Pis.

### By Sander Tanni, January 2018, UCL

import zmq
import multiprocessing
from time import time, sleep
from openEPhys_DACQ.sshScripts import ssh
import json
from threading import Lock, Thread
import numpy as np
from scipy.spatial.distance import euclidean
from openEPhys_DACQ.TrackingDataProcessing import combineCamerasData
from openEPhys_DACQ.ZMQcomms import paired_messenger, remote_object_controller
from multiprocessing.dummy import Pool as ThreadPool
from tempfile import mkdtemp
from shutil import rmtree
import os
from openEPhys_DACQ.HelperFunctions import test_pinging_address, time_string
from openEPhys_DACQ.package_configuration import package_path


def write_files_to_RPi(files, address, username='pi', verbose=False):
    """
    files - list or tuple of files to copy to home directory on the RPi
    address - RPi address
    username - RPi username to use for access
    """
    if verbose:
        qflag = ''
    else:
        qflag = '-q '
    RPi_network_path = username + '@' + address
    for file in files:
        callstr = 'scp ' + qflag + file + ' ' + RPi_network_path + ':/home/pi'
        _ = os.system(callstr)

def read_file_from_RPi(source_filename, target_filename, address, username='pi', verbose=False):
    if verbose:
        qflag = ''
    else:
        qflag = '-q '
    RPi_network_path = username + '@' + address
    src = RPi_network_path + ':/home/pi/' + source_filename
    callstr = 'scp ' + qflag + src + ' ' + target_filename
    _ = os.system(callstr)

def read_files_from_RPi(files, target_path, address, username='pi', verbose=False):
    """
    files - list or tuple of files to copy from home directory of the RPi
    target_path - folder to which the files will be copied to
    address - RPi address
    username - RPi username to use for access
    """
    for file in files:
        target_filename = os.path.join(target_path, file)
        read_file_from_RPi(file, target_filename, address, username, verbose)


class Camera_RPi_file_manager(object):
    def __init__(self, address, username='pi', verbose=False):
        self.address = address
        self.username = username
        self.verbose = verbose

    @staticmethod
    def Camera_RPi_files():
        return (os.path.join(package_path, 'ZMQcomms.py'),
                os.path.join(package_path, 'CameraRPiController.py'))

    @staticmethod
    def video_file_name_on_RPi():
        return 'video.h264'

    @staticmethod
    def video_file_name_on_RecordingPC(cameraID):
        return 'video_' + cameraID + '.h264'

    def update_files_on_RPi(self):
        write_files_to_RPi(Camera_RPi_file_manager.Camera_RPi_files(), self.address, self.username, self.verbose)

    def retrieve_timestamps(self):
        print(['DEBUG', time_string(), 'Retrieving timestamps from ' + self.address])
        temp_folder = mkdtemp('RPiTempFolder')
        files = ('RawVideoEncoderTimestamps.csv', 
                 'VideoEncoderTimestamps.csv', 
                 'TTLpulseTimestamps.csv')
        read_files_from_RPi(files, temp_folder, self.address, self.username, self.verbose)
        filename = os.path.join(temp_folder, files[0])
        self.OnlineTrackerData_timestamps = np.genfromtxt(filename, dtype=int)
        filename = os.path.join(temp_folder, files[1])
        self.VideoData_timestamps = np.genfromtxt(filename, dtype=int)
        filename = os.path.join(temp_folder, files[2])
        self.GlobalClock_timestamps = np.genfromtxt(filename, dtype=int)
        rmtree(temp_folder)
        print(['DEBUG', time_string(), 'Retrieving timestamps from ' + self.address + ' complete.'])

    def retrieve_OnlineTrackerData(self):
        """
        Retrieves OnlineTrackerData. The data transfer failes sometimes, 
        in which case an informing message is printed and retrieval is attempted again.

        Note! OnlineTrackerData_timestamps must be obtained first using retrieve_timestamps() method. 
        """
        print(['DEBUG', time_string(), 'Retrieving tracking data from ' + self.address])
        if hasattr(self, 'OnlineTrackerData_timestamps'):
            temp_folder = mkdtemp('RPiTempFolder')
            self.OnlineTrackerData = np.zeros((0, 0))
            while self.OnlineTrackerData.shape[0] != self.OnlineTrackerData_timestamps.size:
                try:
                    print(['DEBUG', time_string(), 'Retrieving tracking data from ' + self.address + ' attempt'])
                    read_files_from_RPi(('OnlineTrackerData.csv',), temp_folder, self.address, self.username, self.verbose)
                    filename = os.path.join(temp_folder, 'OnlineTrackerData.csv')
                    self.OnlineTrackerData = np.genfromtxt(filename, delimiter=',', dtype=float)
                except ValueError:
                    print('Failed to get OnlineTrackerData, trying again at: ' + self.address)
            rmtree(temp_folder)
        else:
            raise Exception('OnlineTrackerData_timestamps must be obtained first.')
        print(['DEBUG', time_string(), 'Retrieving tracking data from ' + self.address + ' complete.'])

    def retrieve_timestamps_and_OnlineTrackerData(self):
        """
        Must be called before get_timestamps_and_OnlineTrackerData()
        """
        self.retrieve_timestamps()
        self.retrieve_OnlineTrackerData()

    def get_timestamps_and_OnlineTrackerData(self, cameraID='0'):
        """
        Must be called after retrieve_timestamps_and_OnlineTrackerData()
        """
        return {'ColumnLabels': ['X1', 'Y1', 'X2', 'Y2', 'Luminance_1', 'Luminance_2'], 
                'OnlineTrackerData': self.OnlineTrackerData,
                'OnlineTrackerData_timestamps': self.OnlineTrackerData_timestamps,
                'VideoData_timestamps': self.VideoData_timestamps,
                'GlobalClock_timestamps': self.GlobalClock_timestamps, 
                'VideoFile': Camera_RPi_file_manager.video_file_name_on_RecordingPC(cameraID)}

    def copy_over_video_data(self, folder_path, cameraID='0'):
        """
        Copies over video data to folder_path with cameraID included in the filename.
        """
        print(['DEBUG', time_string(), 'Retrieving video data from ' + self.address])
        read_file_from_RPi(Camera_RPi_file_manager.video_file_name_on_RPi(), 
                           os.path.join(folder_path, Camera_RPi_file_manager.video_file_name_on_RecordingPC(cameraID)), 
                           self.address, self.username, self.verbose)
        print(['DEBUG', time_string(), 'Retrieving video data from ' + self.address + ' complete.'])


class CameraControl(object):
    def __init__(self, address, port, username='pi', password='raspberry', 
                 resolution_option=None, OnlineTrackerParams=None, framerate=30):
        # Make sure files on RPi are up to date
        self.RPi_file_manager = Camera_RPi_file_manager(address, username)
        self.RPi_file_manager.update_files_on_RPi()
        # Start initiation protocol and repeat if failed
        init_successful = False
        attempts = 0
        while not init_successful:
            if attempts > 4:
                reboot = True
                attempts = 0
            else:
                reboot = False
            init_successful = self._init_CameraController(address, port, username, password, 
                                                          resolution_option, OnlineTrackerParams, 
                                                          framerate, reboot)
            if not init_successful:
                attempts += 1
                print('Remote CameraController initialization failed, trying again at: ' + address)

    def _init_CameraController(self, address, port, username, password, 
                               resolution_option, OnlineTrackerParams, framerate, reboot=False):
        if hasattr(self, 'RemoteControl'):
            self.RemoteControl.close()
        self.RemoteControl = remote_object_controller(address, int(port))
        self.RPiSSH = self._init_RPiSSH(address, port, username, password, reboot)
        init_successful = self.RemoteControl.pair(10)
        if init_successful:
            init_successful = self.RemoteControl.sendInitCommand(10, 
                                                                 resolution_option, 
                                                                 framerate, 
                                                                 OnlineTrackerParams)

        return init_successful

    def _init_RPiSSH(self, address, port, username='pi', password='raspberry', reboot=False):
        if not hasattr(self, 'RPiSSH') or self.RPiSSH is None:
            self.RPiSSH = ssh(address, username, password)
        if reboot:
            print('Rebooting at ' + address)
            self.RPiSSH.sendCommand('sudo reboot')
            self.RPiSSH.disconnect()
            self.RPiSSH = None
            sleep(30) # Assumed minimum rebooting time for Raspberry Pi
            pinging_successful = False
            while not pinging_successful:
                pinging_successful = test_pinging_address(address)
            self._init_RPiSSH(address, port, username, password, reboot=False)
        else:
            self.RPiSSH.sendCommand('sudo pkill python') # Ensure any past processes have closed
            command = 'source /home/pi/.virtualenvs/python3/bin/activate && '
            command += 'python CameraRPiController.py --remote --port ' + str(port)
            self.RPiSSH.sendCommand_threading(command) # TO DO: This seems to kill the ssh instance if command fails.

    def calibrate(self, calibration_parameters):
        """
        calibration_parameters - dict - output from CameraControl.enter_calibration_parameters
        """
        return self.RemoteControl.sendCommand('calibrate', True, calibration_parameters)

    def start_streaming(self, address, port):
        self.RemoteControl.sendCommand('start_streaming', True, address, int(port))

    def start_processing(self):
        self.RemoteControl.sendCommand('start_processing', True)

    def start(self):
        self.RemoteControl.sendCommand('start_recording_video', True)
        self.RemoteControl.sendCommand('start_processing', True)

    def stop(self):
        self.RemoteControl.sendCommand('stop', True)

    def close(self):
        self.RemoteControl.sendCommand('close', True)
        self.RemoteControl.close()
        if hasattr(self, 'RPiSSH') and not (self.RPiSSH is None):
            self.RPiSSH.disconnect()

    @staticmethod
    def dict_OnlineTrackerParams(calibrationTmatrix, tracking_mode, smoothing_box, 
                                 motion_threshold, motion_size, 
                                 OnlineTracker_port, RPiIP, LED_separation):
        """
        Returns camera_parameters in a dictionary.
        """
        return {'calibrationTmatrix': calibrationTmatrix, # (3 x 3 numpy array)
                'tracking_mode': str(tracking_mode), # str
                'smoothing_box': int(smoothing_box), # int - number of pixels
                'motion_threshold': motion_threshold,  # int
                'motion_size': motion_size,  # int
                'OnlineTracker_port': str(OnlineTracker_port), # str
                'RPiIP': str(RPiIP), # str
                'LED_separation': float(LED_separation)} # LED separation in cm

    @staticmethod
    def dict_calibration_parameters(ndots_xy, spacing, offset_xy):
        """
        Returns calibration_parameters in a dictionary.
        """
        return {'ndots_xy': ndots_xy, 
                'spacing': spacing, 
                'offset_xy': offset_xy}

    @staticmethod
    def CameraControl_unpackPool(kwargs):
        """
        Helper function to allow using pool with argument list.
        kwargs - dict - key value pairs for input arguments to CameraControl
        """
        return CameraControl(**kwargs)

    @staticmethod
    def init_CameraControls_with_CameraSettings(CameraSettings):
        """
        Returns a dictionary where each CameraSettings['use_RPi_nrs'] element is a key
        to corresponding initialized CameraControl object. 
        """
        kwargs_list = []
        cameraID_list = []
        for cameraID in CameraSettings['CameraSpecific'].keys():
            cameraID_list.append(cameraID)
            args = (CameraSettings['CameraSpecific'][cameraID]['CalibrationData']['low']['calibrationTmatrix'], 
                    CameraSettings['General']['tracking_mode'], 
                    CameraSettings['General']['smoothing_box'], 
                    CameraSettings['General']['motion_threshold'], 
                    CameraSettings['General']['motion_size'], 
                    CameraSettings['General']['OnlineTracker_port'], 
                    CameraSettings['CameraSpecific'][cameraID]['address'], 
                    CameraSettings['General']['LED_separation'])
            OnlineTrackerParams = CameraControl.dict_OnlineTrackerParams(*args)
            kwargs_list.append({'address': CameraSettings['CameraSpecific'][cameraID]['address'], 
                                'port': CameraSettings['General']['ZMQcomms_port'], 
                                'username': CameraSettings['General']['username'], 
                                'password': CameraSettings['General']['password'], 
                                'resolution_option': CameraSettings['General']['resolution_option'], 
                                'framerate': CameraSettings['General']['framerate'], 
                                'OnlineTrackerParams': OnlineTrackerParams
                                })
        pool = ThreadPool(len(kwargs_list))
        CameraControlsList = pool.map(CameraControl.CameraControl_unpackPool, kwargs_list)
        pool.close()
        CameraControls = {}
        for camera_controller, cameraID in zip(CameraControlsList, cameraID_list):
            CameraControls[cameraID] = camera_controller

        return CameraControls


class TrackingControl(object):

    def __init__(self, CameraSettings):
        self.CameraControls = CameraControl.init_CameraControls_with_CameraSettings(CameraSettings)

    def start(self):
        T_list = []
        for cameraID in self.CameraControls.keys():
            T = Thread(target=self.CameraControls[cameraID].start)
            T.start()
        for T in T_list:
            T.join()

    def stop(self):
        T_list = []
        for cameraID in self.CameraControls.keys():
            T = Thread(target=self.CameraControls[cameraID].stop)
            T.start()
        for T in T_list:
            T.join()

    def close(self):
        T_list = []
        for cameraID in self.CameraControls.keys():
            T = Thread(target=self.CameraControls[cameraID].close)
            T.start()
        for T in T_list:
            T.join()


class onlineTrackingData(object):
    # Constantly updates position data for all RPis currently in use.
    # Initialize this class as RPIpos = onlineTrackingData(CameraSettings, arena_size)
    # Check for latest position with combPos = RPIpos.combPosHistory[-1]

    # RPIpos.combPosHistory is multiprocessing.managers.List and can take care of locks across processes

    # Optional arguments during initialization:
    #   HistogramParameters is a list: [margins, binSize, histogram_speed_limit]
    def __init__(self, CameraSettings, arena_size, HistogramParameters=None):
        # Initialise the class with input CameraSettings
        self.combPos_update_interval = 0.05 # in seconds
        self.CameraSettings = CameraSettings
        self.arena_size = arena_size
        # Make a list of cameraIDs from CameraSettings for reference to other lists in this class.
        self.cameraIDs = sorted(self.CameraSettings['CameraSpecific'].keys())
        if HistogramParameters is None:
            HistogramParameters = {'margins': 10,  # histogram data margins in centimeters
                                   'binSize': 2,  # histogram binSize in centimeters
                                   'speedLimit': 10}  # centimeters of distance in last second to be included
        self.KeepGettingData = True # Set True for endless while loop of updating latest data
        self.posDatas = [None for i in range(len(self.cameraIDs))]
        self.multiprocess_manager = multiprocessing.Manager()
        self.combPosHistory = self.multiprocess_manager.list([])
        self.sockSUBs = onlineTrackingData.setupSockets(self.cameraIDs, self.CameraSettings)
        # Initialize Locks to avoid errors
        self.posDatasLock = Lock()
        # Initialize histogram
        self.position_histogram_dict = self.multiprocess_manager.dict()
        self.position_histogram_update_parameters = self.multiprocess_manager.dict(HistogramParameters)
        self.position_histogram_dict_updating = self.multiprocess_manager.Value(bool, True)
        self.update_histogram_parameters(init=True)
        # Start updating position data and storing it in history
        self.T_updateCombPosHistory = Thread(target=self.updateCombPosHistory)
        self.T_updateCombPosHistory.start()

        self.is_alive = self.multiprocess_manager.Value(bool, True)

    @staticmethod
    def setupSocket(address, port):
        # Set ZeroMQ socket to listen on incoming position data from all RPis
        context = zmq.Context()
        sockSUB = context.socket(zmq.SUB)
        sockSUB.setsockopt(zmq.SUBSCRIBE, ''.encode())
        sockSUB.RCVTIMEO = 10 # maximum duration to wait for data (in milliseconds)
        sockSUB.connect('tcp://' + address + ':' + str(port))

        return sockSUB

    @staticmethod
    def setupSocket_unpackPool(kwargs):
        """
        Helper function to allow using pool with argument list.
        kwargs - dict - key value pairs for input arguments to setupSocket
        """
        return onlineTrackingData.setupSocket(**kwargs)

    @staticmethod
    def setupSockets(cameraIDs, CameraSettings):
        kwargs_list = []
        for cameraID in cameraIDs:
            kwargs_list.append({'address': CameraSettings['CameraSpecific'][cameraID]['address'], 
                                'port': CameraSettings['General']['OnlineTracker_port']})
        pool = ThreadPool(len(cameraIDs))
        sockSUBs = pool.map(onlineTrackingData.setupSocket_unpackPool, kwargs_list)
        pool.close()

        return sockSUBs

    def updatePosDatas(self, nRPi):
        # Updates self.posDatas when any new position data is received
        # This loop continues until self.KeepGettingData is set False. This is done by self.close function
        while self.KeepGettingData:
            # Wait for position data update
            try:
                message = self.sockSUBs[nRPi].recv()  # Receive message
                message = message.decode()  # Decode bytes into string
            except:
                message = 'no message'
            if message != 'no message':
                posData = json.loads(message)  # Convert from string to original format
                # Ignore messages where all elements are None
                if any(posData):
                    # Update posData for the correct position in the list
                    with self.posDatasLock:
                        self.posDatas[nRPi] = posData

    def combineCurrentLineData(self, previousCombPos):
        with self.posDatasLock:
            posDatas = self.posDatas
        # Combine posDatas from cameras to position data
        if len(posDatas) > 1:
            # Convert posDatas for use in combineCamerasData function
            cameraPos = []
            for posData in posDatas:
                cameraPos.append(np.array(posData[:4], dtype=np.float32))
            # Combine data from cameras
            lastCombPos = combineCamerasData(cameraPos, previousCombPos, self.cameraIDs, 
                                             self.CameraSettings, self.arena_size)
        else:
            # If only a single camera is used, extract position data from posData into numpy array
            lastCombPos = np.array(posDatas[0][:4], dtype=np.float32)

        return lastCombPos

    def update_histogram_parameters(self, init=False):
        # Initialise histogram edgesrameters
        margins = self.position_histogram_update_parameters['margins']
        binSize = self.position_histogram_update_parameters['binSize']
        xHistogram_edges = np.append(np.arange(-margins, self.arena_size[0] + margins, binSize), 
                                     self.arena_size[0] + margins)
        yHistogram_edges = np.append(np.arange(-margins, self.arena_size[1] + margins, binSize), 
                                     self.arena_size[1] + margins)
        # If update requested with new parameters, recompute histogram
        if init:
            histmap = np.zeros((yHistogram_edges.size - 1, xHistogram_edges.size - 1), dtype=np.float32)
        else:
            combPos = np.array(self.combPosHistory)
            # Keep datapoints above speed limit
            one_second_steps = int(np.round(1 / self.combPos_update_interval))
            idx_keep = np.zeros(combPos.shape[0], dtype=bool)
            for npos in range(one_second_steps, combPos.shape[0] - 1):
                lastDistance = euclidean(combPos[npos,:2], combPos[npos - one_second_steps,:2])
                if lastDistance > self.position_histogram_update_parameters['speedLimit']:
                    idx_keep[npos] = True
            combPos = combPos[idx_keep, :]
            histmap, _1, _2 = np.histogram2d(combPos[:,1], combPos[:,0], [yHistogram_edges, xHistogram_edges])
        # Update shared data
        self.position_histogram_dict['data'] = histmap
        self.position_histogram_dict['parameters'] = self.position_histogram_update_parameters
        self.position_histogram_dict['edges'] = {'x': xHistogram_edges, 'y': yHistogram_edges}
        self.position_histogram_dict_updating.set(False)

    def updateCombPosHistory(self):
        # Initialize RPi position data listening, unless synthetic data requested
        self.T_updatePosDatas = []
        for nRPi in range(len(self.posDatas)):
            T = Thread(target=self.updatePosDatas, args=(nRPi,))
            T.start()
            self.T_updatePosDatas.append(T)
        # Continue once data is received from each RPi
        RPi_data_available = np.zeros(len(self.cameraIDs), dtype=bool)
        while not np.all(RPi_data_available) and self.KeepGettingData:
            sleep(0.05)
            for nRPi in range(len(self.posDatas)):
                if not (self.posDatas[nRPi] is None):
                    RPi_data_available[nRPi] = True
        if not self.KeepGettingData:
            return
        print('All RPi data available')
        # Set up speed tracking
        one_second_steps = int(np.round(1 / self.combPos_update_interval))
        self.lastSecondDistance = 0 # vector distance from position 1 second in past
        # Check data is available before proceeding
        lastCombPos = None
        while lastCombPos is None and self.KeepGettingData:
            lastCombPos = self.combineCurrentLineData(None)
        if not self.KeepGettingData:
            return
        self.combPosHistory.append(list(lastCombPos))
        time_of_last_datapoint = time()
        # Update the data at specific interval
        while self.KeepGettingData:
            if self.position_histogram_dict_updating.value:
                self.update_histogram_parameters()
            time_since_last_datapoint = time() - time_of_last_datapoint
            if time_since_last_datapoint > self.combPos_update_interval:
                # If enough time has passed since last update, append to combPosHistory list
                lastCombPos = self.combineCurrentLineData(self.combPosHistory[-1])
                if not (lastCombPos is None):
                    self.combPosHistory.append(list(lastCombPos))
                else:
                    self.combPosHistory.append(lastCombPos)
                time_of_last_datapoint = time()
                if len(self.combPosHistory) > one_second_steps:
                    # Compute distance from one second in the past if enough data available
                    currPos = self.combPosHistory[-1]
                    pastPos = self.combPosHistory[-one_second_steps]
                    if not (currPos is None) and not (pastPos is None):
                        self.lastSecondDistance = euclidean(currPos[:2], pastPos[:2])
                        if self.lastSecondDistance > self.position_histogram_dict['parameters']['speedLimit']:
                            # If animal has been moving enough, update histogram
                            tmp_x = np.array([currPos[0]]).astype(np.float32)
                            tmp_y = np.array([currPos[1]]).astype(np.float32)
                            yedges = self.position_histogram_dict['edges']['y']
                            xedges = self.position_histogram_dict['edges']['x']
                            histmap, _1, _2 = np.histogram2d(tmp_y, tmp_x, [yedges, xedges])
                            self.position_histogram_dict['data'] = self.position_histogram_dict['data'] + histmap
                    else:
                        self.lastSecondDistance = None
            else:
                sleep(self.combPos_update_interval * 0.1)

        self.is_alive.set(False)

    def close(self):
        # Closes the updatePosDatas thread and ZeroMQ socket for position listening
        self.KeepGettingData = False
        for T in self.T_updatePosDatas:
            T.join()
        self.T_updateCombPosHistory.join()
        for sockSUB in self.sockSUBs:
            sockSUB.close()


class RewardControl(object):
    # This class allows control of FEEDERs
    # FEEDER_type can be either 'milk' or 'pellet'
    def __init__(self, FEEDER_type, RPiIP, RPiUsername, RPiPassword, trialAudioSignal=None, 
                 negativeAudioSignal=0, lightSignalIntensity=0, lightSignalPins=[]):
        self.FEEDER_type = FEEDER_type
        self.RPiIP = RPiIP
        self.trialAudioSignal = trialAudioSignal
        self.lightSignalIntensity = lightSignalIntensity
        self.negativeAudioSignal = negativeAudioSignal
        self.lightSignalPins = lightSignalPins
        # Ensure files on the FEEDER RPi are up to date
        RewardControl.update_files_on_RPi(FEEDER_type, RPiIP, RPiUsername, verbose=False)
        # Set up SSH connection
        self.ssh_connection = ssh(RPiIP, RPiUsername, RPiPassword)
        self.ssh_connection.sendCommand('sudo pkill python') # Ensure any past processes have closed
        self.ssh_Controller_connection = ssh(RPiIP, RPiUsername, RPiPassword)
        # Set up ZMQ connection
        self.Controller_messenger = paired_messenger(address=self.RPiIP, port=4186)
        self.Controller_messenger.add_callback(self.Controller_message_parser)
        # Initialize Controller
        init_successful = self.init_Controller_until_positive_feedback(max_attempts=4, max_init_wait_time=25)
        # If initialization was unsuccessful, crash this script
        if not init_successful:
            raise Exception('Initialization was unsuccessful at: ' + RPiIP)

    @staticmethod
    def update_files_on_RPi(FEEDER_type, address, username, verbose=False):
        if FEEDER_type == 'milk':
            files = (os.path.join(package_path, 'milkFeederController.py'),)
        elif FEEDER_type == 'pellet':
            files = (os.path.join(package_path, 'pelletFeederController.py'),)
        else:
            raise ValueError('Unexpected FEEDER_type argument.')
        write_files_to_RPi(files, address, username=username, verbose=verbose)

    def Controller_message_parser(self, message):
        if message == 'init_successful'.encode():
            self.Controller_init_successful = True
        if message == 'releaseMilk successful'.encode():
            self.release_feedback_message_received = True
            self.releaseMilk_successful = True
        if message == 'release_pellets successful'.encode():
            self.release_feedback_message_received = True
            self.release_pellets_successful = True
        if message == 'release_pellets failed'.encode():
            self.release_feedback_message_received = True
            self.release_pellets_successful = False
        if message == 'release_pellets in_progress'.encode():
            self.release_pellets_in_progress = True

    def milkFeederController_init_command(self):
        command = 'python milkFeederController.py'
        command += ' ' + '--pinchValve '
        command += ' ' + '--init_feedback'
        if not (self.trialAudioSignal is None):
            command += ' ' + '--trialAudioSignal ' + ' '.join(map(str, self.trialAudioSignal))
        if self.negativeAudioSignal > 0:
            command += ' ' + '--negativeAudioSignal ' + str(self.negativeAudioSignal)
        if self.lightSignalIntensity > 0:
            command += ' ' + '--lightSignalIntensity ' + str(self.lightSignalIntensity)
        if len(self.lightSignalPins) > 0:
            command += ' ' + '--lightSignalPins ' + ' '.join(map(str, self.lightSignalPins))

        return command

    def pelletFeederController_init_command(self):
        command = 'python pelletFeederController.py'
        command += ' ' + '--init_feedback'

        return command

    def init_Controller_and_wait_for_feedback(self, max_init_wait_time=25):
        """
        Attempts to initialize the Controller class on the FEEDER
        """
        self.Controller_init_successful = False
        # Acquire the correct command for the FEEDER type
        if self.FEEDER_type == 'milk':
            command = self.milkFeederController_init_command()
        elif self.FEEDER_type == 'pellet':
            command = self.pelletFeederController_init_command()
        else:
            raise Exception('Unknown FEEDER_type {}'.format(self.FEEDER_type))
        # Initiate process on the RPi over SSH
        # Set timeout to max_init_wait_time and verbosity to False
        self.ssh_Controller_connection.sendCommand(command, max_init_wait_time, False)
        # Wait until positive initialization feedback or timer runs out
        start_time = time()
        while not self.Controller_init_successful and (time() - start_time) < max_init_wait_time:
            sleep(0.1)

        return self.Controller_init_successful

    def init_Controller_until_positive_feedback(self, max_attempts=4, max_init_wait_time=25):
        """
        Attempts to initialize Controller class on the FEEDER multiple times
        """
        controller_init_successful = False
        for n in range(max_attempts):
            controller_init_successful = \
                self.init_Controller_and_wait_for_feedback(max_init_wait_time=max_init_wait_time)
            if controller_init_successful:
                break
            else:
                print('Controller activation failed at ' + self.RPiIP + ' Re-initializing ...')
                # If initiation is unsuccessful, start again after killing existing processes
                self.ssh_connection.sendCommand('sudo pkill python') # Ensure any past processes have closed

        return controller_init_successful

    def release(self, quantity=1, max_attempts=10):
        """
        Releases the specified quantity of reward.
        Returns True or False, depending if action was successful.
        """
        # Reset feedback message detector
        self.release_feedback_message_received = False
        self.release_pellets_in_progress = False
        # Compute maximum time to wait for feedback depending on FEEDER type
        if self.FEEDER_type == 'milk':
            max_wait_time = quantity + 4
        elif self.FEEDER_type == 'pellet':
            max_wait_time = 5 * max_attempts
        else:
            raise Exception('Unknown FEEDER_type {}'.format(self.FEEDER_type))
        # Send ZMQ command depending on FEEDER type
        if self.FEEDER_type == 'milk':
            command = 'releaseMilk ' + str(quantity) + ' True'
        elif self.FEEDER_type == 'pellet':
            command = 'release_pellets ' + str(quantity) + ' True'
        else:
            raise Exception('Unknown FEEDER_type {}'.format(self.FEEDER_type))
        self.Controller_messenger.sendMessage(command.encode())
        # Wait for feedback message
        start_time = time()
        while not self.release_feedback_message_received and (time() - start_time) < max_wait_time:
            sleep(0.1)
            # The following allows reseting start_time between pellets
            if self.release_pellets_in_progress:
                self.release_pellets_in_progress = False
                start_time = time()
        # Return True or False if message successful/failure or False if no message received in time.
        if self.release_feedback_message_received:
            if self.FEEDER_type == 'milk':
                return self.releaseMilk_successful
            elif self.FEEDER_type == 'pellet':
                return self.release_pellets_successful
        else:
            return False

    def startTrialAudioSignal(self):
        self.Controller_messenger.sendMessage('startTrialAudioSignal'.encode())

    def stopTrialAudioSignal(self):
        self.Controller_messenger.sendMessage('stopTrialAudioSignal'.encode())

    def playNegativeAudioSignal(self):
        self.Controller_messenger.sendMessage('playNegativeAudioSignal'.encode())

    def startLightSignal(self):
        self.Controller_messenger.sendMessage('startLightSignal'.encode())

    def stopLightSignal(self):
        self.Controller_messenger.sendMessage('stopLightSignal'.encode())

    def startAllSignals(self):
        self.Controller_messenger.sendMessage('startAllSignals'.encode())

    def stopAllSignals(self):
        self.Controller_messenger.sendMessage('stopAllSignals'.encode())

    def close(self):
        """
        Closes all opened processes correctly.
        """
        try:
            self.Controller_messenger.sendMessage('close'.encode())
            self.Controller_messenger.close()
            self.ssh_Controller_connection.disconnect()
            self.ssh_connection.disconnect()
        except Exception as e:
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            print('Error in ' + frameinfo.filename + ' line ' + str(frameinfo.lineno - 3))
            print('Failed to close connection: ' + str(self.RPiIP))
            print(e)

class GlobalClockControl(object):

    def __init__(self, address, port, username='pi', password='raspberry'):
        self.initController_messenger(address, port)
        self.T_initRPiController = Thread(target=self.initRPiController, 
                                          args=(address, port, username, password))
        self.T_initRPiController.start()
        # Wait until all RPis have confirmed to be ready
        self.RPi_init_Successful = False
        while not self.RPi_init_Successful:
            sleep(0.1)

    def initController_messenger(self, address, port):
        # Set up ZMQ connection
        self.Controller_messenger = paired_messenger(address=address, port=int(port))
        self.Controller_messenger.add_callback(self.Controller_message_parser)
        sleep(1)

    def initRPiController(self, address, port, username, password):
        self.RPiSSH = ssh(address, username, password)
        self.RPiSSH.sendCommand('sudo pkill python')  # Ensure any past processes have closed
        command = 'python GlobalClock.py --remote --port ' + str(port)
        self.RPiSSH.sendCommand(command)

    def Controller_message_parser(self, message):
        if message == 'init_successful'.encode():
            self.RPi_init_Successful = True

    def start(self):
        self.Controller_messenger.sendMessage('start'.encode())

    def stop(self):
        self.Controller_messenger.sendMessage('stop'.encode())

    def close(self):
        self.stop()
        self.Controller_messenger.sendMessage('close'.encode())
        # Close SSH connections
        self.RPiSSH.disconnect()
        self.T_initRPiController.join()
        # Close ZMQ messengers
        self.Controller_messenger.close()
