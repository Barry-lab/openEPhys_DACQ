### These functions allow interfacing with the Raspberry Pis.

### By Sander Tanni, January 2018, UCL

import zmq
from time import time, sleep
from sshScripts import ssh
import json
from threading import Lock, Thread
import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations
from TrackingDataProcessing import combineCamerasData
from ZMQcomms import paired_messenger
from multiprocessing.dummy import Pool as ThreadPool


class CameraControl(object):
    def __init__(self, address, port, username='pi', password='raspberry', retry=6, tracking=False):
        self._init_successful = False
        self.Controller_messenger = paired_messenger(address, int(port))
        self.Controller_messenger.add_callback(self.Controller_message_parser)
        self.RPiSSH = CameraControl.initRPiController(address, port, username, password, tracking)
        _init_start_time = time()
        while not self._init_successful:
            sleep(0.1)
            if retry and time() - _init_start_time > retry:
                self.RPiSSH = CameraControl.initRPiController(address, port, username, password, tracking)
                _init_start_time = time()

    @staticmethod
    def initRPiController(address, port, username='pi', password='raspberry', tracking=False):
        RPiSSH = ssh(address, username, password)
        RPiSSH.sendCommand('sudo pkill python') # Ensure any past processes have closed
        if tracking:
            command = 'cd ' + 'Tracking' + ' && python tracking.py --remote --port ' + str(port) + ' --tracking'
        else:
            command = 'cd ' + 'Tracking' + ' && python tracking.py --remote --port ' + str(port)
        RPiSSH.sendCommand_threading(command)

        return RPiSSH

    def Controller_message_parser(self, message):
        if message == 'init_successful':
            self._init_successful = True

    def start(self):
        message = 'start'
        self.Controller_messenger.sendMessage(message, verify='Action Completed: ' + message)

    def start_streaming(self, address, port):
        message = ' '.join(('start_streaming', str(address), str(port)))
        self.Controller_messenger.sendMessage(message, verify='Action Completed: ' + message)

    def stop(self):
        message = 'stop'
        self.Controller_messenger.sendMessage(message, verify='Action Completed: ' + message)

    def close(self):
        message = 'close'
        self.Controller_messenger.sendMessage(message, verify='Action Completed: ' + message)
        self.RPiSSH.disconnect()
        self.Controller_messenger.close()

    @staticmethod
    def CameraControl_unpackPool(kwargs):
        '''
        Helper function to allow using pool with argument list.
        kwargs - dict - key value pairs for input arguments to CameraControl
        '''
        return CameraControl(**kwargs)

    @staticmethod
    def init_CameraControls_with_TrackingSettings(TrackingSettings, OnlineTracking=False):
        '''
        Returns a dictionary where each TrackingSettings['use_RPi_nrs'] element is a key
        to corresponding initialized CameraControl object. 
        '''
        kwargs_list = []
        for n_rpi in TrackingSettings['use_RPi_nrs']:
            kwargs_list.append({'address': TrackingSettings['RPiInfo'][str(n_rpi)]['IP'], 
                                'port': TrackingSettings['ZMQcomms_port'], 
                                'username': TrackingSettings['username'], 
                                'password': TrackingSettings['password'], 
                                'tracking': OnlineTracking})
        pool = ThreadPool(len(TrackingSettings['use_RPi_nrs']))
        CameraControlsList = pool.map(CameraControl.CameraControl_unpackPool, kwargs_list)
        pool.close()
        CameraControls = {}
        for camera_controller, n_rpi in zip(CameraControlsList, TrackingSettings['use_RPi_nrs']):
            CameraControls[n_rpi] = camera_controller

        return CameraControls


class TrackingControl(object):

    def __init__(self, TrackingSettings):
        self.CameraControls = CameraControl.init_CameraControls_with_TrackingSettings(TrackingSettings, 
                                                                                      OnlineTracking=True)

    def start(self):
        for n_rpi in self.CameraControls:
            self.CameraControls[n_rpi].start()

    def stop(self):
        for n_rpi in self.CameraControls:
            self.CameraControls[n_rpi].stop()

    def close(self):
        for n_rpi in self.CameraControls:
            self.CameraControls[n_rpi].close()


class onlineTrackingData(object):
    # Constantly updates position data for all RPis currently in use.
    # Initialize this class as RPIpos = onlineTrackingData(TrackingSettings)
    # Check for latest position with combPos = RPIpos.combPosHistory[-1]

    # Make sure to use Locks to avoid errors, for example:
    # with self.combPosLock:
    #     combPos = RPIpos.combPosHistory[-1]

    # Optional arguments during initialization:
    #   HistogramParameters is a list: [margins, binSize, histogram_speed_limit]
    #   SynthData set to True for debugging using synthetically generated position data
    def __init__(self, TrackingSettings, HistogramParameters=None, SynthData=False):
        # Initialise the class with input TrackingSettings
        self.combPos_update_interval = 0.05 # in seconds
        self.SynthData = SynthData
        if HistogramParameters is None:
            HistogramParameters = {'margins': 10, # histogram data margins in centimeters
                                   'binSize': 2, # histogram binSize in centimeters
                                   'speedLimit': 10}# centimeters of distance in last second to be included
        self.HistogramParameters = HistogramParameters
        self.KeepGettingData = True # Set True for endless while loop of updating latest data
        self.TrackingSettings = TrackingSettings
        self.posDatas = [None for i in range(len(self.TrackingSettings['use_RPi_nrs']))]
        self.combPosHistory = []
        self.setupSocket() # Set up listening of position data
        # Initialize Locks to avoid errors
        self.posDatasLock = Lock()
        self.combPosHistoryLock = Lock()
        self.histogramLock = Lock()
        # Start updating position data and storing it in history
        Thread(target=self.updateCombPosHistory).start()

    def setupSocket(self):
        # Set ZeroMQ socket to listen on incoming position data from all RPis
        context = zmq.Context()
        self.sockSUB = context.socket(zmq.SUB)
        self.sockSUB.setsockopt(zmq.SUBSCRIBE, '')
        self.sockSUB.RCVTIMEO = 150 # maximum duration to wait for data (in milliseconds)
        for n_rpi in self.TrackingSettings['use_RPi_nrs']:
            tmp = 'tcp://' + self.TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + self.TrackingSettings['pos_port']
            self.sockSUB.connect(tmp)

    def generatePosData(self, n_rpi):
        # Generates continuous position data and updates self.posDatas for a single RPi
        data_rate = 0.01 # Seconds per datapoint
        nRPi = [i for i,x in enumerate(self.TrackingSettings['use_RPi_nrs']) if x == n_rpi][0]
        oldPos = [0.0, 0.0]
        currPos = [1.0, 1.0]
        time_of_last_datapoint = time()
        while self.KeepGettingData:
            time_since_last_datapoint = time() - time_of_last_datapoint
            if time_since_last_datapoint > data_rate:
                newPos = [-1, -1]
                p0 = np.array(currPos) - np.array(oldPos)
                lastDirection = np.arctan2(p0[0], p0[1])
                while newPos[0] < 0 or newPos[0] > self.TrackingSettings['arena_size'][0] or newPos[1] < 0 or newPos[1] > self.TrackingSettings['arena_size'][1]:
                    time_since_last_datapoint = time() - time_of_last_datapoint
                    newDirection = np.random.normal(loc=lastDirection, scale=np.pi / 32)
                    # Allow circular continuity
                    newDirection = np.arctan2(np.sin(newDirection), np.cos(newDirection))
                    # Compute new position based on speed and angle
                    current_speed = np.random.normal(loc=20.0, scale=20.0) * time_since_last_datapoint
                    if current_speed < 0.1:
                        current_speed = 0.1
                    if time_since_last_datapoint > 0.05:
                        current_speed = 0.1
                    posShift = np.array([np.sin(newDirection) * current_speed, np.cos(newDirection) * current_speed])
                    newPos = np.array(currPos) + posShift
                    if time_since_last_datapoint > 0.05:
                        with self.posDatasLock:
                            self.posDatas[nRPi] = [n_rpi, None, None, None, None, None, None]
                        lastDirection = (np.random.random() - 0.5) * 2 * np.pi
                oldPos = currPos
                currPos = newPos
                with self.posDatasLock:
                    self.posDatas[nRPi] = [n_rpi, newPos[0], newPos[1], None, None, None, None]
                time_of_last_datapoint = time()
            sleep(0.005)

    def updatePosDatas(self):
        # Updates self.posDatas when any new position data is received
        # This loop continues until self.KeepGettingData is set False. This is done by self.close function
        while self.KeepGettingData:
            if not self.SynthData:
                # Wait for position data update
                try:
                    message = self.sockSUB.recv() # Receive message
                except:
                    message = 'no message'
                if message != 'no message':
                    posData = json.loads(message) # Convert from string to original format
                    # Identify the sender of this message as RPi position in list
                    n_rpi = posData[0]
                    nRPi = [i for i,x in enumerate(self.TrackingSettings['use_RPi_nrs']) if x == n_rpi][0]
                    # Update posData for the correct position in the list
                    with self.posDatasLock:
                        self.posDatas[nRPi] = posData
            else:
                # If synthetic data generated, wait a moment before continuing
                sleep(0.02)

    def combineCurrentLineData(self, previousCombPos):
        with self.posDatasLock:
            posDatas = self.posDatas
        # Combine posDatas from cameras to position data
        if len(posDatas) > 1:
            # Convert posDatas for use in combineCamerasData function
            cameraPos = []
            for posData in posDatas:
                cameraPos.append(np.array(posData[1:5], dtype=np.float32))
            # Combine data from cameras
            lastCombPos = combineCamerasData(cameraPos, previousCombPos, self.TrackingSettings)
        else:
            # If only a single camera is used, extract position data from posData into numpy array
            lastCombPos = np.array(posDatas[0][1:5], dtype=np.float32)

        return lastCombPos

    def initializePosHistogram(self, HistogramParameters, update=False):
        # Initialise histogram edgesrameters
        margins = HistogramParameters['margins']
        binSize = HistogramParameters['binSize']
        xHistogram_edges = np.append(np.arange(-margins, self.TrackingSettings['arena_size'][0] + margins, binSize), 
                                     self.TrackingSettings['arena_size'][0] + margins)
        yHistogram_edges = np.append(np.arange(-margins, self.TrackingSettings['arena_size'][1] + margins, binSize), 
                                     self.TrackingSettings['arena_size'][1] + margins)
        # If update requested with new parameters, recompute histogram
        if update:
            with self.combPosHistoryLock:
                combPos = np.array(self.combPosHistory)
            # Keep datapoints above speed limit
            one_second_steps = int(np.round(1 / self.combPos_update_interval))
            idx_keep = np.zeros(combPos.shape[0], dtype=bool)
            for npos in range(one_second_steps, combPos.shape[0] - 1):
                lastDistance = euclidean(combPos[npos,:2], combPos[npos - one_second_steps,:2])
                if lastDistance > HistogramParameters['speedLimit']:
                    idx_keep[npos] = True
            combPos = combPos[idx_keep, :]
            histmap, _1, _2 = np.histogram2d(combPos[:,1], combPos[:,0], [yHistogram_edges, xHistogram_edges])
        else:
            histmap = np.zeros((yHistogram_edges.size - 1, xHistogram_edges.size - 1), dtype=np.float32)
        # Update shared data
        with self.histogramLock:
            self.HistogramParameters = HistogramParameters
            self.positionHistogram = histmap
            self.positionHistogramEdges = {'x': xHistogram_edges, 'y': yHistogram_edges}

    def updateCombPosHistory(self):
        if not self.SynthData:
            # Initialize RPi position data listening, unless synthetic data requested
            Thread(target=self.updatePosDatas).start()
            # Continue once data is received from each RPi
            RPi_data_available = np.zeros(len(self.TrackingSettings['use_RPi_nrs']), dtype=bool)
            while not np.all(RPi_data_available):
                for nRPi in range(len(self.posDatas)):
                    if not (self.posDatas[nRPi] is None):
                        RPi_data_available[nRPi] = True
            print('All RPi data available')
        else:
            # Start generating movement data if synthetic data requested
            Thread(target=self.generatePosData, args=[self.TrackingSettings['use_RPi_nrs'][0]]).start()
            sleep(0.5)
        # Set up speed tracking
        one_second_steps = int(np.round(1 / self.combPos_update_interval))
        self.lastSecondDistance = 0 # vector distance from position 1 second in past
        # Initialize histogram
        self.initializePosHistogram(self.HistogramParameters)
        # Check data is available before proceeding
        with self.combPosHistoryLock:
            lastCombPos = None
            while lastCombPos is None:
                lastCombPos = self.combineCurrentLineData(None)
            self.combPosHistory.append(list(lastCombPos))
        time_of_last_datapoint = time()
        # Update the data at specific interval
        while self.KeepGettingData:
            time_since_last_datapoint = time() - time_of_last_datapoint
            if time_since_last_datapoint > self.combPos_update_interval:
                # If enough time has passed since last update, append to combPosHistory list
                with self.combPosHistoryLock:
                    lastCombPos = self.combineCurrentLineData(self.combPosHistory[-1])
                    if not (lastCombPos is None):
                        self.combPosHistory.append(list(lastCombPos))
                    else:
                        self.combPosHistory.append(lastCombPos)
                time_of_last_datapoint = time()
                if len(self.combPosHistory) > one_second_steps:
                    # Compute distance from one second in the past if enough data available
                    with self.combPosHistoryLock:
                        currPos = self.combPosHistory[-1]
                        pastPos = self.combPosHistory[-one_second_steps]
                    if not (currPos is None) and not (pastPos is None):
                        self.lastSecondDistance = euclidean(currPos[:2], pastPos[:2])
                        if self.lastSecondDistance > self.HistogramParameters['speedLimit']:
                            # If animal has been moving enough, update histogram
                            with self.histogramLock:
                                tmp_x = np.array([currPos[0]]).astype(np.float32)
                                tmp_y = np.array([currPos[1]]).astype(np.float32)
                                yedges = self.positionHistogramEdges['y']
                                xedges = self.positionHistogramEdges['x']
                                histmap, _1, _2 = np.histogram2d(tmp_y, tmp_x, [yedges, xedges])
                                self.positionHistogram = self.positionHistogram + histmap
                    else:
                        self.lastSecondDistance = None
            else:
                sleep(self.combPos_update_interval * 0.1)

    def close(self):
        # Closes the updatePosDatas thread and ZeroMQ socket for position listening
        self.KeepGettingData = False
        sleep(0.25) # Allow the thread to run one last time before closing the socket to avoid error
        if not self.SynthData:
            self.sockSUB.close()

class RewardControl(object):
    # This class allows control of FEEDERs
    # FEEDER_type can be either 'milk' or 'pellet'
    def __init__(self, FEEDER_type, RPiIP, RPiUsername, RPiPassword, trialAudioSignal=None, negativeAudioSignal=0, lightSignalIntensity=0):
        self.FEEDER_type = FEEDER_type
        self.RPiIP = RPiIP
        self.trialAudioSignal = trialAudioSignal
        self.lightSignalIntensity = lightSignalIntensity
        self.negativeAudioSignal = negativeAudioSignal
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

    def Controller_message_parser(self, message):
        if message == 'init_successful':
            self.Controller_init_successful = True
        if message == 'releaseMilk successful':
            self.release_feedback_message_received = True
            self.releaseMilk_successful = True
        if message == 'release_pellets successful':
            self.release_feedback_message_received = True
            self.release_pellets_successful = True
        if message == 'release_pellets failed':
            self.release_feedback_message_received = True
            self.release_pellets_successful = False
        if message == 'release_pellets in_progress':
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
            command += ' ' + '--lightSignal ' + str(self.lightSignalIntensity)

        return command

    def pelletFeederController_init_command(self):
        command = 'python pelletFeederController.py'
        command += ' ' + '--init_feedback'

        return command

    def init_Controller_and_wait_for_feedback(self, max_init_wait_time=25):
        '''
        Attempts to initialize the Controller class on the FEEDER
        '''
        self.Controller_init_successful = False
        # Acquire the correct command for the FEEDER type
        if self.FEEDER_type == 'milk':
            command = self.milkFeederController_init_command()
        elif self.FEEDER_type == 'pellet':
            command = self.pelletFeederController_init_command()
        # Initiate process on the RPi over SSH
        self.T_Controller = Thread(target=self.ssh_Controller_connection.sendCommand, 
                                             args=(command, max_init_wait_time, False)) # Set timeout to max_init_wait_time and verbosity to False
        self.T_Controller.start()
        # Wait until positive initialization feedback or timer runs out
        start_time = time()
        while not self.Controller_init_successful and (time() - start_time) < max_init_wait_time:
            sleep(0.1)

        return self.Controller_init_successful

    def init_Controller_until_positive_feedback(self, max_attempts=4, max_init_wait_time=25):
        '''
        Attempts to initialize Controller class on the FEEDER multiple times
        '''
        for n in range(max_attempts):
            Controller_init_successful = self.init_Controller_and_wait_for_feedback(max_init_wait_time=25)
            if Controller_init_successful:
                break
            else:
                print('Controller activation failed at ' + self.RPiIP + ' Re-initializing ...')
                # If initiation is unsuccessful, start again after killing existing processes
                self.ssh_connection.sendCommand('sudo pkill python') # Ensure any past processes have closed
                self.T_Controller.join()

        return Controller_init_successful

    def release(self, quantity=1, max_attempts=10):
        '''
        Releases the specified quantity of reward.
        Returns True or False, depending if action was successful.
        '''
        # Reset feedback message detector
        self.release_feedback_message_received = False
        self.release_pellets_in_progress = False
        # Compute maximum time to wait for feedback depending on FEEDER type
        if self.FEEDER_type == 'milk':
            max_wait_time = quantity + 4
        elif self.FEEDER_type == 'pellet':
            max_wait_time = 5 * max_attempts
        # Send ZMQ command depending on FEEDER type
        if self.FEEDER_type == 'milk':
            command = 'releaseMilk ' + str(quantity) + ' True'
        elif self.FEEDER_type == 'pellet':
            command = 'release_pellets ' + str(quantity) + ' True'
        self.Controller_messenger.sendMessage(command)
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
        self.Controller_messenger.sendMessage('startTrialAudioSignal')

    def stopTrialAudioSignal(self):
        self.Controller_messenger.sendMessage('stopTrialAudioSignal')

    def playNegativeAudioSignal(self):
        self.Controller_messenger.sendMessage('playNegativeAudioSignal')

    def startLightSignal(self):
        self.Controller_messenger.sendMessage('startLightSignal')

    def stopLightSignal(self):
        self.Controller_messenger.sendMessage('stopLightSignal')

    def startAllSignals(self):
        self.Controller_messenger.sendMessage('startAllSignals')

    def stopAllSignals(self):
        self.Controller_messenger.sendMessage('stopAllSignals')

    def close(self):
        '''
        Closes all opened processes correctly.
        '''
        try:
            self.Controller_messenger.sendMessage('close')
            self.T_Controller.join()
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

    def __init__(self, TrackingSettings):
        self.TrackingSettings = TrackingSettings
        self.initController_messenger()
        self.T_initRPiController = Thread(target=self.initRPiController)
        self.T_initRPiController.start()
        # Wait until all RPis have confirmed to be ready
        self.RPi_init_Successful = False
        while not self.RPi_init_Successful:
            sleep(0.1)

    def initController_messenger(self):
        # Set up ZMQ connection
        self.Controller_messenger = paired_messenger(address=self.TrackingSettings['global_clock_ip'], 
                                                     port=int(self.TrackingSettings['global_clock_port']))
        self.Controller_messenger.add_callback(self.Controller_message_parser)
        sleep(1)

    def initRPiController(self):
        self.RPiSSH = ssh(self.TrackingSettings['global_clock_ip'], self.TrackingSettings['username'], 
                          self.TrackingSettings['password'])
        self.RPiSSH.sendCommand('sudo pkill python') # Ensure any past processes have closed
        command = 'python GlobalClock.py --remote'
        self.RPiSSH.sendCommand(command)

    def Controller_message_parser(self, message):
        if message == 'init_successful':
            self.RPi_init_Successful = True

    def start(self):
        self.Controller_messenger.sendMessage('start')

    def stop(self):
        self.Controller_messenger.sendMessage('stop')

    def close(self):
        self.stop()
        self.Controller_messenger.sendMessage('close')
        # Close SSH connections
        self.RPiSSH.disconnect()
        self.T_initRPiController.join()
        # Close ZMQ messengers
        self.Controller_messenger.close()
