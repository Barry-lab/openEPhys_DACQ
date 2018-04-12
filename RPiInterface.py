### These functions allow interfacing with the Raspberry Pis.

### By Sander Tanni, January 2018, UCL

import zmq
import time
from sshScripts import ssh
import json
import threading
import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations
from TrackingDataProcessing import combineCamerasData
from ZMQcomms import paired_messenger

class TrackingControl(object):

    def __init__(self, TrackingSettings):
        # Load infromation on all RPis
        self.TrackingSettings = TrackingSettings
        # Initialize SSH connection with all RPis
        self.RPiSSH = [None] * len(self.TrackingSettings['use_RPi_nrs'])
        self.RPiSSH_Lock = threading.Lock()
        T_initRPiSSH = []
        for nRPi, n_rpi in enumerate(self.TrackingSettings['use_RPi_nrs']):
            T_initRPiSSH.append(threading.Thread(target=self.initRPiSSH, args=[n_rpi, nRPi]))
            T_initRPiSSH[nRPi].start()
        for T in T_initRPiSSH:
            T.join()

    def initRPiSSH(self, n_rpi, nRPi):
        connection = ssh(self.TrackingSettings['RPiInfo'][str(n_rpi)]['IP'], self.TrackingSettings['username'], self.TrackingSettings['password'])
        connection.sendCommand('pkill python') # Ensure any past processes have closed
        with self.RPiSSH_Lock:
            self.RPiSSH[nRPi] = connection

    def start(self):
        for connection in self.RPiSSH:
            command = 'cd ' + self.TrackingSettings['tracking_folder'] + ' && python tracking.py &'
            connection.sendCommand(command)

    def stop(self):
        # Sends 'stop' message until no more position data is received from RPis
        LocalIP = self.TrackingSettings['centralIP']
        PosPort = self.TrackingSettings['pos_port']
        StopPort = self.TrackingSettings['stop_port']
        RPiIPs = []
        for n_rpi in self.TrackingSettings['use_RPi_nrs']:
            RPiIPs.append(self.TrackingSettings['RPiInfo'][str(n_rpi)]['IP'])
        # Set Stop message Publishing ZeroMQ
        contextPUB = zmq.Context()
        sockPUB = contextPUB.socket(zmq.PUB)
        sockPUB.bind('tcp://' + LocalIP + ':' + StopPort)
        command = 'stop'
        time.sleep(0.1) # Pause script for 100ms for sockets to be bound before messages are sent.
        # Send first Stop message
        sockPUB.send(command)
        # Set ZeroMQ socket to listen on incoming position data from all RPis
        contextSUB = zmq.Context()
        sockSUB = contextSUB.socket(zmq.SUB)
        sockSUB.setsockopt(zmq.SUBSCRIBE, '')
        sockSUB.RCVTIMEO = 250 # maximum duration to wait for data (in milliseconds)
        for nRPi in range(len(RPiIPs)):
            sockSUB.connect('tcp://' + RPiIPs[nRPi] + ':' + PosPort)
        # Send Stop command until no more Position data is received
        ReceivingPos = True
        while ReceivingPos:
            sockPUB.send(command)
            try:
                message = sockSUB.recv()
            except:
                message = 'no message'
            if message == 'no message':
                ReceivingPos = False
        # Close SSH connections
        for connection in self.RPiSSH:
            connection.disconnect()
        # Close Sockets
        sockSUB.close()
        sockPUB.close()


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
        self.posDatas = [None] * len(self.TrackingSettings['use_RPi_nrs'])
        self.combPosHistory = []
        self.setupSocket() # Set up listening of position data
        # Initialize Locks to avoid errors
        self.posDatasLock = threading.Lock()
        self.combPosHistoryLock = threading.Lock()
        self.histogramLock = threading.Lock()
        # Start updating position data and storing it in history
        threading.Thread(target=self.updateCombPosHistory).start()

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
        time_of_last_datapoint = time.time()
        while self.KeepGettingData:
            time_since_last_datapoint = time.time() - time_of_last_datapoint
            if time_since_last_datapoint > data_rate:
                newPos = [-1, -1]
                p0 = np.array(currPos) - np.array(oldPos)
                lastDirection = np.arctan2(p0[0], p0[1])
                while newPos[0] < 0 or newPos[0] > self.TrackingSettings['arena_size'][0] or newPos[1] < 0 or newPos[1] > self.TrackingSettings['arena_size'][1]:
                    time_since_last_datapoint = time.time() - time_of_last_datapoint
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
                            self.posDatas[nRPi] = [n_rpi, None, None, None, None, None, None, None, None]
                        lastDirection = (np.random.random() - 0.5) * 2 * np.pi
                oldPos = currPos
                currPos = newPos
                with self.posDatasLock:
                    self.posDatas[nRPi] = [n_rpi, None, None, newPos[0], newPos[1], None, None, None, None]
                time_of_last_datapoint = time.time()
            time.sleep(0.005)

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
                time.sleep(0.02)

    def combineCurrentLineData(self, previousCombPos):
        with self.posDatasLock:
            posDatas = self.posDatas
        # Combine posDatas from cameras to position data
        if len(posDatas) > 1:
            # Convert posDatas for use in combineCamerasData function
            cameraPos = []
            for posData in posDatas:
                cameraPos.append(np.array(posData[3:7], dtype=np.float32))
            # Combine data from cameras
            lastCombPos = combineCamerasData(cameraPos, previousCombPos, self.TrackingSettings)
        else:
            # If only a single camera is used, extract position data from posData into numpy array
            lastCombPos = np.array(posDatas[0][3:7], dtype=np.float32)

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
            threading.Thread(target=self.updatePosDatas).start()
            # Continue once data is received from each RPi
            RPi_data_available = np.zeros(len(self.TrackingSettings['use_RPi_nrs']), dtype=bool)
            while not np.all(RPi_data_available):
                for nRPi in range(len(self.posDatas)):
                    if not (self.posDatas[nRPi] is None):
                        RPi_data_available[nRPi] = True
            print('All RPi data available')
        else:
            # Start generating movement data if synthetic data requested
            threading.Thread(target=self.generatePosData, args=[self.TrackingSettings['use_RPi_nrs'][0]]).start()
            time.sleep(0.5)
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
        time_of_last_datapoint = time.time()
        # Update the data at specific interval
        while self.KeepGettingData:
            time_since_last_datapoint = time.time() - time_of_last_datapoint
            if time_since_last_datapoint > self.combPos_update_interval:
                # If enough time has passed since last update, append to combPosHistory list
                with self.combPosHistoryLock:
                    lastCombPos = self.combineCurrentLineData(self.combPosHistory[-1])
                    if not (lastCombPos is None):
                        self.combPosHistory.append(list(lastCombPos))
                    else:
                        self.combPosHistory.append(lastCombPos)
                time_of_last_datapoint = time.time()
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
                time.sleep(self.combPos_update_interval * 0.1)

    def close(self):
        # Closes the updatePosDatas thread and ZeroMQ socket for position listening
        self.KeepGettingData = False
        time.sleep(0.25) # Allow the thread to run one last time before closing the socket to avoid error
        if not self.SynthData:
            self.sockSUB.close()

class RewardControl(object):
    # This class allows control of FEEDERs
    # FEEDER_type can be either 'milk' or 'pellet'
    def __init__(self, FEEDER_type, RPiIP, RPiUsername, RPiPassword, audioControl=False, audioFreq=(4000, 0)):
        self.FEEDER_type = FEEDER_type
        self.RPiIP = RPiIP
        self.audioControl = audioControl
        # Set up SSH connection
        self.ssh_connection = ssh(RPiIP, RPiUsername, RPiPassword)
        self.ssh_connection.sendCommand('pkill python') # Ensure any past processes have closed
        # Initialize audio control script if audioControl=True
        if self.audioControl:
            self.ssh_connection.sendCommand('amixer sset PCM,0 99%' + ' &')
            threading.Thread(target=self.ssh_connection.sendCommand, 
                             args=('python audioSignalControl.py ' + \
                                   str(audioFreq[0]) + ' ' + str(audioFreq[1]),)).start()
            self.audioController = paired_messenger(address=self.RPiIP, port=4186)

    def release(self, quantity=1, wait_for_feedback=False, fail_limit=10):
        self.quantity = quantity
        self.success_count = 0
        self.fail_limit = fail_limit
        self.fail_count = 0
        self.FEEDER_Busy = True
        # Set input to feeder to include feedback command if requested
        if wait_for_feedback:
            feedback_string = ' feedback '
            # Set up listening for reward confirmation
            self.responses_incoming = False
            FEEDERmessages = paired_messenger(address=self.RPiIP, port=1232)
            FEEDERmessages.add_callback(self.feederMessageParser)
        else:
            feedback_string = ''
        # Send correct command to the feeder
        if self.FEEDER_type == 'pellet':
            self.ssh_connection.sendCommand('nohup python releasePellet.py ' + \
                                            str(int(quantity)) + feedback_string + ' &')
        elif self.FEEDER_type == 'milk':
            self.ssh_connection.sendCommand('nohup python openPinchValve.py ' + \
                                            str(quantity) + feedback_string + ' &')
        # If feedback requested, 
        if wait_for_feedback:
            if self.FEEDER_type == 'pellet':
                max_wait_time = 4 # This allows enough time for the pellet feeder to send the first message
            if self.FEEDER_type == 'milk':
                max_wait_time = quantity + 4 # This allows enough time for the milk feeder
            waiting_started = time.time()
            # Make sure not to wait too long
            while not self.responses_incoming and time.time() - waiting_started < max_wait_time:
                time.sleep(0.1)
            # If any messages received in time given, wait for feederMessageParser to set FEEDER_Busy to False
            if self.responses_incoming:
                while self.FEEDER_Busy:
                    time.sleep(0.1)
            # Stop listening messages
            FEEDERmessages.close()
            # If feedback failed, stop feeder and return failed message
            if self.FEEDER_Busy:
                self.ssh_connection.sendCommand('pkill python') # Stop the feeder
                return 'failed'
            else:
                # If feedback successful, send the feedback
                return self.message
        else:
            self.FEEDER_Busy = False

    def feederMessageParser(self, message):
        '''
        Checks the feedback message. If successful, sets FEEDER_Busy to False.
        Counts failed attempts and stops feeder when fail_limit reached.
        If multiple pellets requested, fail_count is reset after each successful pellet.
        '''
        self.responses_incoming = True
        if message == 'failed':
            self.fail_count += 1
            if self.fail_count > self.fail_limit:
                self.ssh_connection.sendCommand('pkill python') # Stop the feeder
                self.message = 'failed'
                self.FEEDER_Busy = False
        elif message == 'successful':
            self.success_count += 1
            if self.FEEDER_type == 'milk' or (self.FEEDER_type == 'pellet' and self.success_count >= self.quantity):
                self.message = 'successful'
                self.FEEDER_Busy = False
            else:
                self.fail_count = 0
        else:
            self.ssh_connection.sendCommand('pkill python') # Stop the feeder
            self.message = 'failed'
            self.FEEDER_Busy = False

    def playAudioSignal(self):
        if self.audioControl:
            self.audioController.sendMessage('play')
        else:
            raise ValueError('audioControl not activated!')

    def stopAudioSignal(self):
        if self.audioControl:
            self.audioController.sendMessage('stop')
        else:
            raise ValueError('audioControl not activated!')

    def close(self):
        if self.audioControl:
            self.audioController.close()
        self.ssh_connection.sendCommand('pkill python')
        self.ssh_connection.disconnect()
