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
import CombineTrackingData

def RPiStarter(RPiSettings):
    # This function starts the tracking.py script in all RPis in the RPiSettings input
    # Load infromation on all RPis
    RPiUsername = RPiSettings['username']
    RPiPassword = RPiSettings['password']
    RPiIPs = []
    for n_rpi in RPiSettings['use_RPi_nrs']:
        RPiIPs.append(RPiSettings['RPiIP'][n_rpi])
    # Initialize SSH connection with all RPis
    for nRPi in range(len(RPiIPs)):
        connection = ssh(RPiIPs[nRPi], RPiUsername, RPiPassword)
        connection.sendCommand('pkill python') # Ensure any past processes have closed
        # Start tracking.py script
        connection.sendCommand('cd ' + RPiSettings['tracking_folder'] + ' && nohup python tracking.py &')

def check_if_running(RPiSettings):
    # Check if all RPis are transmitting position signals
    # Get RPiSettings data
    LocalIP = RPiSettings['centralIP']
    PosPort = RPiSettings['pos_port']
    RPiIPs = []
    for n_rpi in RPiSettings['use_RPi_nrs']:
        RPiIPs.append(RPiSettings['RPiIP'][n_rpi])
    # Set ZeroMQ socket to listen on incoming position data from all RPis
    contextSUB = zmq.Context()
    sockSUB = contextSUB.socket(zmq.SUB)
    sockSUB.setsockopt(zmq.SUBSCRIBE, '')
    sockSUB.RCVTIMEO = 500 # maximum duration to wait for data (in milliseconds)
    for nRPi in range(len(RPiIPs)):
        sockSUB.connect('tcp://' + RPiIPs[nRPi] + ':' + PosPort)
    # Check if all RPis in use are sending data
    ReceivingPos = [False] * len(RPiSettings['use_RPi_nrs'])
    duration = 0
    start_time = time.time()
    while not all(ReceivingPos) and duration < 5:
        # This loop continues for 5 seconds, unless all RPis are detected to send data
        try:
            message = sockSUB.recv() # Check for message
        except:
            message = 'no message'
        if message != 'no message': # If a message was received
            posData = json.loads(message) # Convert from string to original format
            RPi_number = posData[0] # Identify RPi number that sent this message
            # Find that RPi number in the list and set it to True, as in sending messages
            RPiIDX = [i for i,x in enumerate(RPiSettings['use_RPi_nrs']) if x == RPi_number]
            ReceivingPos[RPiIDX[0]] = True
        duration = time.time() - start_time # Update duration of this while loop
    # Output True, if receiving position data from all RPis. False otherwise.
    return all(ReceivingPos)

def StopRPi(RPiSettings):
    # Sends 'stop' message until no more position data is received from RPis
    LocalIP = RPiSettings['centralIP']
    PosPort = RPiSettings['pos_port']
    StopPort = RPiSettings['stop_port']
    RPiIPs = []
    for n_rpi in RPiSettings['use_RPi_nrs']:
        RPiIPs.append(RPiSettings['RPiIP'][n_rpi])
    # Set ZeroMQ socket to listen on incoming position data from all RPis
    contextSUB = zmq.Context()
    sockSUB = contextSUB.socket(zmq.SUB)
    sockSUB.setsockopt(zmq.SUBSCRIBE, '')
    sockSUB.RCVTIMEO = 500 # maximum duration to wait for data (in milliseconds)
    for nRPi in range(len(RPiIPs)):
        sockSUB.connect('tcp://' + RPiIPs[nRPi] + ':' + PosPort)
    # Set Stop message Publishing ZeroMQ
    contextPUB = zmq.Context()
    sockPUB = contextPUB.socket(zmq.PUB)
    sockPUB.bind('tcp://' + LocalIP + ':' + StopPort)
    # Pause script for 100ms for sockets to be bound before messages are sent.
    time.sleep(0.1)
    # Send Stop command until no more Position data is received
    command = 'stop'
    ReceivingPos = True
    while ReceivingPos:
        sockPUB.send(command)
        try:
            message = sockSUB.recv()
        except:
            message = 'no message'
        if message == 'no message':
            ReceivingPos = False
    # Close Sockets
    sockSUB.close()
    sockPUB.close()


class onlineTrackingData(object):
    # Constantly updates position data for all RPis currently in use.
    # Initialize this class as RPIpos = onlineTrackingData(RPiSettings)
    # Check for latest position with combPos = RPIpos.combPosHistory[-1]

    # Make sure to use Locks to avoid errors, for example:
    # with self.combPosLock:
    #     combPos = RPIpos.combPosHistory[-1]

    # Optional arguments during initialization:
    #   HistogramParameters is a list: [margins, binSize, histogram_speed_limit]
    #   SynthData set to True for debugging using synthetically generated position data
    def __init__(self, RPiSettings, HistogramParameters, SynthData=False):
        # Initialise the class with input RPiSettings
        self.SynthData = SynthData
        self.HistogramParameters = HistogramParameters
        self.KeepGettingData = True # Set True for endless while loop of updating latest data
        self.RPiSettings = RPiSettings
        self.posDatas = [None] * len(self.RPiSettings['use_RPi_nrs'])
        self.combPosHistory = []
        # Initialize Locks to avoid errors
        self.posDatasLock = threading.Lock()
        self.combPosHistoryLock = threading.Lock()
        self.histogramLock = threading.Lock()
        if not self.SynthData:
            # Initialize RPi position data listening, unless synthetic data requested
            self.setupSocket() # Set up listening of position data
            threading.Thread(target=self.updatePosDatas).start()
            # Continue once data is received from each RPi
            RPi_data_available = np.zeros(len(self.RPiSettings['use_RPi_nrs']), dtype=bool)
            while not np.all(RPi_data_available):
                for n_rpi in range(len(self.RPiSettings['use_RPi_nrs'])):
                    if self.posDatas[n_rpi]:
                        RPi_data_available[n_rpi] = True
            print('All RPi data available, starting updatePosDatas')
        else:
            # Start generating movement data if synthetic data requested
            threading.Thread(target=self.generatePosData, args=[self.RPiSettings['use_RPi_nrs'][0]]).start()
            time.sleep(0.5)
            self.setupSocket() # Set up listening of position data
        # Start updating position data and storing it in history
        threading.Thread(target=self.updateCombPosHistory).start()

    def setupSocket(self):
        # Set ZeroMQ socket to listen on incoming position data from all RPis
        context = zmq.Context()
        self.sockSUB = context.socket(zmq.SUB)
        self.sockSUB.setsockopt(zmq.SUBSCRIBE, '')
        self.sockSUB.RCVTIMEO = 150 # maximum duration to wait for data (in milliseconds)
        for n_rpi in self.RPiSettings['use_RPi_nrs']:
            tmp = 'tcp://' + self.RPiSettings['RPiIP'][n_rpi] + ':' + self.RPiSettings['pos_port']
            self.sockSUB.connect(tmp)

    def generatePosData(self, n_rpi):
        # Generates continuous position data and updates self.posDatas for a single RPi
        data_rate = 0.01 # Seconds per datapoint
        nRPi = [i for i,x in enumerate(self.RPiSettings['use_RPi_nrs']) if x == n_rpi][0]
        oldPos = [0.0, 0.0]
        currPos = [1.0, 1.0]
        time_of_last_datapoint = time.time()
        while self.KeepGettingData:
            time_since_last_datapoint = time.time() - time_of_last_datapoint
            if time_since_last_datapoint > data_rate:
                newPos = [-1, -1]
                p0 = np.array(currPos) - np.array(oldPos)
                lastDirection = np.arctan2(p0[0], p0[1])
                while newPos[0] < 0 or newPos[0] > self.RPiSettings['arena_size'][0] or newPos[1] < 0 or newPos[1] > self.RPiSettings['arena_size'][1]:
                    time_since_last_datapoint = time.time() - time_of_last_datapoint
                    newDirection = np.random.normal(loc=lastDirection, scale=np.pi / 32)
                    # Allow circular continuity
                    newDirection = np.arctan2(np.sin(newDirection), np.cos(newDirection))
                    # Compute new position based on speed and angle
                    current_speed = np.random.normal(loc=50.0, scale=30.0) * time_since_last_datapoint
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
                    nRPi = [i for i,x in enumerate(self.RPiSettings['use_RPi_nrs']) if x == n_rpi]
                    nRPi = nRPi[0]
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
            lastCombPos = CombineTrackingData.combineCamerasData(cameraPos, previousCombPos, self.RPiSettings)
        else:
            # If only a single camera is used, extract position data from posData into numpy array
            lastCombPos = np.array(posDatas[0][3:7], dtype=np.float32)

        return lastCombPos

    def initializePosHistogram(self, HistogramParameters, update=False):
        # Initialise histogram edgesrameters
        margins = HistogramParameters['margins']
        binSize = HistogramParameters['binSize']
        xHistogram_edges = np.append(np.arange(-margins, self.RPiSettings['arena_size'][0] + margins, binSize), 
                                     self.RPiSettings['arena_size'][0] + margins)
        yHistogram_edges = np.append(np.arange(-margins, self.RPiSettings['arena_size'][1] + margins, binSize), 
                                     self.RPiSettings['arena_size'][1] + margins)
        # If update requested with new parameters, recompute histogram
        if update:
            with self.combPosHistoryLock:
                combPos = np.array(self.combPosHistory)
            # Keep datapoints above speed limit
            one_second_steps = int(np.round(1 / self.update_interval))
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
        # Updates position data information at a fixed interval
        self.update_interval = 0.05 # in seconds
        # Set up speed tracking
        one_second_steps = int(np.round(1 / self.update_interval))
        self.lastSecondDistance = 0 # vector distance from position 1 second in past
        # Initialize histogram
        self.initializePosHistogram(self.HistogramParameters)
        # Check data is available before proceeding
        with self.combPosHistoryLock:
            lastCombPos = self.combineCurrentLineData(None)
            self.combPosHistory.append(lastCombPos)
        time_of_last_datapoint = time.time()
        # Update the data at specific interval
        while self.KeepGettingData:
            time_since_last_datapoint = time.time() - time_of_last_datapoint
            if time_since_last_datapoint > self.update_interval:
                # If enough time has passed since last update, append to combPosHistory list
                with self.combPosHistoryLock:
                    lastCombPos = self.combineCurrentLineData(self.combPosHistory[-1])
                    self.combPosHistory.append(lastCombPos)
                time_of_last_datapoint = time.time()
                if len(self.combPosHistory) > one_second_steps:
                    # Compute distance from one second in the past if enough data available
                    with self.combPosHistoryLock:
                        currPos = self.combPosHistory[-1]
                        pastPos = self.combPosHistory[-one_second_steps]
                    if not np.isnan(currPos[0]) and not np.isnan(pastPos[0]):
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
                time.sleep(0.005)

    def close(self):
        # Closes the updatePosDatas thread and ZeroMQ socket for position listening
        self.KeepGettingData = False
        time.sleep(0.25) # Allow the thread to run one last time before closing the socket to avoid error
        if not self.SynthData:
            self.sockSUB.close()
