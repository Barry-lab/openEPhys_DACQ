### These functions allow interfacing with the Raspberry Pis.

### By Sander Tanni, April 2017, UCL

import zmq
import time
from sshScripts import ssh
import json
import threading
import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations

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
            linedata = json.loads(message) # Convert from string to original format
            RPi_number = linedata[0] # Identify RPi number that sent this message
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

# def computeAbsolutePosition(linedatas, RPiSettings):
#     ########## Note! this function should be changed to utilize CombineTrackingData.py method
#     ########## Note! This function is also used by CumulativePosPlot.
#     # This function return the absolute position, 
#     # by combining information from multiple cameras
#     # Get Position values of all RPis and populate update text boxes
#     positions = np.zeros((len(linedatas), 2), dtype=np.float32)
#     positions_2 = np.zeros((len(linedatas), 2), dtype=np.float32)
#     for nRPi in range(len(linedatas)):
#         if linedatas[nRPi]:
#             positions[nRPi, 0] = linedatas[nRPi][3]
#             positions[nRPi, 1] = linedatas[nRPi][4]
#             if linedatas[nRPi][5]:
#                 # Compute position of second LED
#                 positions_2[nRPi, 0] = linedatas[nRPi][5]
#                 positions_2[nRPi, 1] = linedatas[nRPi][6]
#             else:
#                 positions_2[nRPi, 0] = None
#                 positions_2[nRPi, 1] = None
#         else:
#             positions[nRPi, 0] = None
#             positions[nRPi, 1] = None
#             positions_2[nRPi, 0] = None
#             positions_2[nRPi, 1] = None

#     position = np.mean(positions, axis=0)
#     positions_2 = np.mean(positions_2, axis=0)

#     return position, positions_2

def combineCamerasData(cameraPos, lastCombPos=None, RPiSettings=None):
    # This outputs position data based on which camera is closest to tracking target.

    # cameraPos - list of numpy vecors with 4 elements (x1,y1,x2,y2) for each camera
    # lastCombPos - Last known output from this function
    # RPiSettings - settings file saved by CameraSettingsGUI.py

    # Output - numpy vector (x1,y1,x2,y2) with data from closest camera

    # If lastCombPos is not provided, the function will attempt to locate the animal
    # simultaneously from at least 2 cameras with smaller separation than half of RPiSettings['camera_transfer_radius']
    #   If successful, closest mean coordinate is set as output
    #   If unsuccessful, output is None

    N_RPis = len(cameraPos)
    cameraPos = np.array(cameraPos, dtype=np.float32)
    camera_relative_locs = []
    for n_rpi in range(N_RPis):
        camera_relative_locs.append(np.fromstring(RPiSettings['RPi_location'][n_rpi],dtype=float,sep=','))
    camera_relative_locs = np.array(camera_relative_locs, dtype=np.float32)

    # Only work with camera data from inside the enviornment
    # Find bad pos data lines
    idxBad = np.zeros(cameraPos.shape[0], dtype=bool)
    arena_size = RPiSettings['arena_size'] # Points beyond arena size
    x_too_big = cameraPos[:,0] > arena_size[0] + 20
    y_too_big = cameraPos[:,1] > arena_size[1] + 20
    idxBad = np.logical_or(idxBad, np.logical_or(x_too_big, y_too_big))
    x_too_small = cameraPos[:,0] < -20
    y_too_small = cameraPos[:,1] < -20
    idxBad = np.logical_or(idxBad, np.logical_or(x_too_small, y_too_small))
    # Only keep camera data from within the environment
    N_RPis = np.sum(np.logical_not(idxBad))
    # Only continue if at least one RPi data remains
    if N_RPis > 0:
        cameraPos = cameraPos[np.logical_not(idxBad),:]
        camera_relative_locs = camera_relative_locs[np.logical_not(idxBad),:]
        if np.any(lastCombPos):
            # Check which cameras provide data close enough to lastCombPos
            RPi_correct = []
            for n_rpi in range(N_RPis):
                lastCombPos_distance = euclidean(cameraPos[n_rpi, :2], lastCombPos[:2])
                RPi_correct.append(lastCombPos_distance < RPiSettings['camera_transfer_radius'])
            RPi_correct = np.array(RPi_correct, dtype=bool)
            # If none were found to be withing search radius, set output to None
            if not np.any(RPi_correct):
                combPos = None
            else:
                # Use the reading from closest camera to target mean location that detects correct location
                if np.sum(RPi_correct) > 1:
                    # Only use correct cameras
                    N_RPis = np.sum(RPi_correct)
                    cameraPos = cameraPos[RPi_correct, :]
                    camera_relative_locs = camera_relative_locs[RPi_correct, :]
                    meanPos = np.mean(cameraPos[:, :2], axis=0)
                    # Find mean position distance from all cameras
                    cam_distances = []
                    for n_rpi in range(N_RPis):
                        camera_loc = camera_relative_locs[n_rpi, :] * np.array(RPiSettings['arena_size'])
                        cam_distances.append(euclidean(camera_loc, meanPos))
                    # Find closest distance camera and output its location coordinates
                    closest_camera = np.argmin(np.array(cam_distances))
                    combPos = cameraPos[closest_camera, :]
                else:
                    # If target only detected close enough to lastCombPos in a single camera, use it as output
                    combPos = cameraPos[np.where(RPi_correct)[0][0], :]
        else:
            # If no lastCombPos provided, check if position can be verified from more than one camera
            #### NOTE! This solution breaks down if more than two cameras incorrectly identify the same object
            ####       as the brightes spot, instead of the target LED.
            cameraPairs = []
            pairDistances = []
            for c in combinations(range(N_RPis), 2):
                pairDistances.append(euclidean(cameraPos[c[0], :2], cameraPos[c[1], :2]))
                cameraPairs.append(np.array(c))
            cameraPairs = np.array(cameraPairs)
            cameraPairs_Match = np.array(pairDistances) < (RPiSettings['camera_transfer_radius'] / 2)
            # If position can not be verified from multiple cameras, set output to none
            if not np.any(cameraPairs_Match):
                combPos = None
            else:
                # Otherwise, set output to mean of two cameras with best matching detected locations
                pairToUse = np.argmin(pairDistances)
                camerasToUse = np.array(cameraPairs[pairToUse, :])
                combPos = np.mean(cameraPos[camerasToUse, :2], axis=0)
                # Add NaN values for second LED
                combPos = np.append(combPos, np.empty(2) * np.nan)
    else:
        combPos = None

    return combPos


class latestPosData(object):
    # Constantly updates a list of linedatas for all RPis currently in use.
    # Initialize this class as lastPos = latestPosData(RPiSettings)
    # Check for latest pos with lastLineDatas = lastPos.linedatas
    def __init__(self, RPiSettings):
        # Initialise the class with input RPiSettings
        self.RPiSettings = RPiSettings
        self.linedatas = [None] * len(self.RPiSettings['use_RPi_nrs'])
        self.lastCombPos = None
        self.setupSocket() # Set up listening of position data
        # Continue once data is received from each RPi
        RPi_data_available = np.zeros(len(self.linedatas), dtype=bool)
        while not np.all(RPi_data_available):
            nRPi = self.updateLinedata()
            if nRPi is not None:
                RPi_data_available[nRPi] = True
        print('All RPi data available, starting updatePosData')
        self.KeepGettingData = True # Set True for endless while loop of updating latest data
        threading.Thread(target=self.updatePosData).start() # Start updatePosData function in a separate thread

    def setupSocket(self):
        # Set ZeroMQ socket to listen on incoming position data from all RPis
        context = zmq.Context()
        self.sockSUB = context.socket(zmq.SUB)
        self.sockSUB.setsockopt(zmq.SUBSCRIBE, '')
        self.sockSUB.RCVTIMEO = 150 # maximum duration to wait for data (in milliseconds)
        for n_rpi in self.RPiSettings['use_RPi_nrs']:
            tmp = 'tcp://' + self.RPiSettings['RPiIP'][n_rpi] + ':' + self.RPiSettings['pos_port']
            self.sockSUB.connect(tmp)

    def updateLinedata(self):
        try:
            message = self.sockSUB.recv() # Receive message
        except:
            message = 'no message'
            nRPi = None
        if message != 'no message':
            linedata = json.loads(message) # Convert from string to original format
            # Identify the sender of this message as RPi position in list
            n_rpi = linedata[0]
            nRPi = [i for i,x in enumerate(self.RPiSettings['use_RPi_nrs']) if x == n_rpi]
            nRPi = nRPi[0]
            # Update linedata for the correct position in the list
            self.linedatas[nRPi] = linedata

        return nRPi

    def updatePosData(self):
        # Updates self.linedatas variable with any input received on position data
        # This loop continues until self.KeepGettingData is set False. This is done by self.close function
        while self.KeepGettingData:
            nRPi = self.updateLinedata()
            if nRPi is not None:
                # Combine linedatas from cameras to position data
                if len(self.linedatas) > 1:
                    # Convert linedatas for use in combineCamerasData function
                    cameraPos = []
                    for linedata in self.linedatas:
                        cameraPos.append(np.array(linedata[3:7], dtype=np.float32))
                    # Combine data from cameras
                    self.lastCombPos = combineCamerasData(cameraPos, self.lastCombPos, self.RPiSettings)
                else:
                    # If only a single camera is used, extract position data from linedata into numpy array
                    self.lastCombPos = np.array(self.linedatas[0][3:7], dtype=np.float32)

    def close(self):
        # Closes the updatePosData thread and ZeroMQ socket for position listening
        self.KeepGettingData = False
        time.sleep(0.5) # Allow the thread to run one last time before closing the socket to avoid error
        self.sockSUB.close()
