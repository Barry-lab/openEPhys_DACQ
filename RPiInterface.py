### These functions allow interfacing with the Raspberry Pis.

### By Sander Tanni, April 2017, UCL

import zmq
import time
from sshScripts import ssh
import json
import threading
import numpy as np
import time

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
    sockSUB.RCVTIMEO = 150 # maximum duration to wait for data (in milliseconds)
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

def computeAbsolutePosition(linedatas, RPiSettings):
    ########## Note! this function should be changed to utilize CombineTrackingData.py method
    ########## Note! This function is also used by CumulativePosPlot.
    # This function return the absolute position, 
    # by combining information from multiple cameras
    # Get Position values of all RPis and populate update text boxes
    positions = np.zeros((len(linedatas), 2), dtype=np.float32)
    positions_2 = np.zeros((len(linedatas), 2), dtype=np.float32)
    for nRPi in range(len(linedatas)):
        if linedatas[nRPi]:
            positions[nRPi, 0] = linedatas[nRPi][3] + RPiSettings['corner_offset'][0]
            positions[nRPi, 1] = linedatas[nRPi][4] + RPiSettings['corner_offset'][1]
            if linedatas[nRPi][5]:
                # Compute position of second LED
                positions_2[nRPi, 0] = linedatas[nRPi][5] + RPiSettings['corner_offset'][0]
                positions_2[nRPi, 1] = linedatas[nRPi][6] + RPiSettings['corner_offset'][1]
            else:
                positions_2[nRPi, 0] = None
                positions_2[nRPi, 1] = None
        else:
            positions[nRPi, 0] = None
            positions[nRPi, 1] = None
            positions_2[nRPi, 0] = None
            positions_2[nRPi, 1] = None

    position = np.mean(positions, axis=0)
    positions_2 = np.mean(positions_2, axis=0)

    return position, positions_2

class latestPosData(object):
    # Constantly updates a list of linedatas for all RPis currently in use.
    # Initialize this class as lastPos = latestPosData(RPiSettings)
    # Check for latest pos with lastLineDatas = lastPos.linedatas
    def __init__(self, RPiSettings):
        # Initialise the class with input RPiSettings
        self.RPiSettings = RPiSettings
        self.linedatas = [None] * len(self.RPiSettings['use_RPi_nrs'])
        self.setupSocket() # Set up listening of position data
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

    def updatePosData(self):
        # Updates self.linedatas variable with any input received on position data
        # This loop continues until self.KeepGettingData is set False. This is done by self.close function
        while self.KeepGettingData:
            try:
                message = self.sockSUB.recv() # Receive message
            except:
                message = 'no message'
            if message != 'no message':
                linedata = json.loads(message) # Convert from string to original format
                # Identify the sender of this message as RPi position in list
                n_rpi = linedata[0]
                nRPi = [i for i,x in enumerate(self.RPiSettings['use_RPi_nrs']) if x == n_rpi]
                nRPi = nRPi[0]
                # Update linedata for the correct position in the list
                self.linedatas[nRPi] = linedata

    def close(self):
        # Closes the updatePosData thread and ZeroMQ socket for position listening
        self.KeepGettingData = False
        time.sleep(0.5) # Allow the thread to run one last time before closing the socket to avoid error
        self.sockSUB.close()
