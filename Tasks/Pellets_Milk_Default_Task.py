# This is a task

import pygame
import numpy as np
import threading
from RPiInterface import RewardControl
import time
from scipy.spatial.distance import euclidean
import random
from PyQt4 import QtGui, QtCore
from copy import deepcopy
from ZMQcomms import listenMessagesPAIR

def activateFEEDER(FEEDER_type, RPiIPBox, RPiUsernameBox, RPiPasswordBox, quantityBox):
    feeder = RewardControl(FEEDER_type, str(RPiIPBox.text()), 
                           str(RPiUsernameBox.text()), str(RPiPasswordBox.text()))
    feeder.release(float(str(quantityBox.text())), wait_for_feedback=False)
    feeder.close()

def addFeedersToList(self, FEEDER_type, FEEDER_settings=None):
    if FEEDER_settings is None:
        FEEDER_settings = {'ID': '1', 
                           'Present': True, 
                           'Active': True, 
                           'IP': '192.168.0.40', 
                           'Position': np.array([100,50]), 
                           'SignalHz': np.array(2800), 
                           'ModulHz': np.array(0)}
    # Create interface for interacting with this FEEDER
    FEEDER = {'Type': FEEDER_type}
    vbox = QtGui.QVBoxLayout()
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('ID:'))
    FEEDER['ID'] = QtGui.QLineEdit(FEEDER_settings['ID'])
    FEEDER['ID'].setMaximumWidth(40)
    hbox.addWidget(FEEDER['ID'])
    hbox.addWidget(QtGui.QLabel('IP:'))
    FEEDER['IP'] = QtGui.QLineEdit(FEEDER_settings['IP'])
    FEEDER['IP'].setMinimumWidth(105)
    hbox.addWidget(FEEDER['IP'])
    hbox.addWidget(QtGui.QLabel('Position:'))
    FEEDER['Position'] = QtGui.QLineEdit(','.join(map(str,FEEDER_settings['Position'])))
    FEEDER['Position'].setMinimumWidth(70)
    FEEDER['Position'].setMaximumWidth(70)
    hbox.addWidget(FEEDER['Position'])
    vbox.addLayout(hbox)
    hbox = QtGui.QHBoxLayout()
    FEEDER['Present'] = QtGui.QCheckBox('Present')
    FEEDER['Present'].setChecked(FEEDER_settings['Present'])
    hbox.addWidget(FEEDER['Present'])
    FEEDER['Active'] = QtGui.QCheckBox('Active')
    FEEDER['Active'].setChecked(FEEDER_settings['Active'])
    hbox.addWidget(FEEDER['Active'])
    activateButton = QtGui.QPushButton('Activate')
    activateButton.setMinimumWidth(70)
    activateButton.setMaximumWidth(70)
    FEEDER['ReleaseQuantity'] = QtGui.QLineEdit('1')
    FEEDER['ReleaseQuantity'].setMaximumWidth(40)
    activateButton.clicked.connect(lambda: activateFEEDER(FEEDER_type, FEEDER['IP'], 
                                                          self.settings['Username'], 
                                                          self.settings['Password'], 
                                                          FEEDER['ReleaseQuantity']))
    hbox.addWidget(activateButton)
    hbox.addWidget(FEEDER['ReleaseQuantity'])
    vbox.addLayout(hbox)
    if FEEDER_type == 'milk':
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(QtGui.QLabel('Signal (Hz):'))
        FEEDER['SignalHz'] = QtGui.QLineEdit(str(FEEDER_settings['SignalHz']))
        hbox.addWidget(FEEDER['SignalHz'])
        FEEDER['ModulHz'] = QtGui.QLineEdit(str(FEEDER_settings['ModulHz']))
        hbox.addWidget(FEEDER['ModulHz'])
        playSignalButton = QtGui.QPushButton('Play')
        playSignalButton.setMaximumWidth(40)
        playSignalButton.clicked.connect(lambda: playSineWaveSound(FEEDER['SignalHz'], 
                                                                   FEEDER['ModulHz']))
        hbox.addWidget(playSignalButton)
        vbox.addLayout(hbox)
    frame = QtGui.QFrame()
    frame.setLayout(vbox)
    frame.setFrameStyle(3)
    if FEEDER_type == 'milk':
        frame.setMaximumHeight(120)
    else:
        frame.setMaximumHeight(90)
    if FEEDER_type == 'pellet':
        self.pellet_feeder_settings_layout.addWidget(frame)
    elif FEEDER_type == 'milk':
        self.milk_feeder_settings_layout.addWidget(frame)
    self.settings['FEEDERs'][FEEDER_type].append(FEEDER)

def setDoubleHBoxStretch(hbox):
    hbox.setStretch(0,2)
    hbox.setStretch(1,1)

    return hbox

def exportSettingsFromGUI(self):
    TaskSettings = {'LastTravelTime': np.float64(str(self.settings['LastTravelTime'].text())), 
                    'LastTravelSmooth': np.int64(float(str(self.settings['LastTravelSmooth'].text()))), 
                    'LastTravelDist': np.int64(float(str(self.settings['LastTravelDist'].text()))), 
                    'PelletMilkRatio': np.float64(str(self.settings['PelletMilkRatio'].text())), 
                    'Chewing_TTLchan': np.int64(float(str(self.settings['Chewing_TTLchan'].text()))), 
                    'Username': str(self.settings['Username'].text()), 
                    'Password': str(self.settings['Password'].text()), 
                    'pelletGameOn': np.array(self.settings['pelletGameOn'].isChecked()), 
                    'InitPellets': np.int64(float(str(self.settings['InitPellets'].text()))), 
                    'PelletQuantity': np.int64(float(str(self.settings['PelletQuantity'].text()))), 
                    'PelletRewardMinSeparationMean': np.int64(float(str(self.settings['PelletRewardMinSeparationMean'].text()))), 
                    'PelletRewardMinSeparationVariance': np.float64(str(self.settings['PelletRewardMinSeparationVariance'].text())), 
                    'Chewing_Target': np.int64(float(str(self.settings['Chewing_Target'].text()))), 
                    'MaxInactivityDuration': np.int64(float(str(self.settings['MaxInactivityDuration'].text()))), 
                    'MilkTrialFailPenalty': np.int64(float(str(self.settings['MilkTrialFailPenalty'].text()))), 
                    'milkGameOn': np.array(self.settings['milkGameOn'].isChecked()), 
                    'InitMilk': np.float64(str(self.settings['InitMilk'].text())), 
                    'MilkQuantity': np.float64(str(self.settings['MilkQuantity'].text())), 
                    'MilkTrialMinSeparationMean': np.int64(float(str(self.settings['MilkTrialMinSeparationMean'].text()))), 
                    'MilkTrialMinSeparationVariance': np.float64(str(self.settings['MilkTrialMinSeparationVariance'].text())), 
                    'MilkTaskMinStartDistance': np.int64(float(str(self.settings['MilkTaskMinStartDistance'].text()))), 
                    'MilkTaskMinGoalDistance': np.int64(float(str(self.settings['MilkTaskMinGoalDistance'].text()))), 
                    'MilkTrialMaxDuration': np.int64(float(str(self.settings['MilkTrialMaxDuration'].text())))}
    FEEDERs = {}
    for FEEDER_type in self.settings['FEEDERs'].keys():
        FEEDERs[FEEDER_type] = {}
        IDs = []
        for feeder in self.settings['FEEDERs'][FEEDER_type]:
            IDs.append(str(int(str(feeder['ID'].text()))))
            FEEDERs[FEEDER_type][IDs[-1]] = {'ID': IDs[-1], 
                                             'Present': np.array(feeder['Present'].isChecked()), 
                                             'Active': np.array(feeder['Active'].isChecked()), 
                                             'IP': str(feeder['IP'].text()), 
                                             'Position': np.array(map(int, str(feeder['Position'].text()).split(',')))}
            if FEEDER_type == 'milk':
                FEEDERs[FEEDER_type][IDs[-1]]['SignalHz'] = np.int64(float(str(feeder['SignalHz'].text())))
                FEEDERs[FEEDER_type][IDs[-1]]['ModulHz'] = np.int64(float(str(feeder['ModulHz'].text())))
        # Check if there are duplicates of FEEDER IDs
        if any(IDs.count(ID) > 1 for ID in IDs):
            raise ValueError('Duplicates of IDs in ' + FEEDER_type + ' feeders!')
        # Give notification in case game was not set on but feeders were entered
        if len(FEEDERs[FEEDER_type].keys()) > 0 and not TaskSettings[FEEDER_type + 'GameOn']:
            from RecordingManager import show_message
            show_message(FEEDER_type + ' FEEDER(s) selected but game is NOT ON')
    TaskSettings['FEEDERs'] = FEEDERs

    return TaskSettings

def importSettingsToGUI(self, TaskSettings):
    # First remove all FEEDERs
    self.clearLayout(self.pellet_feeder_settings_layout, keep=1)
    self.clearLayout(self.milk_feeder_settings_layout, keep=1)
    # Load all settings
    for key in TaskSettings.keys():
        if isinstance(TaskSettings[key], np.ndarray) and TaskSettings[key].dtype == 'bool':
            self.settings[key].setChecked(TaskSettings[key])
        elif key == 'FEEDERs':
            for FEEDER_type in TaskSettings['FEEDERs'].keys():
                for ID in sorted(TaskSettings['FEEDERs'][FEEDER_type].keys(), key=int):
                    FEEDER_settings = TaskSettings['FEEDERs'][FEEDER_type][ID]
                    addFeedersToList(self, FEEDER_type, FEEDER_settings)
        elif key in self.settings.keys():
            self.settings[key].setText(str(TaskSettings[key]))

def createSineWaveSound(frequency, modulation_frequency=0):
    # Calling this function must be preceded by calling the following lines
    # for pygame to recognise the mono sound
    # pygame.mixer.pre_init(44100, -16, 1) # here 44100 needs to match the sampleRate in this function
    # pygame.init()
    sampleRate = 44100 # Must be the same as in the line 
    peak = 4096 # 4096 : the peak ; volume ; loudness
    arr = np.array([peak * np.sin(2.0 * np.pi * frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
    if modulation_frequency > 0:
        # Modulate pure tone at specified frequency
        modulation_frequency = int(modulation_frequency)
        marr = np.array([np.sin(2.0 * np.pi * modulation_frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.float64)
        marr = marr - min(marr)
        marr = marr / max(marr)
        arr = np.int16(arr.astype(np.float64) * marr)
    sound = pygame.sndarray.make_sound(arr)

    # Use sound.play(-1) to start sound. (-1) means it is in infinite loop
    # Use sound.stop() to stop the sound

    return sound

def playSineWaveSound(frequency, modulation_frequency=0):
    if type(frequency) == QtGui.QLineEdit:
        frequency = np.int64(float(str(frequency.text())))
    if type(modulation_frequency) == QtGui.QLineEdit:
        modulation_frequency = np.int64(float(str(modulation_frequency.text())))
    # Initialize pygame for playing sound
    pygame.mixer.pre_init(44100, -16, 1)
    pygame.init()
    # Get sound
    sound = createSineWaveSound(frequency, modulation_frequency)
    # Play 2 seconds of the sound
    sound.play(2)

def SettingsGUI(self):
    self.settings = {}
    self.settings['FEEDERs'] = {'pellet': [], 'milk': []}
    # Create General settings menu
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Last travel time (s)'))
    self.settings['LastTravelTime'] = QtGui.QLineEdit('2')
    hbox.addWidget(self.settings['LastTravelTime'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),0,0)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Last travel smoothing (cm)'))
    self.settings['LastTravelSmooth'] = QtGui.QLineEdit('3')
    hbox.addWidget(self.settings['LastTravelSmooth'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),1,0)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Last travel min distance (cm)'))
    self.settings['LastTravelDist'] = QtGui.QLineEdit('50')
    hbox.addWidget(self.settings['LastTravelDist'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),2,0)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Pellet vs Milk Reward ratio'))
    self.settings['PelletMilkRatio'] = QtGui.QLineEdit('0.25')
    hbox.addWidget(self.settings['PelletMilkRatio'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),3,0)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Chewing TTL channel'))
    self.settings['Chewing_TTLchan'] = QtGui.QLineEdit('5')
    hbox.addWidget(self.settings['Chewing_TTLchan'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),0,1)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Raspberry Pi usernames'))
    self.settings['Username'] = QtGui.QLineEdit('pi')
    hbox.addWidget(self.settings['Username'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),1,1)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Raspberry Pi passwords'))
    self.settings['Password'] = QtGui.QLineEdit('raspberry')
    hbox.addWidget(self.settings['Password'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),2,1)
    # Create Pellet task specific menu items
    vbox = QtGui.QVBoxLayout()
    font = QtGui.QFont('SansSerif', 15)
    string = QtGui.QLabel('Pellet Game Settings')
    string.setFont(font)
    vbox.addWidget(string)
    self.settings['pelletGameOn'] = QtGui.QCheckBox('Pellet Game ON')
    vbox.addWidget(self.settings['pelletGameOn'])
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Initial Pellets'))
    self.settings['InitPellets'] = QtGui.QLineEdit('5')
    hbox.addWidget(self.settings['InitPellets'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Reward Quantity'))
    self.settings['PelletQuantity'] = QtGui.QLineEdit('1')
    hbox.addWidget(self.settings['PelletQuantity'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Min Separation (s)'))
    self.settings['PelletRewardMinSeparationMean'] = QtGui.QLineEdit('10')
    hbox.addWidget(self.settings['PelletRewardMinSeparationMean'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Min Separation variance (%)'))
    self.settings['PelletRewardMinSeparationVariance'] = QtGui.QLineEdit('0.5')
    hbox.addWidget(self.settings['PelletRewardMinSeparationVariance'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Chewing Target count'))
    self.settings['Chewing_Target'] = QtGui.QLineEdit('4')
    hbox.addWidget(self.settings['Chewing_Target'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Inactivity pellet time (s)'))
    self.settings['MaxInactivityDuration'] = QtGui.QLineEdit('90')
    hbox.addWidget(self.settings['MaxInactivityDuration'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Milk Trial Fail Penalty (s)'))
    self.settings['MilkTrialFailPenalty'] = QtGui.QLineEdit('10')
    hbox.addWidget(self.settings['MilkTrialFailPenalty'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    # Create Pellet FEEDER items
    scroll_widget = QtGui.QWidget()
    self.pellet_feeder_settings_layout = QtGui.QVBoxLayout(scroll_widget)
    self.addPelletFeederButton = QtGui.QPushButton('Add FEEDER')
    self.addPelletFeederButton.clicked.connect(lambda: addFeedersToList(self, 'pellet'))
    self.pellet_feeder_settings_layout.addWidget(self.addPelletFeederButton)
    scroll = QtGui.QScrollArea()
    scroll.setWidget(scroll_widget)
    scroll.setWidgetResizable(True)
    vbox.addWidget(scroll)
    # Add Pellet Task settings to task specific settings layout
    frame = QtGui.QFrame()
    frame.setLayout(vbox)
    frame.setFrameStyle(3)
    self.task_specific_settings_layout.addWidget(frame)
    # Create Milk task specific menu items
    vbox = QtGui.QVBoxLayout()
    font = QtGui.QFont('SansSerif', 15)
    string = QtGui.QLabel('Milk Game Settings')
    string.setFont(font)
    vbox.addWidget(string)
    self.settings['milkGameOn'] = QtGui.QCheckBox('Milk Game ON')
    vbox.addWidget(self.settings['milkGameOn'])
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Initial Milk'))
    self.settings['InitMilk'] = QtGui.QLineEdit('2')
    hbox.addWidget(self.settings['InitMilk'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Reward Quantity'))
    self.settings['MilkQuantity'] = QtGui.QLineEdit('1')
    hbox.addWidget(self.settings['MilkQuantity'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Min Separation (s)'))
    self.settings['MilkTrialMinSeparationMean'] = QtGui.QLineEdit('40')
    hbox.addWidget(self.settings['MilkTrialMinSeparationMean'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Min Separation variance (%)'))
    self.settings['MilkTrialMinSeparationVariance'] = QtGui.QLineEdit('0.5')
    hbox.addWidget(self.settings['MilkTrialMinSeparationVariance'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Minimum Start Distance (cm)'))
    self.settings['MilkTaskMinStartDistance'] = QtGui.QLineEdit('50')
    hbox.addWidget(self.settings['MilkTaskMinStartDistance'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Minimum Goal Distance (cm)'))
    self.settings['MilkTaskMinGoalDistance'] = QtGui.QLineEdit('10')
    hbox.addWidget(self.settings['MilkTaskMinGoalDistance'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Maximum Trial Duration (s)'))
    self.settings['MilkTrialMaxDuration'] = QtGui.QLineEdit('9')
    hbox.addWidget(self.settings['MilkTrialMaxDuration'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    # Create Milk FEEDER items
    scroll_widget = QtGui.QWidget()
    self.milk_feeder_settings_layout = QtGui.QVBoxLayout(scroll_widget)
    self.addMilkFeederButton = QtGui.QPushButton('Add FEEDER')
    self.addMilkFeederButton.clicked.connect(lambda: addFeedersToList(self, 'milk'))
    self.milk_feeder_settings_layout.addWidget(self.addMilkFeederButton)
    scroll = QtGui.QScrollArea()
    scroll.setWidget(scroll_widget)
    scroll.setWidgetResizable(True)
    vbox.addWidget(scroll)
    # Add Milk Task settings to task specific settings layout
    frame = QtGui.QFrame()
    frame.setLayout(vbox)
    frame.setFrameStyle(3)
    self.task_specific_settings_layout.addWidget(frame)
    # Add necessary functions to TaskSettingsGUI
    self.exportSettingsFromGUI = exportSettingsFromGUI
    self.importSettingsToGUI = importSettingsToGUI

    return self

def smooth_edge_padding(data, smoothing):
    originalSize = data.size
    data = np.convolve(data, np.ones((smoothing,))/smoothing, mode='valid')
    missing = originalSize - data.size
    addStart = int(np.floor(missing / 2.0))
    addEnd = int(np.ceil(missing / 2.0))
    data = np.lib.pad(data, (addStart, addEnd), 'edge')

    return data

def compute_distance_travelled(posHistory, smoothing):
    distances = []
    posHistory = np.array(posHistory)[:, :2]
    posHistory[:,0] = smooth_edge_padding(posHistory[:,0], smoothing)
    posHistory[:,1] = smooth_edge_padding(posHistory[:,1], smoothing)
    for npos in range(posHistory.shape[0] - 1):
        prev_pos = posHistory[npos, :]
        curr_pos = posHistory[npos + 1, :]
        distance = euclidean(prev_pos, curr_pos)
        distances.append(distance)
    total_distance = np.sum(np.array(distances))

    return total_distance

class Core(object):
    def __init__(self, TaskSettings, TaskIO):
        self.TaskIO = TaskIO
        # Set Task Settings. This should be moved to Task Settings GUI
        self.FEEDERs = TaskSettings.pop('FEEDERs')
        self.TaskSettings = TaskSettings
        # Pre-compute variables
        self.one_second_steps = int(np.round(1 / self.TaskIO['RPIPos'].combPos_update_interval))
        self.distance_steps = int(np.round(self.TaskSettings['LastTravelTime'])) * self.one_second_steps
        self.max_distance_in_arena = int(round(np.hypot(self.TaskSettings['arena_size'][0], self.TaskSettings['arena_size'][1])))
        # Prepare TTL pulse time list
        self.ttlTimes = []
        self.ttlTimesLock = threading.Lock()
        self.TaskIO['OEmessages'].add_callback(self.append_ttl_pulses)
        # Set up Pellet Rewards
        self.pelletGameOn = self.TaskSettings['pelletGameOn']
        if self.pelletGameOn:
            self.activePfeeders = []
            for ID in sorted(self.FEEDERs['pellet'].keys(), key=int):
                if self.FEEDERs['pellet'][ID]['Active']:
                    self.activePfeeders.append(ID)
            self.lastPelletRewardLock = threading.Lock()
            self.lastPelletReward = time.time()
            self.updatePelletMinSepratation()
        # Set up Milk Rewards
        self.milkGameOn = self.TaskSettings['milkGameOn']
        if self.milkGameOn:
            self.activeMfeeders = []
            for ID in sorted(self.FEEDERs['milk'].keys(), key=int):
                if self.FEEDERs['milk'][ID]['Active']:
                    self.activeMfeeders.append(ID)
            self.milkTrialPerformance = [{'n_trials': 0, 'successful': 0, 'failed': 0} for i in range(len(self.activeMfeeders))]
            self.lastMilkTrial = time.time()
            self.updateMilkTrialMinSepratation()
            self.milkTrialFailTime = time.time() - self.TaskSettings['MilkTrialFailPenalty']
            self.feederID_milkTrial = self.chooseMilkTrialFeeder()
        self.milkTrialOn = False
        # Initialize FEEDERs
        print('Initializing FEEDERs...')
        T_initFEEDER = []
        self.TaskSettings_Lock = threading.Lock()
        if self.pelletGameOn:
            for ID in self.activePfeeders:
                T = threading.Thread(target=self.initFEEDER, args=('pellet', ID))
                T.start()
                T_initFEEDER.append(T)
        if self.milkGameOn:
            for ID in self.activeMfeeders:
                T = threading.Thread(target=self.initFEEDER, args=('milk', ID))
                T.start()
                T_initFEEDER.append(T)
        for T in T_initFEEDER:
            T.join()
        print('Initializing FEEDERs Successful')
        # Set game speed
        self.responseRate = 60 # Hz
        self.gameRate = 10 # Hz
        # Initialize game
        self.lastRewardLock = threading.Lock()
        self.lastReward = time.time()
        self.rewardInProgressLock = threading.Lock()
        self.rewardInProgress = []
        self.gameOn = False
        self.mainLoopActive = True
        self.clock = pygame.time.Clock()
        pygame.mixer.pre_init(44100, -16, 1)  # This is necessary for mono sound to work
        pygame.init()

    def initFEEDER(self, FEEDER_type, ID):
        with self.TaskSettings_Lock:
            IP = self.FEEDERs[FEEDER_type][ID]['IP']
            username = self.TaskSettings['Username']
            password = self.TaskSettings['Password']
        actuator = RewardControl(FEEDER_type, IP, username, password)
        with self.TaskSettings_Lock:
            self.FEEDERs[FEEDER_type][ID]['actuator'] = actuator

    def renderText(self, text):
        renderedText = self.font.render(text, True, self.textColor)

        return renderedText

    def append_ttl_pulses(self, message):
        parts = message.split()
        if parts[2] == str(self.TaskSettings['Chewing_TTLchan']) and parts[3] == str(1):
            with self.ttlTimesLock:
                self.ttlTimes.append(time.time())

    def number_of_chewings(self, lastReward):
        with self.ttlTimesLock:
            chewing_times = self.ttlTimes
        n_chewings = np.sum(np.array(chewing_times) > lastReward)

        return n_chewings

    def updatePelletMinSepratation(self):
        mean_val = self.TaskSettings['PelletRewardMinSeparationMean']
        var_val = self.TaskSettings['PelletRewardMinSeparationVariance']
        jitter = [int(- mean_val * var_val), int(mean_val * var_val)]
        jitter = random.randint(jitter[0], jitter[1])
        new_val = int(mean_val + jitter)
        self.TaskSettings['PelletRewardMinSeparation'] = new_val

    def updateMilkTrialMinSepratation(self):
        mean_val = self.TaskSettings['MilkTrialMinSeparationMean']
        var_val = self.TaskSettings['MilkTrialMinSeparationVariance']
        jitter = [int(- mean_val * var_val), int(mean_val * var_val)]
        jitter = random.randint(jitter[0], jitter[1])
        new_val = int(mean_val + jitter)
        self.TaskSettings['MilkTrialMinSeparation'] = new_val

    def releaseReward(self, FEEDER_type, ID, action='undefined', quantity=1):
        # Notify rest of the program that this is onging
        with self.rewardInProgressLock:
            self.rewardInProgress.append(FEEDER_type + ' ' + ID)
        # Make process visible on GUI
        if FEEDER_type == 'pellet':
            feeder_button = self.getButton('buttonReleasePellet', FEEDER_type, ID)
        elif FEEDER_type == 'milk':
            feeder_button = self.getButton('buttonReleaseMilk', FEEDER_type, ID)
        feeder_button['button_pressed'] = True
        # Send command to release reward and wait for positive feedback
        self.FEEDERs[FEEDER_type][ID]['actuator'].release(quantity, wait_for_feedback=True)
        # Send message to Open Ephys GUI
        OEmessage = 'Reward ' + FEEDER_type + ' ' + ID + ' ' + action + ' ' + str(quantity)
        self.TaskIO['MessageToOE'](OEmessage)
        # Reset GUI signal for feeder activity
        feeder_button['button_pressed'] = False
        # Reset last reward timer
        with self.lastRewardLock:
            self.lastReward = time.time()
        if 'pellet' == FEEDER_type:
            with self.lastPelletRewardLock:
                self.lastPelletReward = time.time()
        # Remove notification of onging reward delivery
        with self.rewardInProgressLock:
            self.rewardInProgress.remove(FEEDER_type + ' ' + ID)

    def initRewards(self):
        '''
        Deposits initiation rewards simultaneously from all FEEDERs.
        Completes only once all rewards have been released.
        '''
        T_initRewards = []
        if self.pelletGameOn and 'InitPellets' in self.TaskSettings.keys() and self.TaskSettings['InitPellets'] > 0:
            minPellets = int(np.floor(float(self.TaskSettings['InitPellets']) / len(self.activePfeeders)))
            extraPellets = np.mod(self.TaskSettings['InitPellets'], len(self.activePfeeders))
            n_pellets_Feeders = minPellets * np.ones(len(self.activePfeeders), dtype=np.int16)
            n_pellets_Feeders[:extraPellets] = n_pellets_Feeders[:extraPellets] + 1
            for ID, n_pellets in zip(self.activePfeeders, n_pellets_Feeders):
                if n_pellets > 0:
                    T = threading.Thread(target=self.releaseReward, 
                                         args=('pellet', ID, 'game_init', n_pellets))
                    T.start()
                    T_initRewards.append(T)
        if self.milkGameOn and 'InitMilk' in self.TaskSettings.keys() and self.TaskSettings['InitMilk'] > 0:
            for ID in self.activeMfeeders:
                T = threading.Thread(target=self.releaseReward, 
                                     args=('milk', ID, 'game_init', self.TaskSettings['InitMilk']))
                T.start()
                T_initRewards.append(T)
        for T in T_initRewards:
            T.join()

    def buttonGameOnOff_callback(self, button):
        # Switch Game On and Off
        self.gameOn = not self.gameOn
        button['button_pressed'] = self.gameOn
        OEmessage = 'Game On: ' + str(self.gameOn)
        self.TaskIO['MessageToOE'](OEmessage)

    def buttonReleaseReward_callback(self, button):
        [FEEDER_type, ID] = button['callargs']
        # Release reward from specified feeder and mark as User action
        if 'pellet' == FEEDER_type:
            self.releaseReward(FEEDER_type, ID, 'user', self.TaskSettings['PelletQuantity'])
        elif 'milk' == FEEDER_type:
            self.releaseReward(FEEDER_type, ID, 'user', self.TaskSettings['MilkQuantity'])

    def buttonManualPellet_callback(self, button):
        button['button_pressed'] = True
        # Update last reward time
        with self.lastPelletRewardLock:
            self.lastPelletReward = time.time()
        with self.lastRewardLock:
            self.lastReward = time.time()
        # Send message to Open Ephys GUI
        OEmessage = 'Reward pelletManual'
        self.TaskIO['MessageToOE'](OEmessage)
        # Keep button toggled for another 0.5 seconds
        time.sleep(0.5)
        button['button_pressed'] = False

    def buttonMilkTrial_callback(self, button):
        # Starts the trial with specific feeder as goal
        self.feederID_milkTrial = button['callargs'][0]
        self.start_milkTrial(action='user')

    def getButton(self, button_name, FEEDER_type=None, FEEDER_ID=None):
        button = self.buttons[self.button_names.index(button_name)]
        if isinstance(button, list):
            if FEEDER_type == 'milk':
                button = button[self.activeMfeeders.index(FEEDER_ID) + 1]
            elif FEEDER_type == 'pellet':
                button = button[self.activePfeeders.index(FEEDER_ID) + 1]

        return button


    def defineButtons(self):
        # Add or remove buttons in this function
        # Create new callbacks for new button if necessary
        # Callbacks are called at button click in a new thread with button dictionary as argument
        # Note default settings applied in self.createButtons()
        buttons = []
        button_names = []
        # Game On/Off button
        buttonGameOnOff = {'callback': self.buttonGameOnOff_callback, 
                           'text': 'Game Off', 
                           'toggled': {'text': 'Game On', 
                                       'color': (0, 128, 0)}}
        buttons.append(buttonGameOnOff)
        button_names.append('buttonGameOnOff')
        # Button to mark manually released pellet
        buttonManualPellet = {'callback': self.buttonManualPellet_callback, 
                              'text': 'Manual Pellet', 
                              'toggled': {'text': 'Manual Pellet', 
                                          'color': (0, 128, 0)}}
        buttons.append(buttonManualPellet)
        button_names.append('buttonManualPellet')
        if self.pelletGameOn: # The buttons are only active if pellet FEEDERs available
            # Button to release pellet
            buttonReleasePellet = []
            buttonReleasePellet.append({'text': 'Release Pellet'})
            for ID in self.activePfeeders:
                nFeederButton = {'callback': self.buttonReleaseReward_callback, 
                                 'callargs': ['pellet', ID], 
                                 'text': ID, 
                                 'toggled': {'text': ID, 
                                             'color': (0, 128, 0)}}
                buttonReleasePellet.append(nFeederButton)
            buttons.append(buttonReleasePellet)
            button_names.append('buttonReleasePellet')
        if self.milkGameOn: # The buttons are only active if milk FEEDERs available
            # Button to start milkTrial
            buttonMilkTrial = []
            buttonMilkTrial.append({'text': 'Milk Trial'})
            for ID in self.activeMfeeders:
                nFeederButton = {'callback': self.buttonMilkTrial_callback, 
                                 'callargs': [ID], 
                                 'text': ID, 
                                 'toggled': {'text': ID, 
                                             'color': (0, 128, 0)}}
                buttonMilkTrial.append(nFeederButton)
            buttons.append(buttonMilkTrial)
            button_names.append('buttonMilkTrial')
            # Button to release milk
            buttonReleaseMilk = []
            buttonReleaseMilk.append({'text': 'Deposit Milk'})
            for ID in self.activeMfeeders:
                nFeederButton = {'callback': self.buttonReleaseReward_callback, 
                                 'callargs': ['milk', ID], 
                                 'text': ID, 
                                 'toggled': {'text': ID, 
                                             'color': (0, 128, 0)}}
                buttonReleaseMilk.append(nFeederButton)
            buttons.append(buttonReleaseMilk)
            button_names.append('buttonReleaseMilk')

        return buttons, button_names

    def createButtons(self):
        buttons, button_names = self.defineButtons()
        # Add default color to all buttons
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                if not 'color' in button.keys():
                    buttons[i]['color'] = (128, 128, 128)
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    if not 'color' in subbutton.keys():
                        buttons[i][j + 1]['color'] = (128, 128, 128)
        # Add default button un-pressed state
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                if not 'button_pressed' in button.keys():
                    buttons[i]['button_pressed'] = False
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    if not 'button_pressed' in subbutton.keys():
                        buttons[i][j + 1]['button_pressed'] = False
        # Compute button locations
        xpos = self.screen_size[0] - self.screen_size[0] * self.buttonProportions + self.screen_margins
        xlen = self.screen_size[0] * self.buttonProportions - 2 * self.screen_margins
        ypos = np.linspace(self.screen_margins, self.screen_size[1] - self.screen_margins, 2 * len(buttons))
        ylen = ypos[1] - ypos[0]
        ypos = ypos[::2]
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                buttons[i]['Position'] = (int(round(xpos)), int(round(ypos[i])), int(round(xlen)), int(round(ylen)))
            elif isinstance(button, list):
                xsubpos = np.linspace(xpos, xpos + xlen, 2 * (len(button) - 1))
                xsublen = xsubpos[1] - xsubpos[0]
                xsubpos = xsubpos[::2]
                for j, subbutton in enumerate(button):
                    if j == 0:
                        buttons[i][j]['Position'] = (int(round(xpos)), int(round(ypos[i])), int(round(xlen)), int(round(ylen / 2.0)))
                    else:
                        buttons[i][j]['Position'] = (int(round(xsubpos[j - 1])), int(round(ypos[i] + ylen / 2.0)), int(round(xsublen)), int(round(ylen / 2.0)))
        # Create button rectangles
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                buttons[i]['Rect'] = pygame.Rect(button['Position'])
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    buttons[i][j + 1]['Rect'] = pygame.Rect(subbutton['Position'])
        # Render button texts
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                buttons[i]['textRendered'] = self.renderText(button['text'])
                if 'toggled' in button.keys():
                    buttons[i]['toggled']['textRendered'] = self.renderText(button['toggled']['text'])
            elif isinstance(button, list):
                for j, subbutton in enumerate(button):
                    buttons[i][j]['textRendered'] = self.renderText(subbutton['text'])
                    if 'toggled' in subbutton.keys():
                        buttons[i][j]['toggled']['textRendered'] = self.renderText(subbutton['toggled']['text'])

        return buttons, button_names

    def draw_button(self, button):
        # If button is pressed down, use the toggled color and text
        if button['button_pressed']:
            color = button['toggled']['color']
            textRendered = button['toggled']['textRendered']
        else:
            color = button['color']
            textRendered = button['textRendered']
        pygame.draw.rect(self.screen, color, button['Position'], 0)
        self.screen.blit(textRendered, button['Position'][:2])

    def draw_buttons(self):
        # Draw all buttons here
        for i, button in enumerate(self.buttons):
            if isinstance(button, dict):
                self.draw_button(button)
            elif isinstance(button, list):
                # Display name for button group
                self.screen.blit(button[0]['textRendered'], button[0]['Position'][:2])
                for j, subbutton in enumerate(button[1:]):
                    self.draw_button(subbutton)

    def create_progress_bars(self):
        game_progress = self.game_logic()
        while len(game_progress) == 0:
            game_progress = self.game_logic()
        # Initialise random color generation
        random.seed(123)
        rcolor = lambda: random.randint(0,255)
        # Compute progress bar locations
        textSpacing = 3
        textSpace = 10 + textSpacing
        textLines = 3
        xpos = np.linspace(self.screen_margins, self.screen_size[0] * (1 - self.buttonProportions) - self.screen_margins, 2 * len(game_progress))
        xlen = xpos[1] - xpos[0]
        xpos = xpos[::2]
        ybottompos = self.screen_size[1] - self.screen_margins - textSpace * textLines
        ymaxlen = self.screen_size[1] - 2 * self.screen_margins - textSpace * textLines
        progress_bars = []
        for i, gp in enumerate(game_progress):
            progress_bars.append({'name_text': self.renderText(gp['name']), 
                                  'target_text': '', 
                                  'value_text': '', 
                                  'name_position': (xpos[i], ybottompos + 1 * textSpace), 
                                  'target_position': (xpos[i], ybottompos + 2 * textSpace), 
                                  'value_position': (xpos[i], ybottompos + 3 * textSpace), 
                                  'color': (rcolor(), rcolor(), rcolor()), 
                                  'Position': {'xpos': xpos[i], 
                                               'xlen': xlen, 
                                               'ybottompos': ybottompos, 
                                               'ymaxlen': ymaxlen}})
        random.seed(None)

        return progress_bars

    def draw_progress_bars(self, game_progress):
        if len(game_progress) > 0:
            for gp, pb in zip(game_progress, self.progress_bars):
                self.screen.blit(pb['name_text'], pb['name_position'])
                self.screen.blit(self.renderText('T: ' + str(gp['target'])), pb['target_position'])
                self.screen.blit(self.renderText('C: ' + str(gp['status'])), pb['value_position'])
                if gp['complete']:
                    color = (255, 255, 255)
                    ylen = int(round(pb['Position']['ymaxlen']))
                    ypos = int(round(pb['Position']['ybottompos'] - pb['Position']['ymaxlen']))
                else:
                    color = pb['color']
                    ylen = int(round(gp['percentage'] * pb['Position']['ymaxlen']))
                    ypos = pb['Position']['ybottompos'] - ylen
                position = (pb['Position']['xpos'], ypos, pb['Position']['xlen'], ylen)
                pygame.draw.rect(self.screen, color, position, 0)

    def choose_pellet_feeder(self):
        '''
        Uses relative mean occupancy in bins closest to each feeder
        to increase probability of selecting feeder with lower mean occupancy.
        '''
        # Get list of FEEDER locations
        N_feeders = len(self.activePfeeders)
        if N_feeders > 1:
            FEEDER_Locs = []
            for ID in self.activePfeeders:
                FEEDER_Locs.append(np.array(self.FEEDERs['pellet'][ID]['Position'], dtype=np.float32))
            # Get occupancy information from RPIPos class
            with self.TaskIO['RPIPos'].histogramLock:
                histparam = deepcopy(self.TaskIO['RPIPos'].HistogramParameters)
                histmap = deepcopy(self.TaskIO['RPIPos'].positionHistogram)
                histXedges = deepcopy(self.TaskIO['RPIPos'].positionHistogramEdges['x'])
                histYedges = deepcopy(self.TaskIO['RPIPos'].positionHistogramEdges['y'])
            # Only recompute histogram bin bindings to feeders if histogram parameters have been updated
            calc_valid = hasattr(self, 'old_PelletHistogramParameters') and self.old_PelletHistogramParameters == histparam
            if not calc_valid:
                self.old_PelletHistogramParameters = histparam
                self.histogramPfeederMap = {}
                # Convert histogram edges to bin centers
                histXbin = (histXedges[1:] + histXedges[:-1]) / 2
                histYbin = (histYedges[1:] + histYedges[:-1]) / 2
                # Crop data to only include parts inside the arena boundaries
                arena_size = self.TaskSettings['arena_size']
                idx_X = np.logical_and(0 < histXbin, histXbin < arena_size[0])
                idx_Y = np.logical_and(0 < histYbin, histYbin < arena_size[1])
                histXbin = histXbin[idx_X]
                histYbin = histYbin[idx_Y]
                self.histogramPfeederMap['idx_crop_X'] = np.repeat(np.where(idx_X)[0][None, :], 
                                                                      histYbin.size, axis=0)
                self.histogramPfeederMap['idx_crop_Y'] = np.repeat(np.where(idx_Y)[0][:, None], 
                                                                      histXbin.size, axis=1)
                # Find closest feeder for each spatial bin
                histFeeder = np.zeros((histYbin.size, histXbin.size), dtype=np.int16)
                for xpos in range(histXbin.size):
                    for ypos in range(histYbin.size):
                        pos = np.array([histXbin[xpos], histYbin[ypos]], dtype=np.float32)
                        dists = np.zeros(N_feeders, dtype=np.float32)
                        for n_feeder in range(N_feeders):
                            dists[n_feeder] = np.linalg.norm(FEEDER_Locs[n_feeder] - pos)
                        histFeeder[ypos, xpos] = np.argmin(dists)
                self.histogramPfeederMap['feeder_map'] = histFeeder
            # Crop histogram to relavant parts
            histmap = histmap[self.histogramPfeederMap['idx_crop_Y'], self.histogramPfeederMap['idx_crop_X']]
            # Find mean occupancy in bins nearest to each feeder
            feeder_bin_occupancy = np.zeros(N_feeders, dtype=np.float64)
            for n_feeder in range(N_feeders):
                bin_occupancies = histmap[self.histogramPfeederMap['feeder_map'] == n_feeder]
                feeder_bin_occupancy[n_feeder] = np.mean(bin_occupancies)
            # Choose feeder with weighted randomness if any parts occupied
            if np.any(feeder_bin_occupancy > 0):
                feederProbabilityWeights = (np.sum(feeder_bin_occupancy) - feeder_bin_occupancy) ** 2
                feederProbability = feederProbabilityWeights / np.sum(feederProbabilityWeights)
                n_feeder = np.random.choice(N_feeders, p=feederProbability)
            else:
                n_feeder = np.random.choice(N_feeders)
        else:
            n_feeder = 0
        ID = self.activePfeeders[n_feeder]

        return ID

    def chooseMilkTrialFeeder(self):
        '''
        Uses performance to weight probability of selecting a feeder
        '''
        N_feeders = len(self.activeMfeeders)
        if N_feeders > 1:
            # Find percentage of failed trials
            performance = np.zeros(N_feeders, dtype=np.float64)
            for n_feeder in range(N_feeders):
                if self.milkTrialPerformance[n_feeder]['n_trials'] > 0:
                    srate = self.milkTrialPerformance[n_feeder]['successful'] / self.milkTrialPerformance[n_feeder]['n_trials']
                else:
                    srate = 0
                performance[n_feeder] = srate
            # Choose feeder with weighted randomness
            if np.any(performance > 0):
                feederProbabilityWeights = np.sum(performance) - performance
                feederProbability = feederProbabilityWeights / np.sum(feederProbabilityWeights)
                n_feeder = np.random.choice(N_feeders, p=feederProbability)
            else:
                n_feeder = np.random.choice(N_feeders)
        else:
            n_feeder = 0
        ID = self.activeMfeeders[n_feeder]

        return ID

    def start_milkTrial(self, action='undefined'):
        # Make process visible on GUI
        feeder_button = self.getButton('buttonMilkTrial', 'milk', self.feederID_milkTrial)
        feeder_button['button_pressed'] = True
        # These settings put the game_logic into milkTrial mode
        self.lastMilkTrial = time.time()
        self.milkTrialTone = createSineWaveSound(self.FEEDERs['milk'][self.feederID_milkTrial]['SignalHz'], 
                                                 self.FEEDERs['milk'][self.feederID_milkTrial]['ModulHz'])
        self.milkTrialTone.play(-1)
        self.milkTrialOn = True
        OEmessage = 'milkTrialStart ' + action + ' ' + self.feederID_milkTrial
        self.TaskIO['MessageToOE'](OEmessage)

    def stop_milkTrial(self, successful):
        self.milkTrialOn = False
        self.milkTrialTone.stop()
        OEmessage = 'milkTrialEnd Success:' + str(successful)
        self.TaskIO['MessageToOE'](OEmessage)
        # Reset GUI signal of trial process
        feeder_button = self.getButton('buttonMilkTrial', 'milk', self.feederID_milkTrial)
        feeder_button['button_pressed'] = False
        # Record outcome and release reward if successful
        n_feeder = self.activeMfeeders.index(self.feederID_milkTrial)
        self.milkTrialPerformance[n_feeder]['n_trials'] += 1
        if successful:
            threading.Thread(target=self.releaseReward, 
                             args=('milk', self.feederID_milkTrial, 'goal_milk', 
                                   self.TaskSettings['MilkQuantity'])
                             ).start()
            self.milkTrialPerformance[n_feeder]['successful'] += 1
        else:
            self.milkTrialFailTime = time.time()
            self.milkTrialPerformance[n_feeder]['failed'] += 1
        # Update n_feeder_milkTrial for next trial
        self.feederID_milkTrial = self.chooseMilkTrialFeeder()
        self.updateMilkTrialMinSepratation()

    def check_if_reward_in_progress(self):
        '''
        Returns True if any rewards are currently being delivered
        '''
        with self.rewardInProgressLock:
            n_rewards_in_progress = len(self.rewardInProgress)
        reward_in_progress = n_rewards_in_progress > 0

        return reward_in_progress

    def game_logic(self):
        game_progress = []
        reward_in_progress = self.check_if_reward_in_progress()
        # Get animal position history
        with self.TaskIO['RPIPos'].combPosHistoryLock:
            posHistory = self.TaskIO['RPIPos'].combPosHistory[-self.distance_steps:]
        if not (None in posHistory):
            if self.pelletGameOn: # The following progress is monitored only if pellet reward used
                with self.lastRewardLock:
                    timeSinceLastReward = time.time() - self.lastReward
                with self.lastPelletRewardLock:
                    timeSinceLastPelletReward = self.lastPelletReward
                # Check if animal has been without pellet reward for too long
                game_progress.append({'name': 'Inactivity', 
                                      'goals': ['inactivity'], 
                                      'target': self.TaskSettings['MaxInactivityDuration'], 
                                      'status': int(round(timeSinceLastReward)), 
                                      'complete': timeSinceLastReward >= self.TaskSettings['MaxInactivityDuration'], 
                                      'percentage': timeSinceLastReward / float(self.TaskSettings['MaxInactivityDuration'])})
                # Check if enough time as passed since last pellet reward
                game_progress.append({'name': 'Since Pellet', 
                                      'goals': ['pellet', 'milkTrialStart'], 
                                      'target': self.TaskSettings['PelletRewardMinSeparation'], 
                                      'status': int(round(timeSinceLastPelletReward)), 
                                      'complete': timeSinceLastPelletReward >= self.TaskSettings['PelletRewardMinSeparation'], 
                                      'percentage': timeSinceLastPelletReward / float(self.TaskSettings['PelletRewardMinSeparation'])})
                # Check if animal has been chewing enough since last reward
                n_chewings = self.number_of_chewings(timeSinceLastPelletReward)
                game_progress.append({'name': 'Chewing', 
                                      'goals': ['pellet'], 
                                      'target': self.TaskSettings['Chewing_Target'], 
                                      'status': n_chewings, 
                                      'complete': n_chewings >= self.TaskSettings['Chewing_Target'], 
                                      'percentage': n_chewings / float(self.TaskSettings['Chewing_Target'])})
                # Check if has been moving enough in the last few seconds
                total_distance = compute_distance_travelled(posHistory, self.TaskSettings['LastTravelSmooth'])
                game_progress.append({'name': 'Mobility', 
                                      'goals': ['pellet', 'milkTrialStart'], 
                                      'target': self.TaskSettings['LastTravelDist'], 
                                      'status': int(round(total_distance)), 
                                      'complete': total_distance >= self.TaskSettings['LastTravelDist'], 
                                      'percentage': total_distance / float(self.TaskSettings['LastTravelDist'])})
            if self.milkGameOn: # The following progress is monitored only if milk reward used
                # Check if milk trial penalty still applies to pellet rewards
                timeSinceLastMilkFailedTrial = time.time() - self.milkTrialFailTime
                game_progress.append({'name': 'Fail Penalty', 
                                      'goals': ['pellet'], 
                                      'target': self.TaskSettings['MilkTrialFailPenalty'], 
                                      'status': int(round(timeSinceLastMilkFailedTrial)), 
                                      'complete': timeSinceLastMilkFailedTrial >= self.TaskSettings['MilkTrialFailPenalty'], 
                                      'percentage': timeSinceLastMilkFailedTrial / float(self.TaskSettings['MilkTrialFailPenalty'])})
                # Check if enough time as passed since last milk trial
                timeSinceLastMilkTrial = time.time() - self.lastMilkTrial
                game_progress.append({'name': 'Since Trial', 
                                      'goals': ['milkTrialStart'], 
                                      'target': self.TaskSettings['MilkTrialMinSeparation'], 
                                      'status': int(round(timeSinceLastMilkTrial)), 
                                      'complete': timeSinceLastMilkTrial >= self.TaskSettings['MilkTrialMinSeparation'], 
                                      'percentage': timeSinceLastMilkTrial / float(self.TaskSettings['MilkTrialMinSeparation'])})
                # Check if animal is far enough from milk rewards
                distances = []
                for ID in self.activeMfeeders:
                    distances.append(euclidean(np.array(posHistory[-1][:2]), self.FEEDERs['milk'][ID]['Position']))
                minDistance = min(distances)
                game_progress.append({'name': 'Milk Distance', 
                                  'goals': ['milkTrialStart'], 
                                  'target': self.TaskSettings['MilkTaskMinStartDistance'], 
                                  'status': int(round(minDistance)), 
                                  'complete': minDistance >= self.TaskSettings['MilkTaskMinStartDistance'], 
                                  'percentage': minDistance / float(self.TaskSettings['MilkTaskMinStartDistance'])})
                if self.milkTrialOn:
                    # Check if animal is close enough to goal location
                    distance = euclidean(np.array(posHistory[-1][:2]), self.FEEDERs['milk'][self.feederID_milkTrial]['Position'])
                    game_progress.append({'name': 'Goal Distance', 
                                          'goals': ['milkTrialSuccess'], 
                                          'target': self.TaskSettings['MilkTaskMinGoalDistance'], 
                                          'status': int(round(distance)), 
                                          'complete': distance <= self.TaskSettings['MilkTaskMinGoalDistance'], 
                                          'percentage': 1 - (distance - self.TaskSettings['MilkTaskMinGoalDistance']) / float(self.max_distance_in_arena)})
                    trial_run_time = time.time() - self.lastMilkTrial
                    # Check if trial has been running for too long
                    game_progress.append({'name': 'Trial Duration', 
                                          'goals': ['milkTrialFail'], 
                                          'target': self.TaskSettings['MilkTrialMaxDuration'], 
                                          'status': int(round(trial_run_time)), 
                                          'complete': trial_run_time > self.TaskSettings['MilkTrialMaxDuration'], 
                                          'percentage': trial_run_time / float(self.TaskSettings['MilkTrialMaxDuration'])})
                else:
                    # Create empty progress info if trial not ongoing
                    game_progress.append({'name': 'Goal Distance', 
                                          'goals': ['milkTrialSuccess'], 
                                          'target': self.TaskSettings['MilkTaskMinGoalDistance'], 
                                          'status': 0, 
                                          'complete': False, 
                                          'percentage': 0})
                    game_progress.append({'name': 'Trial Duration', 
                                          'goals': ['milkTrialFail'], 
                                          'target': self.TaskSettings['MilkTrialMaxDuration'], 
                                          'status': 0, 
                                          'complete': False, 
                                          'percentage': 0})
            if not self.milkTrialOn and not reward_in_progress:
                # If milk trial currently not active
                # Check if game progress complete for any outcome
                pellet_status_complete = []
                inactivity_complete = []
                milkTrialStart_complete = []
                for gp in game_progress:
                    if 'pellet' in gp['goals']:
                        pellet_status_complete.append(gp['complete'])
                    if 'inactivity' in gp['goals']:
                        inactivity_complete.append(gp['complete'])
                    if 'milkTrialStart' in gp['goals']:
                        milkTrialStart_complete.append(gp['complete'])
                if self.pelletGameOn and self.milkGameOn and all(pellet_status_complete) and all(milkTrialStart_complete):
                    # Conditions for both pellet release and milkTrial are met, choose one based on chance
                    if random.uniform(0, 1) < self.TaskSettings['PelletMilkRatio']:
                        ID = self.choose_pellet_feeder()
                        threading.Thread(target=self.releaseReward, 
                                         args=('pellet', ID, 'goal_pellet', self.TaskSettings['PelletQuantity'])
                                         ).start()
                        self.updatePelletMinSepratation()
                    else:
                        self.start_milkTrial(action='goal_milkTrialStart')
                elif self.milkGameOn and all(milkTrialStart_complete):
                    # If conditions met, start milkTrial
                    self.start_milkTrial(action='goal_milkTrialStart')
                elif self.pelletGameOn and all(pellet_status_complete):
                    # If conditions met, release pellet reward
                    ID = self.choose_pellet_feeder()
                    threading.Thread(target=self.releaseReward, 
                                     args=('pellet', ID, 'goal_pellet', self.TaskSettings['PelletQuantity'])
                                     ).start()
                    self.updatePelletMinSepratation()
                elif self.pelletGameOn and all(inactivity_complete):
                    # If animal has been inactive and without pellet rewards, release pellet reward
                    ID = self.choose_pellet_feeder()
                    threading.Thread(target=self.releaseReward, 
                                     args=('pellet', ID, 'goal_inactivity', self.TaskSettings['PelletQuantity'])
                                     ).start()
            elif self.milkTrialOn:
                # If milk trial is currently ongoing
                # Check if any outcome criteria is reached
                milkTrialSuccess_complete = []
                milkTrialFail_complete = []
                for gp in game_progress:
                    if 'milkTrialSuccess' in gp['goals']:
                        milkTrialSuccess_complete.append(gp['complete'])
                    if 'milkTrialFail' in gp['goals']:
                        milkTrialFail_complete.append(gp['complete'])
                if all(milkTrialSuccess_complete):
                    # If animal has reached to goal, stop trial and give reward
                    self.stop_milkTrial(successful=True)
                elif all(milkTrialFail_complete):
                    # If time has run out, stop trial
                    self.stop_milkTrial(successful=False)

        return game_progress

    def update_display(self, game_progress):
        self.screen.fill((0, 0, 0))
        self.draw_buttons()
        self.draw_progress_bars(game_progress)
        # Update display
        pygame.display.update()

    def update_states(self):
        if self.gameOn:
            game_progress = self.game_logic()
        self.update_display(game_progress)

    def main_loop(self):
        # Ensure that Position data is available
        posHistory = []
        while len(posHistory) < self.distance_steps:
            time.sleep(0.5)
            with self.TaskIO['RPIPos'].combPosHistoryLock:
                posHistory = self.TaskIO['RPIPos'].combPosHistory
        # Initialize interactive elements
        self.screen_size = (1000, 300)
        self.screen_margins = 20
        self.buttonProportions = 0.3
        self.font = pygame.font.SysFont('Arial', 10)
        self.textColor = (255, 255, 255)
        self.screen = pygame.display.set_mode(self.screen_size)
        [self.buttons, self.button_names] = self.createButtons()
        self.progress_bars = self.create_progress_bars()
        # Signal game start to Open Ephys GUI
        buttonGameOnOff = self.getButton('buttonGameOnOff')
        buttonGameOnOff['callback'](buttonGameOnOff)
        # Release initation rewards
        self.initRewards()
        # Initialize game state update thread
        T_update_states = threading.Thread(target=self.update_states)
        T_update_states.start()
        # Activate main loop
        lastUpdatedState = 0
        while self.mainLoopActive:
            self.clock.tick(self.responseRate)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.mainLoopActive = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for button in self.buttons:
                            if isinstance(button, dict) and button['Rect'].collidepoint(event.pos):
                                threading.Thread(target=button['callback'], args=(button,)).start()
                            elif isinstance(button, list):
                                for subbutton in button[1:]:
                                    if subbutton['Rect'].collidepoint(event.pos):
                                        threading.Thread(target=subbutton['callback'], args=(subbutton,)).start()
            if lastUpdatedState < (time.time() - 1 / float(self.gameRate)):
                lastUpdatedState = time.time()
                T_update_states.join()
                T_update_states = threading.Thread(target=self.update_states)
                T_update_states.start()
        # Quit game when out of gameOn loop
        pygame.quit()

    def run(self):
        threading.Thread(target=self.main_loop).start()

    def stop(self):
        self.mainLoopActive = False
