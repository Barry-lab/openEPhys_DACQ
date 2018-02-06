# This is a task

import pygame
import numpy as np
import threading
from RPiInterface import RewardControl
import time
from scipy.spatial.distance import euclidean
import random
from PyQt4 import QtGui, QtCore

def activateFEEDER(FEEDER_type, RPiIPBox, RPiUsernameBox, RPiPasswordBox, quantityBox):
    feeder = RewardControl(FEEDER_type, str(RPiIPBox.text()), 
                           str(RPiUsernameBox.text()), str(RPiPasswordBox.text()))
    feeder.release(float(str(quantityBox.text())))
    feeder.close()

def addFeedersToList(self, FEEDER_type, FEEDER_settings=None):
    if FEEDER_settings is None:
        FEEDER_settings = {'ID': '1', 
                           'Present': True, 
                           'Active': True, 
                           'IP': '192.168.0.40', 
                           'Position': '100,50', 
                           'SignalHz': '2800'}
    # Create interface for interacting with this FEEDER
    FEEDER = {'Type': FEEDER_type}
    vbox = QtGui.QVBoxLayout()
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('ID:'))
    FEEDER['ID'] = QtGui.QLineEdit(FEEDER_settings['ID'])
    FEEDER['ID'].setMaximumWidth(40)
    hbox.addWidget(FEEDER['ID'])
    FEEDER['Present'] = QtGui.QCheckBox('Present')
    FEEDER['Present'].setChecked(FEEDER_settings['Present'])
    hbox.addWidget(FEEDER['Present'])
    FEEDER['Active'] = QtGui.QCheckBox('Active')
    FEEDER['Active'].setChecked(FEEDER_settings['Active'])
    hbox.addWidget(FEEDER['Active'])
    hbox.addWidget(QtGui.QLabel('IP:'))
    FEEDER['IP'] = QtGui.QLineEdit(FEEDER_settings['IP'])
    hbox.addWidget(FEEDER['IP'])
    vbox.addLayout(hbox)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Position:'))
    FEEDER['Position'] = QtGui.QLineEdit(FEEDER_settings['Position'])
    FEEDER['Position'].setMinimumWidth(70)
    hbox.addWidget(FEEDER['Position'])
    hbox.addWidget(QtGui.QLabel('Signal (Hz):'))
    FEEDER['SignalHz'] = QtGui.QLineEdit(FEEDER_settings['SignalHz'])
    hbox.addWidget(FEEDER['SignalHz'])
    testButton = QtGui.QPushButton('Test')
    testButton.setMaximumWidth(40)
    FEEDER['TestQuantity'] = QtGui.QLineEdit('1')
    FEEDER['TestQuantity'].setMaximumWidth(40)
    testButton.clicked.connect(lambda: activateFEEDER(FEEDER_type, FEEDER['IP'], 
                                                      self.settings['Username'], 
                                                      self.settings['Password'], 
                                                      FEEDER['TestQuantity']))
    hbox.addWidget(testButton)
    hbox.addWidget(FEEDER['TestQuantity'])
    vbox.addLayout(hbox)
    frame = QtGui.QFrame()
    frame.setLayout(vbox)
    frame.setFrameStyle(3)
    frame.setMaximumHeight(90)
    if FEEDER_type == 'pellet':
        self.pellet_feeder_settings_layout.addWidget(frame)
    elif FEEDER_type == 'milk':
        self.milk_feeder_settings_layout.addWidget(frame)
    self.settings['FEEDERs'].append(FEEDER)

def setDoubleHBoxStretch(hbox):
    hbox.setStretch(0,2)
    hbox.setStretch(1,1)

    return hbox

def exportSettings(self):
    TaskSettings = {'LastTravelTime': float(str(self.settings['LastTravelTime'].text())), 
                    'LastTravelSmooth': int(float(str(self.settings['LastTravelSmooth'].text()))), 
                    'LastTravelDist': int(float(str(self.settings['LastTravelDist'].text()))), 
                    'PelletMilkRatio': float(str(self.settings['PelletMilkRatio'].text())), 
                    'Chewing_TTLchan': int(float(str(self.settings['Chewing_TTLchan'].text()))), 
                    'Username': str(self.settings['Username'].text()), 
                    'Password': str(self.settings['Password'].text()), 
                    'pelletGameOn': self.settings['pelletGameOn'].isChecked(), 
                    'InitPellets': int(float(str(self.settings['InitPellets'].text()))), 
                    'PelletQuantity': int(float(str(self.settings['PelletQuantity'].text()))), 
                    'PelletRewardMinSeparationMean': int(float(str(self.settings['PelletRewardMinSeparationMean'].text()))), 
                    'PelletRewardMinSeparationVariance': float(str(self.settings['PelletRewardMinSeparationVariance'].text())), 
                    'Chewing_Target': int(float(str(self.settings['Chewing_Target'].text()))), 
                    'MaxInactivityDuration': int(float(str(self.settings['MaxInactivityDuration'].text()))), 
                    'MilkTrialFailPenalty': int(float(str(self.settings['MilkTrialFailPenalty'].text()))), 
                    'milkGameOn': self.settings['milkGameOn'].isChecked(), 
                    'InitMilk': float(str(self.settings['InitMilk'].text())), 
                    'MilkQuantity': int(float(str(self.settings['MilkQuantity'].text()))), 
                    'MilkTrialMinSeparationMean': int(float(str(self.settings['MilkTrialMinSeparationMean'].text()))), 
                    'MilkTrialMinSeparationVariance': float(str(self.settings['MilkTrialMinSeparationVariance'].text())), 
                    'MilkTaskMinStartDistance': int(float(str(self.settings['MilkTaskMinStartDistance'].text()))), 
                    'MilkTaskMinGoalDistance': int(float(str(self.settings['MilkTaskMinGoalDistance'].text()))), 
                    'MilkTrialMaxDuration': int(float(str(self.settings['MilkTrialMaxDuration'].text())))}
    FEEDERs = []
    for feeder in self.settings['FEEDERs']:
        FEEDERs.append({'Type': feeder['Type'], 
                        'ID': int(str(feeder['ID'].text())), 
                        'Present': feeder['Present'].isChecked(), 
                        'Active': feeder['Active'].isChecked(), 
                        'IP': str(feeder['IP'].text()), 
                        'Position': map(int, str(feeder['Position'].text()).split(',')), 
                        'SignalHz': int(float(str(feeder['SignalHz'].text())))})
    TaskSettings['FEEDERs'] = FEEDERs

    return TaskSettings

def importSettings(self, TaskSettings):
    # First remove all FEEDERs
    self.clearLayout(self.pellet_feeder_settings_layout, keep=1)
    self.clearLayout(self.milk_feeder_settings_layout, keep=1)
    for key in TaskSettings.keys():
        if isinstance(TaskSettings[key], bool):
            self.settings[key].setChecked(TaskSettings[key])
        elif key == 'FEEDERs':
            self.settings['FEEDERs'] = []
            for feeder in TaskSettings[key]:
                FEEDER_type = feeder['Type']
                FEEDER_settings = {'ID': str(feeder['ID']), 
                                   'Present': feeder['Present'], 
                                   'Active': feeder['Active'], 
                                   'IP': feeder['IP'], 
                                   'Position': ','.join(map(str,feeder['Position'])), 
                                   'SignalHz': str(feeder['SignalHz'])}
                addFeedersToList(self, FEEDER_type, FEEDER_settings)
        else:
            self.settings[key].setText(str(TaskSettings[key]))

def createSineWaveSound(frequency):
    # Calling this function must be preceded by calling the following lines
    # for pygame to recognise the mono sound
    # pygame.mixer.pre_init(44100, -16, 1) # here 44100 needs to match the sampleRate in this function
    # pygame.init()
    sampleRate = 44100 # Must be the same as in the line 
    peak = 4096 # 4096 : the peak ; volume ; loudness
    arr = np.array([peak * np.sin(2.0 * np.pi * frequency * x / sampleRate) for x in range(0, sampleRate)]).astype(np.int16)
    sound = pygame.sndarray.make_sound(arr)

    # Use sound.play(-1) to start sound. (-1) means it is in infinite loop
    # Use sound.stop() to stop the sound

    return sound

def SettingsGUI(self):
    self.settings = {}
    self.settings['FEEDERs'] = []
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
    self.exportSettings = exportSettings
    self.importSettings = importSettings

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
        self.arena_size = self.TaskIO['RPIPos'].RPiSettings['arena_size']
        # Set Task Settings. This should be moved to Task Settings GUI
        self.TaskSettings = TaskSettings
        # Pre-compute variables
        self.one_second_steps = int(np.round(1 / self.TaskIO['RPIPos'].combPos_update_interval))
        self.distance_steps = int(np.round(self.TaskSettings['LastTravelTime'])) * self.one_second_steps
        self.max_distance_in_arena = int(round(np.hypot(self.arena_size[0], self.arena_size[1])))
        # Prepare TTL pulse time list
        self.ttlTimes = []
        self.ttlTimesLock = threading.Lock()
        self.TaskIO['OEmessages'].add_callback(self.append_ttl_pulses)
        # Initialize FEEDERs
        print('Initializing FEEDERs...')
        T_initFEEDER = []
        self.FEEDER_Lock = threading.Lock()
        for n_feeder, feeder in enumerate(self.TaskSettings['FEEDERs']):
            if feeder['Active']:
                T = threading.Thread(target=self.initFEEDER, args=[n_feeder])
                T.start()
        for T in T_initFEEDER:
            T.join()
        print('Initializing FEEDERs Successful')
        # Set up Pellet Rewards
        self.pelletGameOn = self.TaskSettings['pelletGameOn']
        if self.pelletGameOn:
            self.activePfeeders = []
            for n_feeder, feeder in enumerate(self.TaskSettings['FEEDERs']):
                if feeder['Type'] == 'pellet' and feeder['Active']:
                    self.activePfeeders.append(n_feeder)
            self.lastPelletReward = time.time()
            self.updatePelletMinSepratation()
        # Set up Milk Rewards
        self.milkGameOn = self.TaskSettings['milkGameOn']
        if self.milkGameOn:
            self.activeMfeeders = []
            for n_feeder, feeder in enumerate(self.TaskSettings['FEEDERs']):
                if feeder['Type'] == 'milk' and feeder['Active']:
                    self.activeMfeeders.append(n_feeder)
            self.lastMilkTrial = time.time()
            self.updateMilkTrialMinSepratation()
            self.milkTrialFailTime = time.time() - self.TaskSettings['MilkTrialFailPenalty']
        self.milkTrialOn = False
        # Set game speed
        self.responseRate = 60 # Hz
        self.gameRate = 10 # Hz
        # Initialize game
        self.lastReward = time.time()
        self.gameOn = True
        self.mainLoopActive = True
        self.clock = pygame.time.Clock()
        pygame.mixer.pre_init(44100, -16, 1)  # This is necessary for mono sound to work
        pygame.init()

    def initFEEDER(self, n_feeder):
        with self.FEEDER_Lock:
            FEEDER_type = self.TaskSettings['FEEDERs'][n_feeder]['Type']
            IP = self.TaskSettings['FEEDERs'][n_feeder]['IP']
            username = self.TaskSettings['Username']
            password = self.TaskSettings['Password']
        actuator = RewardControl(FEEDER_type, IP, username, password)
        with self.FEEDER_Lock:
            self.TaskSettings['FEEDERs'][n_feeder]['actuator'] = actuator

    def renderText(self, text):
        renderedText = self.font.render(text, True, self.textColor)

        return renderedText

    def append_ttl_pulses(self, message):
        parts = message.split()
        if parts[2] == str(self.TaskSettings['Chewing_TTLchan']):
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

    def releaseReward(self, n_feeder, action='undefined', quantity=1):
        self.TaskSettings['FEEDERs'][n_feeder]['actuator'].release(quantity)
        # Send message to Open Ephys GUI
        rewardType = self.TaskSettings['FEEDERs'][n_feeder]['Type']
        feederID = self.TaskSettings['FEEDERs'][n_feeder]['ID']
        OEmessage = 'Reward ' + rewardType + ' ' + str(feederID + 1) + ' ' + action + ' ' + str(quantity)
        self.TaskIO['MessageToOE'](OEmessage)
        # Reset last reward timer
        self.lastReward = time.time()
        if 'pellet' == self.TaskSettings['FEEDERs'][n_feeder]['Type']:
            self.lastPelletReward = time.time()

    def initRewards(self):
        if self.pelletGameOn and 'InitPellets' in self.TaskSettings.keys() and self.TaskSettings['InitPellets'] > 0:
            minPellets = int(np.floor(float(self.TaskSettings['InitPellets']) / len(self.activePfeeders)))
            extraPellets = np.mod(self.TaskSettings['InitPellets'], len(self.activePfeeders))
            n_pellets_Feeders = minPellets * np.ones(len(self.activePfeeders), dtype=np.int16)
            n_pellets_Feeders[:extraPellets] = n_pellets_Feeders[:extraPellets] + 1
            for feeder_list_idx, n_pellets in enumerate(n_pellets_Feeders):
                n_feeder = self.activePfeeders[feeder_list_idx]
                self.releaseReward(n_feeder, 'game_init', n_pellets)
        if self.milkGameOn and 'InitMilk' in self.TaskSettings.keys() and self.TaskSettings['InitMilk'] > 0:
            for n_feeder in self.activeMfeeders:
                self.releaseReward(n_feeder, 'game_init', self.TaskSettings['InitMilk'])

    def buttonGameOnOff_callback(self):
        # Switch Game On and Off
        self.gameOn = not self.gameOn
        OEmessage = 'Game On: ' + str(self.gameOn)
        self.TaskIO['MessageToOE'](OEmessage)

    def buttonReleaseReward_callback(self, n_feeder):
        # Release reward from specified feeder and mark as User action
        if 'pellet' == self.TaskSettings['FEEDERs'][n_feeder]['Type']:
            self.releaseReward(n_feeder, 'user', self.TaskSettings['PelletQuantity'])
        elif 'milk' == self.TaskSettings['FEEDERs'][n_feeder]['Type']:
            self.releaseReward(n_feeder, 'user', self.TaskSettings['MilkQuantity'])

    def buttonManualPellet_callback(self):
        # Update last reward time
        self.lastPelletReward = time.time()
        self.lastReward = self.lastPelletReward
        # Send message to Open Ephys GUI
        OEmessage = 'Reward pelletManual'
        self.TaskIO['MessageToOE'](OEmessage)

    def buttonMilkTrial_callback(self, n_feeder):
        # Starts the trial with specific feeder as goal
        self.start_milkTrial(n_feeder=n_feeder, action='user')

    def defineButtons(self):
        # Add or remove buttons from this function
        # Create new callbacks for new button if necessary
        # Note default settings applied in self.createButtons()
        buttons = []
        # Game On/Off button
        buttonGameOnOff = {'callback': self.buttonGameOnOff_callback, 
                           'text': 'Game On', 
                           'toggled': {'text': 'Game Off', 
                                       'color': (0, 0, 255)}}
        buttons.append(buttonGameOnOff)
        # Button to mark manually released pellet
        buttonManualPellet = {'callback': self.buttonManualPellet_callback, 
                              'text': 'Manual Pellet'}
        buttons.append(buttonManualPellet)
        if self.pelletGameOn: # The buttons are only active if pellet FEEDERs available
            # Button to release pellet
            buttonReleasePellet = []
            buttonReleasePellet.append({'text': 'Release Pellet'})
            for n_feeder in self.activePfeeders:
                nFeederButton = {'callback': self.buttonReleaseReward_callback, 
                                 'callargs': [n_feeder], 
                                 'text': str(self.TaskSettings['FEEDERs'][n_feeder]['ID'] + 1)}
                buttonReleasePellet.append(nFeederButton)
            buttons.append(buttonReleasePellet)
        if self.milkGameOn: # The buttons are only active if milk FEEDERs available
            # Button to start milkTrial
            buttonReleasePellet = []
            buttonReleasePellet.append({'text': 'Milk Trial'})
            for n_feeder in self.activeMfeeders:
                nFeederButton = {'callback': self.buttonMilkTrial_callback, 
                                 'callargs': [n_feeder], 
                                 'text': str(self.TaskSettings['FEEDERs'][n_feeder]['ID'] + 1)}
                buttonReleasePellet.append(nFeederButton)
            buttons.append(buttonReleasePellet)
            # Button to release milk
            buttonReleasePellet = []
            buttonReleasePellet.append({'text': 'Deposit Milk'})
            for n_feeder in self.activeMfeeders:
                nFeederButton = {'callback': self.buttonReleaseReward_callback, 
                                 'callargs': [n_feeder], 
                                 'text': str(self.TaskSettings['FEEDERs'][n_feeder]['ID'] + 1)}
                buttonReleasePellet.append(nFeederButton)
            buttons.append(buttonReleasePellet)

        return buttons

    def createButtons(self):
        buttons = self.defineButtons()
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

        return buttons

    def draw_buttons(self):
        # Draw all buttons here
        # When drawing buttons in Pushed state color, reset the state unless Toggled in keys
        for i, button in enumerate(self.buttons):
            if isinstance(button, dict):
                textRendered = button['textRendered']
                if button['button_pressed']:
                    if 'toggled' in button.keys():
                        color = button['toggled']['color']
                        textRendered = button['toggled']['textRendered']
                    else:
                        self.buttons[i]['button_pressed'] = False
                        color = (int(button['color'][0] * 0.5), int(button['color'][1] * 0.5), int(button['color'][2] * 0.5))
                else:
                    color = button['color']
                pygame.draw.rect(self.screen, color, button['Position'], 0)
                self.screen.blit(textRendered, button['Position'][:2])
            elif isinstance(button, list):
                self.screen.blit(button[0]['textRendered'], button[0]['Position'][:2])
                for j, subbutton in enumerate(button[1:]):
                    if subbutton['button_pressed']:
                        self.buttons[i][j + 1]['button_pressed'] = False
                        color = (int(subbutton['color'][0] * 0.5), int(subbutton['color'][1] * 0.5), int(subbutton['color'][2] * 0.5))
                    else:
                        color = subbutton['color']
                    pygame.draw.rect(self.screen, subbutton['color'], subbutton['Position'], 0)
                    self.screen.blit(subbutton['textRendered'], subbutton['Position'][:2])

    def create_progress_bars(self):
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
        n_feeder = random.randint(0, len(self.activePfeeders) - 1)
        n_feeder = self.activePfeeders[n_feeder]

        return n_feeder

    def start_milkTrial(self, n_feeder=None, action='undefined'):
        if n_feeder is None:
            # If no feeder specified, pick one at random
            n_feeder = random.randint(0, len(self.activeMfeeders) - 1)
            n_feeder = self.activeMfeeders[n_feeder]
        self.n_feeder_milkTrial = n_feeder
        self.updateMilkTrialMinSepratation()
        # These settings put the game_logic into milkTrial mode
        self.milkTrialOn = True
        self.milkTrialTone = createSineWaveSound(self.TaskSettings['FEEDERs'][n_feeder]['SignalHz'])
        self.milkTrialTone.play(-1)
        self.lastMilkTrial = time.time()
        feederID = self.TaskSettings['FEEDERs'][self.n_feeder_milkTrial]['ID']
        OEmessage = 'milkTrialStart ' + action + ' ' + str(feederID + 1)
        self.TaskIO['MessageToOE'](OEmessage)

    def stop_milkTrial(self, successful):
        self.milkTrialTone.stop()
        self.milkTrialOn = False
        OEmessage = 'milkTrialEnd Success:' + str(successful)
        self.TaskIO['MessageToOE'](OEmessage)
        if successful:
            self.releaseReward(self.n_feeder_milkTrial, 'goal_milk', self.TaskSettings['MilkQuantity'])
        else:
            self.milkTrialFailTime = time.time()

    def game_logic(self):
        game_progress = []
        # Get animal position history
        with self.TaskIO['RPIPos'].combPosHistoryLock:
            posHistory = self.TaskIO['RPIPos'].combPosHistory[-self.distance_steps:]
        if self.pelletGameOn: # The following progress is monitored only if pellet reward used
            # Check if animal has been without pellet reward for too long
            timeSinceLastReward = time.time() - self.lastReward
            game_progress.append({'name': 'Inactivity', 
                                  'goals': ['inactivity'], 
                                  'target': self.TaskSettings['MaxInactivityDuration'], 
                                  'status': int(round(timeSinceLastReward)), 
                                  'complete': timeSinceLastReward >= self.TaskSettings['MaxInactivityDuration'], 
                                  'percentage': timeSinceLastReward / float(self.TaskSettings['MaxInactivityDuration'])})
            # Check if enough time as passed since last pellet reward
            timeSinceLastReward = time.time() - self.lastPelletReward
            game_progress.append({'name': 'Since Pellet', 
                                  'goals': ['pellet', 'milkTrialStart'], 
                                  'target': self.TaskSettings['PelletRewardMinSeparation'], 
                                  'status': int(round(timeSinceLastReward)), 
                                  'complete': timeSinceLastReward >= self.TaskSettings['PelletRewardMinSeparation'], 
                                  'percentage': timeSinceLastReward / float(self.TaskSettings['PelletRewardMinSeparation'])})
            # Check if animal has been chewing enough since last reward
            n_chewings = self.number_of_chewings(self.lastPelletReward)
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
            for n_feeder in self.activeMfeeders:
                distances.append(euclidean(np.array(posHistory[-1][:2]), np.array(self.TaskSettings['FEEDERs'][n_feeder]['Position'])))
            minDistance = min(distances)
            game_progress.append({'name': 'Milk Distance', 
                                  'goals': ['milkTrialStart'], 
                                  'target': self.TaskSettings['MilkTaskMinStartDistance'], 
                                  'status': int(round(minDistance)), 
                                  'complete': minDistance >= self.TaskSettings['MilkTaskMinStartDistance'], 
                                  'percentage': minDistance / float(self.TaskSettings['MilkTaskMinStartDistance'])})
            if self.milkTrialOn:
                # Check if animal is close enough to goal location
                distance = euclidean(np.array(posHistory[-1][:2]), np.array(self.TaskSettings['FEEDERs'][self.n_feeder_milkTrial]['Position']))
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
        if not self.milkTrialOn:
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
                    n_feeder = self.choose_pellet_feeder()
                    self.releaseReward(n_feeder, 'goal_pellet', self.TaskSettings['PelletQuantity'])
                    self.updatePelletMinSepratation()
                else:
                    self.start_milkTrial(action='goal_milkTrialStart')
            elif self.milkGameOn and all(milkTrialStart_complete):
                # If conditions met, start milkTrial
                self.start_milkTrial(action='goal_milkTrialStart')
            elif self.pelletGameOn and all(pellet_status_complete):
                # If conditions met, release pellet reward
                n_feeder = self.choose_pellet_feeder()
                self.releaseReward(n_feeder, 'goal_pellet', self.TaskSettings['PelletQuantity'])
                self.updatePelletMinSepratation()
            elif self.pelletGameOn and all(inactivity_complete):
                # If animal has been inactive and without pellet rewards, release pellet reward
                n_feeder = self.activePfeeders[random.randint(0,len(self.activePfeeders) - 1)]
                self.releaseReward(n_feeder, 'goal_inactivity', self.TaskSettings['PelletQuantity'])
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

    def update_display(self):
        self.screen.fill((0, 0, 0))
        self.draw_buttons()
        self.draw_progress_bars(self.game_progress)
        # Update display
        pygame.display.update()


    def update_states(self):
        if self.gameOn:
            self.game_progress = self.game_logic()
        self.update_display()

    def main_loop(self):
        # Ensure that Position data is available
        posHistory = []
        while len(posHistory) < self.distance_steps:
            time.sleep(0.5)
            with self.TaskIO['RPIPos'].combPosHistoryLock:
                posHistory = self.TaskIO['RPIPos'].combPosHistory
        # Initialize interactive elements
        self.screen_size = (800, 300)
        self.screen_margins = 20
        self.buttonProportions = 0.2
        self.font = pygame.font.SysFont('Arial', 10)
        self.textColor = (255, 255, 255)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.buttons = self.createButtons()
        self.progress_bars = self.create_progress_bars()
        # Signal game start to Open Ephys GUI
        OEmessage = 'Game On: ' + str(self.gameOn)
        self.TaskIO['MessageToOE'](OEmessage)
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
                        for i, button in enumerate(self.buttons):
                            if isinstance(button, dict) and button['Rect'].collidepoint(event.pos):
                                if 'toggled' in button.keys():
                                    self.buttons[i]['button_pressed'] = not button['button_pressed']
                                else:
                                    self.buttons[i]['button_pressed'] = True
                                if 'callargs' in button.keys():
                                    button['callback'](*button['callargs'])
                                else:
                                    button['callback']()
                            elif isinstance(button, list):
                                for j, subbutton in enumerate(button[1:]):
                                    if subbutton['Rect'].collidepoint(event.pos):
                                        self.buttons[i][j + 1]['button_pressed'] = True
                                        if 'callargs' in subbutton.keys():
                                            subbutton['callback'](*subbutton['callargs'])
                                        else:
                                            subbutton['callback']()
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