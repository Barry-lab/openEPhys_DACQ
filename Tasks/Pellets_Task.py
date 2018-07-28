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
from sshScripts import ssh

def activateFEEDER(FEEDER_type, RPiIPBox, RPiUsernameBox, RPiPasswordBox, quantityBox):
    ssh_connection = ssh(str(RPiIPBox.text()), str(RPiUsernameBox.text()), str(RPiPasswordBox.text()))
    if FEEDER_type == 'milk':
        command = 'python milkFeederController.py --openValve ' + str(float(str(quantityBox.text())))
    elif FEEDER_type == 'pellet':
        command = 'python pelletFeederController.py --releasePellet ' + str(int(str(quantityBox.text())))
    ssh_connection.sendCommand(command)
    ssh_connection.disconnect()

def addFeedersToList(self, FEEDER_type, FEEDER_settings=None):
    if FEEDER_settings is None:
        FEEDER_settings = {'ID': '1', 
                           'Present': True, 
                           'Active': True, 
                           'IP': '192.168.0.40', 
                           'Position': np.array([100,50]), 
                           'SignalHz': np.array(4000), 
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
    # Get task settings from text boxes
    TaskSettings = {'LastTravelTime': np.float64(str(self.settings['LastTravelTime'].text())), 
                    'LastTravelSmooth': np.int64(float(str(self.settings['LastTravelSmooth'].text()))), 
                    'LastTravelDist': np.int64(float(str(self.settings['LastTravelDist'].text()))), 
                    'Chewing_TTLchan': np.int64(float(str(self.settings['Chewing_TTLchan'].text()))), 
                    'Username': str(self.settings['Username'].text()), 
                    'Password': str(self.settings['Password'].text()), 
                    'InitPellets': np.int64(float(str(self.settings['InitPellets'].text()))), 
                    'PelletQuantity': np.int64(float(str(self.settings['PelletQuantity'].text()))), 
                    'PelletRewardMinSeparationMean': np.int64(float(str(self.settings['PelletRewardMinSeparationMean'].text()))), 
                    'PelletRewardMinSeparationVariance': np.float64(str(self.settings['PelletRewardMinSeparationVariance'].text())), 
                    'Chewing_Target': np.int64(float(str(self.settings['Chewing_Target'].text()))), 
                    'MaxInactivityDuration': np.int64(float(str(self.settings['MaxInactivityDuration'].text())))}
    # Get FEEDER specific information
    FEEDERs = {}
    for FEEDER_type in self.settings['FEEDERs'].keys():
        if len(self.settings['FEEDERs'][FEEDER_type]) > 0:
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
        else:
            from RecordingManager import show_message
            show_message('No ' + FEEDER_type + ' FEEDERs entered.')
    TaskSettings['FEEDERs'] = FEEDERs

    return TaskSettings

def importSettingsToGUI(self, TaskSettings):
    # First remove all FEEDERs
    self.clearLayout(self.pellet_feeder_settings_layout, keep=1)
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

def SettingsGUI(self):
    self.settings = {}
    self.settings['FEEDERs'] = {'pellet': []}
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
    posHistory = np.array(posHistory)
    posHistory = posHistory[:, :2]
    posHistory[:,0] = smooth_edge_padding(posHistory[:,0], smoothing)
    posHistory[:,1] = smooth_edge_padding(posHistory[:,1], smoothing)
    for npos in range(posHistory.shape[0] - 1):
        prev_pos = posHistory[npos, :]
        curr_pos = posHistory[npos + 1, :]
        distance = euclidean(prev_pos, curr_pos)
        distances.append(distance)
    total_distance = np.sum(np.array(distances))

    return total_distance

def draw_rect_with_border(surface, fill_color, outline_color, position, border=1):
    rect = pygame.Rect(position)
    surface.fill(outline_color, rect)
    surface.fill(fill_color, rect.inflate(-border*2, -border*2))

class Core(object):
    def __init__(self, TaskSettings, TaskIO):
        self.TaskIO = TaskIO
        # Set Task Settings. This should be moved to Task Settings GUI
        self.FEEDERs = TaskSettings.pop('FEEDERs')
        self.TaskSettings = TaskSettings
        # Pre-compute variables
        self.one_second_steps = int(np.round(1 / self.TaskIO['RPIPos'].combPos_update_interval))
        self.distance_steps = int(np.round(self.TaskSettings['LastTravelTime'])) * self.one_second_steps
        # Prepare TTL pulse time list
        self.ttlTimes = []
        self.ttlTimesLock = threading.Lock()
        self.TaskIO['OEmessages'].add_callback(self.append_ttl_pulses)
        # Set up Pellet Rewards
        self.activePfeeders = []
        for ID in sorted(self.FEEDERs['pellet'].keys(), key=int):
            if self.FEEDERs['pellet'][ID]['Active']:
                self.activePfeeders.append(ID)
        self.lastPelletRewardLock = threading.Lock()
        self.lastPelletReward = time.time()
        self.updatePelletMinSepratation()
        # Initialize FEEDERs
        print('Initializing FEEDERs...')
        T_initFEEDER = []
        self.TaskSettings_Lock = threading.Lock()
        for ID in self.activePfeeders:
            T = threading.Thread(target=self.initFEEDER, args=('pellet', ID))
            T.start()
            T_initFEEDER.append(T)
        for T in T_initFEEDER:
            T.join()
        print('Initializing FEEDERs Successful')
        # Initialize counters
        self.game_counters = {'Pellets': {'ID': deepcopy(self.activePfeeders), 
                                          'count': [0] * len(self.activePfeeders)}}
        # Set game speed
        self.responseRate = 60 # Hz
        self.gameRate = 10 # Hz
        # Initialize game
        self.lastRewardLock = threading.Lock()
        self.lastReward = time.time()
        self.rewardInProgressLock = threading.Lock()
        self.rewardInProgress = []
        self.gameOn = False
        self.game_state = 'interval'
        self.mainLoopActive = True
        self.clock = pygame.time.Clock()
        pygame.mixer.pre_init(48000, -16, 2)  # This is necessary for mono sound to work
        pygame.init()

    def initFEEDER(self, FEEDER_type, ID):
        with self.TaskSettings_Lock:
            IP = self.FEEDERs[FEEDER_type][ID]['IP']
            username = self.TaskSettings['Username']
            password = self.TaskSettings['Password']
        try:
            actuator = RewardControl(FEEDER_type, IP, username, password)
            with self.TaskSettings_Lock:
                self.FEEDERs[FEEDER_type][ID]['actuator'] = actuator
                self.FEEDERs[FEEDER_type][ID]['init_successful'] = True
        except Exception as e:
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            print('Error in ' + frameinfo.filename + ' line ' + str(frameinfo.lineno - 3))
            print('initFEEDER failed for feeder: ' + FEEDER_type + ' ' + ID)
            print(e)
            with self.TaskSettings_Lock:
                self.FEEDERs[FEEDER_type][ID]['init_successful'] = False

    def closeAllFEEDERs(self):
        for FEEDER_type in self.FEEDERs.keys():
            for ID in self.FEEDERs[FEEDER_type].keys():
                if 'actuator' in self.FEEDERs[FEEDER_type][ID].keys():
                    if hasattr(self.FEEDERs[FEEDER_type][ID]['actuator'], 'close'):
                        self.FEEDERs[FEEDER_type][ID]['actuator'].close()

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

    def inactivate_feeder(self, FEEDER_type, ID):
        # Remove feeder from active feeder list
        if FEEDER_type == 'pellet':
            self.activePfeeders.remove(ID)
            # Make sure Pellet feeder choice function will reset
            self.old_PelletHistogramParameters = None
        elif FEEDER_type == 'milk':
            self.activeMfeeders.remove(ID)
        # Get FEEDER button on GUI
        if FEEDER_type == 'pellet':
            feeder_button = self.getButton('buttonReleasePellet', ID)
        elif FEEDER_type == 'milk':
            feeder_button = self.getButton('buttonReleaseMilk', ID)
        # Inactivate feeder reward button
        feeder_button['enabled'] = False
        if FEEDER_type == 'milk':
            # If milk feeder, inactivate the corresponding milk trial button
            buttonMilkTrial = self.getButton('buttonMilkTrial', ID)
            buttonMilkTrial['enabled'] = False
        # Send Note to Open Ephys GUI
        OEmessage = 'FEEDER ' + FEEDER_type + ' ' + ID + ' inactivated'
        self.TaskIO['MessageToOE'](OEmessage)

    def releaseReward(self, FEEDER_type, ID, action='undefined', quantity=1):
        if self.FEEDERs[FEEDER_type][ID]['init_successful']:
            # Notify rest of the program that this is onging
            with self.rewardInProgressLock:
                self.rewardInProgress.append(FEEDER_type + ' ' + ID)
            # Make process visible on GUI
            if FEEDER_type == 'pellet':
                feeder_button = self.getButton('buttonReleasePellet', ID)
            elif FEEDER_type == 'milk':
                feeder_button = self.getButton('buttonReleaseMilk', ID)
            feeder_button['button_pressed'] = True
            # Update counter if pellet reward and not user input
            if FEEDER_type == 'pellet' and (action == 'goal_inactivity' or action == 'goal_pellet'):
                idx = self.game_counters['Pellets']['ID'].index(ID)
                self.game_counters['Pellets']['count'][idx] += 1
            # Send command to release reward and wait for positive feedback
            feedback = self.FEEDERs[FEEDER_type][ID]['actuator'].release(quantity)
            if feedback:
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
            else:
                # Send message to Open Ephys GUI
                OEmessage = 'Feeder Failure: ' + FEEDER_type + ' ' + ID + ' ' + action + ' ' + str(quantity)
                self.TaskIO['MessageToOE'](OEmessage)
                # If failed, remove feeder from game and change button(s) red
                self.inactivate_feeder(FEEDER_type, ID)
            # Remove notification of onging reward delivery
            with self.rewardInProgressLock:
                self.rewardInProgress.remove(FEEDER_type + ' ' + ID)
        else:
            OEmessage = 'Feeder Failure: ' + FEEDER_type + ' ' + ID + ' ' + action + ' ' + str(quantity)
            self.TaskIO['MessageToOE'](OEmessage)
            self.inactivate_feeder(FEEDER_type, ID)

    def initRewards(self):
        '''
        Deposits initiation rewards simultaneously from all FEEDERs.
        Completes only once all rewards have been released.
        '''
        T_initRewards = []
        if 'InitPellets' in self.TaskSettings.keys() and self.TaskSettings['InitPellets'] > 0:
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
        if 'InitMilk' in self.TaskSettings.keys() and self.TaskSettings['InitMilk'] > 0:
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
        if self.game_state == 'interval' or self.game_state == 'pellet' or self.game_state == 'milk':
            self.game_state = 'reward_in_progress'
            if len(button['callargs']) == 2:
                [FEEDER_type, ID] = button['callargs']
            elif len(button['callargs']) == 1:
                ID = button['callargs'][0]
                FEEDER_type = 'milk'
            # Release reward from specified feeder and mark as User action
            if 'pellet' == FEEDER_type:
                self.releaseReward(FEEDER_type, ID, 'user', self.TaskSettings['PelletQuantity'])
            elif 'milk' == FEEDER_type:
                self.releaseReward(FEEDER_type, ID, 'user', self.TaskSettings['MilkQuantity'])
        else:
            button['enabled'] = False
            time.sleep(0.5)
            button['enabled'] = True

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

    def getButton(self, button_name, FEEDER_ID=None):
        button = self.buttons[self.button_names.index(button_name)]
        if isinstance(button, list): 
            if FEEDER_ID is None:
                raise ValueError('getButton needs FEEDER_ID for this button_name.')
            else:
                for subbutton in button:
                    if 'text' in subbutton.keys() and subbutton['text'] == FEEDER_ID:
                        button = subbutton
                        break

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
        # Button to release pellet
        buttonReleasePellet = []
        buttonReleasePellet.append({'text': 'Release Pellet'})
        for ID in self.activePfeeders:
            nFeederButton = {'callback': self.buttonReleaseReward_callback, 
                             'callargs': ['pellet', ID], 
                             'text': ID, 
                             'enabled': self.FEEDERs['pellet'][ID]['init_successful'], 
                             'toggled': {'text': ID, 
                                         'color': (0, 128, 0)}}
            buttonReleasePellet.append(nFeederButton)
        buttons.append(buttonReleasePellet)
        button_names.append('buttonReleasePellet')

        return buttons, button_names

    def createButtons(self):
        buttons, button_names = self.defineButtons()
        # Add default color to all buttons
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                if not ('color' in button.keys()):
                    buttons[i]['color'] = (128, 128, 128)
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    if not ('color' in subbutton.keys()):
                        buttons[i][j + 1]['color'] = (128, 128, 128)
        # Add default button un-pressed state
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                if not ('button_pressed' in button.keys()):
                    buttons[i]['button_pressed'] = False
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    if not ('button_pressed' in subbutton.keys()):
                        buttons[i][j + 1]['button_pressed'] = False
        # Add default button enabled state
        for i, button in enumerate(buttons):
            if isinstance(button, dict):
                if not ('enabled' in button.keys()):
                    buttons[i]['enabled'] = True
                if not ('enabled' in button.keys()):
                    buttons[i]['not_enabled_color'] = (255, 0, 0)
            elif isinstance(button, list):
                for j, subbutton in enumerate(button[1:]):
                    if not ('enabled' in subbutton.keys()):
                        buttons[i][j + 1]['enabled'] = True
                    if not ('not_enabled_color' in subbutton.keys()):
                        buttons[i][j + 1]['not_enabled_color'] = (255, 0, 0)
        # Compute button locations
        xpos = self.screen_button_space_start + self.screen_margins
        xlen = self.screen_button_space_width - 2 * self.screen_margins
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
        if button['enabled']:
            # If button is pressed down, use the toggled color and text
            if button['button_pressed']:
                color = button['toggled']['color']
                textRendered = button['toggled']['textRendered']
            else:
                color = button['color']
                textRendered = button['textRendered']
        else:
            # If button is not enabled, use the 'not_enabled_color' and default text
            color = button['not_enabled_color']
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
        game_progress, game_progress_names = self.get_game_progress()
        while len(game_progress) == 0:
            game_progress, game_progress_names = self.get_game_progress()
        # Initialise random color generation
        random.seed(1232)
        rcolor = lambda: random.randint(0,255)
        # Compute progress bar locations
        textSpacing = 3
        textSpace = 10 + textSpacing
        textLines = 3
        xpos = np.linspace(self.screen_progress_bar_space_start + self.screen_margins, self.screen_progress_bar_space_width - self.screen_margins, 2 * len(game_progress))
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
                if self.game_state in gp['game_states']:
                    color = pb['color']
                else:
                    color = (30, 30, 30)
                if gp['complete']:
                    ylen = int(round(pb['Position']['ymaxlen']))
                    ypos = int(round(pb['Position']['ybottompos'] - pb['Position']['ymaxlen']))
                    position = (pb['Position']['xpos'], ypos, pb['Position']['xlen'], ylen)
                    draw_rect_with_border(self.screen, color, (255, 255, 255), position, border=2)
                else:
                    ylen = int(round(gp['percentage'] * pb['Position']['ymaxlen']))
                    ypos = pb['Position']['ybottompos'] - ylen
                    position = (pb['Position']['xpos'], ypos, pb['Position']['xlen'], ylen)
                    pygame.draw.rect(self.screen, color, position, 0)

    def draw_text_info(self):
        # Compute window borders
        xborder = (self.screen_text_info_space_start + self.screen_margins, 
                   self.screen_text_info_space_start + self.screen_text_info_space_width - self.screen_margins)
        yborder = (self.screen_margins, self.screen_size[1] - self.screen_margins)
        # Compute text spacing
        textSpacing = 3
        textSpace = 10 + textSpacing
        # Display game state
        game_state_pos = (xborder[0], yborder[0])
        self.screen.blit(self.renderText('Game State:'), game_state_pos)
        game_state_current_pos = (xborder[0], yborder[0] + textSpace)
        self.screen.blit(self.renderText(str(self.game_state.upper())), game_state_current_pos)
        # Split rest of screen in 5 columns
        title_topedge = game_state_pos[1] + 3 * textSpace
        topedge = game_state_pos[1] + 4 * textSpace
        columnedges = np.linspace(xborder[0], xborder[1], 10)
        columnedges = columnedges[::2]
        # Display pellet feeder IDs
        self.screen.blit(self.renderText('ID'), (columnedges[0], title_topedge))
        for i, ID in enumerate(self.game_counters['Pellets']['ID']):
            self.screen.blit(self.renderText(ID), (columnedges[0], topedge + i * textSpace))
        # Display pellet counts
        self.screen.blit(self.renderText('pellets'), (columnedges[1], title_topedge))
        for i, count in enumerate(self.game_counters['Pellets']['count']):
            self.screen.blit(self.renderText(str(count)), (columnedges[1], topedge + i * textSpace))

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

    def check_if_reward_in_progress(self):
        '''
        Returns True if any rewards are currently being delivered
        '''
        with self.rewardInProgressLock:
            n_rewards_in_progress = len(self.rewardInProgress)
        reward_in_progress = n_rewards_in_progress > 0

        return reward_in_progress

    def get_game_progress(self):
        # Get animal position history
        with self.TaskIO['RPIPos'].combPosHistoryLock:
            posHistory = self.TaskIO['RPIPos'].combPosHistory[-self.distance_steps:]
        if not (None in posHistory):
            self.lastKnownPos = posHistory[-1]
        else:
            posHistory = [self.lastKnownPos] * self.distance_steps
        # Load variables in thread safe way
        with self.lastRewardLock:
            timeSinceLastReward = time.time() - self.lastReward
        with self.lastPelletRewardLock:
            lastPelletRewardTime = self.lastPelletReward
        timeSinceLastPelletReward = time.time() - lastPelletRewardTime
        # Compute all game progress variables
        game_progress = []
        game_progress_names = []
        # Check if animal has been without pellet reward for too long
        game_progress_names.append('inactivity')
        game_progress.append({'name': 'Inactivity', 
                              'game_states': ['interval', 'pellet', 'milk'], 
                              'target': self.TaskSettings['MaxInactivityDuration'], 
                              'status': int(round(timeSinceLastReward)), 
                              'complete': timeSinceLastReward >= self.TaskSettings['MaxInactivityDuration'], 
                              'percentage': timeSinceLastReward / float(self.TaskSettings['MaxInactivityDuration'])})
        # Check if animal has been chewing enough since last reward
        game_progress_names.append('chewing')
        n_chewings = self.number_of_chewings(lastPelletRewardTime)
        game_progress.append({'name': 'Chewing', 
                              'game_states': ['interval'], 
                              'target': self.TaskSettings['Chewing_Target'], 
                              'status': n_chewings, 
                              'complete': n_chewings >= self.TaskSettings['Chewing_Target'], 
                              'percentage': n_chewings / float(self.TaskSettings['Chewing_Target'])})
        # Check if enough time as passed since last pellet reward
        game_progress_names.append('time_since_last_pellet')
        game_progress.append({'name': 'Since Pellet', 
                              'game_states': ['interval'], 
                              'target': self.TaskSettings['PelletRewardMinSeparation'], 
                              'status': int(round(timeSinceLastPelletReward)), 
                              'complete': timeSinceLastPelletReward >= self.TaskSettings['PelletRewardMinSeparation'], 
                              'percentage': timeSinceLastPelletReward / float(self.TaskSettings['PelletRewardMinSeparation'])})
        # Check if has been moving enough in the last few seconds
        game_progress_names.append('mobility')
        total_distance = compute_distance_travelled(posHistory, self.TaskSettings['LastTravelSmooth'])
        game_progress.append({'name': 'Mobility', 
                              'game_states': ['pellet', 'milk'], 
                              'target': self.TaskSettings['LastTravelDist'], 
                              'status': int(round(total_distance)), 
                              'complete': total_distance >= self.TaskSettings['LastTravelDist'], 
                              'percentage': total_distance / float(self.TaskSettings['LastTravelDist'])})

        return game_progress, game_progress_names

    def game_logic(self):
        game_progress, game_progress_names = self.get_game_progress()
        # IF IN INTERVAL STATE
        if self.game_state == 'interval':
            # If in interval state, check if conditions met for milk or pellet state transition
            conditions = {'inactivity': game_progress[game_progress_names.index('inactivity')]['complete'], 
                          'chewing': game_progress[game_progress_names.index('chewing')]['complete'], 
                          'pellet_interval': game_progress[game_progress_names.index('time_since_last_pellet')]['complete']}
            if conditions['inactivity']:
                # If animal has been without any rewards for too long, release pellet reward
                self.game_state = 'reward_in_progress'
                ID = self.choose_pellet_feeder()
                threading.Thread(target=self.releaseReward, 
                                 args=('pellet', ID, 'goal_inactivity', self.TaskSettings['PelletQuantity'])
                                 ).start()
            elif conditions['chewing'] and conditions['pellet_interval']:
                # If conditions are met for pellet but not for milk, change game state to pellet
                self.game_state = 'pellet'
            # Make sure there are feeders for this game state
            if self.game_state == 'pellet' and len(self.activePfeeders) == 0:
                self.game_state = 'interval'
        # IF IN PELLET STATE
        elif self.game_state == 'pellet':
            conditions = {'inactivity': game_progress[game_progress_names.index('inactivity')]['complete'], 
                          'mobility': game_progress[game_progress_names.index('mobility')]['complete']}
            if conditions['inactivity']:
                # If animal has been without any rewards for too long, release pellet reward
                self.game_state = 'reward_in_progress'
                ID = self.choose_pellet_feeder()
                threading.Thread(target=self.releaseReward, 
                                 args=('pellet', ID, 'goal_inactivity', self.TaskSettings['PelletQuantity'])
                                 ).start()
            elif conditions['mobility']:
                # If animal has chewed enough and is mobile enough, release pellet reward
                self.game_state = 'reward_in_progress'
                ID = self.choose_pellet_feeder()
                threading.Thread(target=self.releaseReward, 
                                 args=('pellet', ID, 'goal_pellet', self.TaskSettings['PelletQuantity'])
                                 ).start()
        # IF IN REWARD_IN_PROGRESS STATE
        elif self.game_state == 'reward_in_progress':
            reward_in_progress = self.check_if_reward_in_progress()
            if not reward_in_progress:
                # If reward is not in progress anymore, change game state to interval
                self.game_state = 'interval'
                # Set new variable timer limits
                self.updatePelletMinSepratation()

        return game_progress

    def update_display(self, game_progress):
        with self.screen_Lock:
            self.screen.fill((0, 0, 0))
            self.draw_buttons()
            self.draw_progress_bars(game_progress)
            self.draw_text_info()
            # Update display
            pygame.display.update()

    def update_states(self):
        if self.gameOn:
            game_progress = self.game_logic()
        else:
            game_progress = []
        self.update_display(game_progress)

    def init_main_loop(self):
        # Ensure that Position data is available
        posHistory = []
        while len(posHistory) < self.distance_steps and not (None in posHistory):
            time.sleep(0.1)
            with self.TaskIO['RPIPos'].combPosHistoryLock:
                posHistory = self.TaskIO['RPIPos'].combPosHistory
        self.lastKnownPos = posHistory[-1]
        # Initialize GUI elements
        self.screen_Lock = threading.Lock()
        self.screen_size = (850, 300)
        self.screen_margins = 20
        self.screen_progress_bar_space_start = 0
        self.screen_progress_bar_space_width = 400
        self.screen_button_space_start = 400
        self.screen_button_space_width = 300
        self.screen_text_info_space_start = 700
        self.screen_text_info_space_width = 150
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

    def main_loop(self):
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
                    # Check which button was pressed
                    # The callback function is called for that button, if it is enabled (['enabled'] = True)
                    if event.button == 1:
                        for button in self.buttons:
                            if isinstance(button, dict):
                                if button['Rect'].collidepoint(event.pos) and button['enabled']:
                                    threading.Thread(target=button['callback'], args=(button,)).start()
                            elif isinstance(button, list):
                                for subbutton in button[1:]:
                                    if subbutton['Rect'].collidepoint(event.pos) and subbutton['enabled']:
                                        threading.Thread(target=subbutton['callback'], args=(subbutton,)).start()
            # Update the game state and display if enough time has passed since last update
            if lastUpdatedState < (time.time() - 1 / float(self.gameRate)):
                if T_update_states.is_alive():
                    print('Game State Processing thread pileup! ' + time.asctime())
                T_update_states.join() # Ensure the previous state update thread has finished
                T_update_states = threading.Thread(target=self.update_states)
                T_update_states.start()
                lastUpdatedState = time.time()
        # Quit game when out of gameOn loop
        with self.screen_Lock:
            pygame.quit()

    def run(self):
        self.init_main_loop()
        threading.Thread(target=self.main_loop).start()

    def stop(self):
        # Make sure reward delivery is finished before closing game processes
        while self.game_state == 'reward_in_progress':
            print('Waiting for reward_in_progress to finish...')
            time.sleep(1)
        # Stop main game loop
        self.mainLoopActive = False
        # Send message to Open Ephys
        self.gameOn = False
        OEmessage = 'Game On: ' + str(self.gameOn)
        self.TaskIO['MessageToOE'](OEmessage)
        # Close FEEDER connections
        print('Closing FEEDER connections...')
        self.closeAllFEEDERs()
        print('Closing FEEDER connections successful.')
