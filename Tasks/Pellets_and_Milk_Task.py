# This is a task

import pygame
import numpy as np
import threading
from RPiInterface import RewardControl
from time import asctime, time, sleep
from scipy.spatial.distance import euclidean
import random
from PyQt4 import QtGui, QtCore
from copy import deepcopy
from audioSignalGenerator import createAudioSignal
from sshScripts import ssh
from HelperFunctions import show_message

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
                           'SignalHzWidth': np.array(500), 
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
        hbox.addWidget(QtGui.QLabel('W:'))
        if not 'SignalHzWidth' in FEEDER_settings.keys():
            print('Remove this section in Pellets_and_Milk_Task.py when settings resaved!')
            FEEDER_settings['SignalHzWidth'] = np.array(500)
        FEEDER['SignalHzWidth'] = QtGui.QLineEdit(str(FEEDER_settings['SignalHzWidth']))
        hbox.addWidget(FEEDER['SignalHzWidth'])
        hbox.addWidget(QtGui.QLabel('M:'))
        FEEDER['ModulHz'] = QtGui.QLineEdit(str(FEEDER_settings['ModulHz']))
        hbox.addWidget(FEEDER['ModulHz'])
        playSignalButton = QtGui.QPushButton('Play')
        playSignalButton.setMaximumWidth(40)
        playSignalButton.clicked.connect(lambda: playSiginal(FEEDER['SignalHz'], 
                                                             FEEDER['SignalHzWidth'], 
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
                    'PelletMilkRatio': np.float64(str(self.settings['PelletMilkRatio'].text())), 
                    'Chewing_TTLchan': np.int64(float(str(self.settings['Chewing_TTLchan'].text()))), 
                    'Username': str(self.settings['Username'].text()), 
                    'Password': str(self.settings['Password'].text()), 
                    'lightSignalIntensity': np.int64(str(self.settings['lightSignalIntensity'].text())), 
                    'lightSignalDelay': np.float64(str(self.settings['lightSignalDelay'].text())), 
                    'NegativeAudioSignal': np.float64(str(self.settings['NegativeAudioSignal'].text())), 
                    'InitPellets': np.int64(float(str(self.settings['InitPellets'].text()))), 
                    'PelletQuantity': np.int64(float(str(self.settings['PelletQuantity'].text()))), 
                    'PelletRewardMinSeparationMean': np.int64(float(str(self.settings['PelletRewardMinSeparationMean'].text()))), 
                    'PelletRewardMinSeparationVariance': np.float64(str(self.settings['PelletRewardMinSeparationVariance'].text())), 
                    'Chewing_Target': np.int64(float(str(self.settings['Chewing_Target'].text()))), 
                    'MaxInactivityDuration': np.int64(float(str(self.settings['MaxInactivityDuration'].text()))), 
                    'MilkTrialFailPenalty': np.int64(float(str(self.settings['MilkTrialFailPenalty'].text()))), 
                    'InitMilk': np.float64(str(self.settings['InitMilk'].text())), 
                    'MilkQuantity': np.float64(str(self.settings['MilkQuantity'].text())), 
                    'MilkTrialMinSeparationMean': np.int64(float(str(self.settings['MilkTrialMinSeparationMean'].text()))), 
                    'MilkTrialMinSeparationVariance': np.float64(str(self.settings['MilkTrialMinSeparationVariance'].text())), 
                    'MilkTaskMinStartDistance': np.int64(float(str(self.settings['MilkTaskMinStartDistance'].text()))), 
                    'MilkTaskMinGoalDistance': np.int64(float(str(self.settings['MilkTaskMinGoalDistance'].text()))), 
                    'MilkTaskMinGoalAngularDistance': np.int64(float(str(self.settings['MilkTaskMinGoalAngularDistance'].text()))), 
                    'MilkTaskGoalAngularDistanceTime': np.float64(float(str(self.settings['MilkTaskGoalAngularDistanceTime'].text()))), 
                    'MilkTrialMaxDuration': np.int64(float(str(self.settings['MilkTrialMaxDuration'].text())))}
    # Get radio button selection
    for key in self.settings['AudioSignalMode'].keys():
        if self.settings['AudioSignalMode'][key].isChecked():
            TaskSettings['AudioSignalMode'] = key
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
                    FEEDERs[FEEDER_type][IDs[-1]]['SignalHzWidth'] = np.int64(float(str(feeder['SignalHzWidth'].text())))
                    FEEDERs[FEEDER_type][IDs[-1]]['ModulHz'] = np.int64(float(str(feeder['ModulHz'].text())))
            # Check if there are duplicates of FEEDER IDs
            if any(IDs.count(ID) > 1 for ID in IDs):
                raise ValueError('Duplicates of IDs in ' + FEEDER_type + ' feeders!')
        else:
            show_message('No ' + FEEDER_type + ' FEEDERs entered.')
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
        elif key == 'AudioSignalMode':
            for mode_key in self.settings['AudioSignalMode'].keys():
                if TaskSettings['AudioSignalMode'] == mode_key:
                    self.settings['AudioSignalMode'][mode_key].setChecked(True)
                else:
                    self.settings['AudioSignalMode'][mode_key].setChecked(False)
        elif key == 'FEEDERs':
            for FEEDER_type in TaskSettings['FEEDERs'].keys():
                for ID in sorted(TaskSettings['FEEDERs'][FEEDER_type].keys(), key=int):
                    FEEDER_settings = TaskSettings['FEEDERs'][FEEDER_type][ID]
                    addFeedersToList(self, FEEDER_type, FEEDER_settings)
        elif key in self.settings.keys():
            self.settings[key].setText(str(TaskSettings[key]))

def playSiginal(frequency, frequency_band_width, modulation_frequency):
    if type(frequency) == QtGui.QLineEdit:
        frequency = np.int64(float(str(frequency.text())))
    if type(frequency_band_width) == QtGui.QLineEdit:
        frequency_band_width = np.int64(float(str(frequency_band_width.text())))
    if type(modulation_frequency) == QtGui.QLineEdit:
        modulation_frequency = np.int64(float(str(modulation_frequency.text())))
    # Initialize pygame for playing sound
    pygame.mixer.pre_init(48000, -16, 2)
    pygame.init()
    # Get sound
    sound = createAudioSignal(frequency, frequency_band_width, modulation_frequency)
    # Play 2 seconds of the sound
    sound.play(-1, maxtime=2000)

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
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),4,0)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Raspberry Pi usernames'))
    self.settings['Username'] = QtGui.QLineEdit('pi')
    hbox.addWidget(self.settings['Username'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),0,1)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Raspberry Pi passwords'))
    self.settings['Password'] = QtGui.QLineEdit('raspberry')
    hbox.addWidget(self.settings['Password'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),1,1)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Audio Signal Mode'))
    self.settings['AudioSignalMode'] = {'ambient': QtGui.QRadioButton('Ambient'), 
                                        'localised': QtGui.QRadioButton('Localised')}
    self.settings['AudioSignalMode']['ambient'].setChecked(True)
    hbox.addWidget(self.settings['AudioSignalMode']['ambient'])
    hbox.addWidget(self.settings['AudioSignalMode']['localised'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),2,1)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Light Signal intensity (0 - 100)'))
    self.settings['lightSignalIntensity'] = QtGui.QLineEdit('100')
    hbox.addWidget(self.settings['lightSignalIntensity'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),3,1)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Light Signal delay (s)'))
    self.settings['lightSignalDelay'] = QtGui.QLineEdit('0')
    hbox.addWidget(self.settings['lightSignalDelay'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),4,1)
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Negative Audio Feedback (s)'))
    self.settings['NegativeAudioSignal'] = QtGui.QLineEdit('0')
    hbox.addWidget(self.settings['NegativeAudioSignal'])
    self.task_general_settings_layout.addLayout(setDoubleHBoxStretch(hbox),5,1)
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
    hbox.addWidget(QtGui.QLabel('Minimum Goal angular distance (deg)'))
    self.settings['MilkTaskMinGoalAngularDistance'] = QtGui.QLineEdit('45')
    hbox.addWidget(self.settings['MilkTaskMinGoalAngularDistance'])
    vbox.addLayout(setDoubleHBoxStretch(hbox))
    hbox = QtGui.QHBoxLayout()
    hbox.addWidget(QtGui.QLabel('Goal angular distance time (s)'))
    self.settings['MilkTaskGoalAngularDistanceTime'] = QtGui.QLineEdit('2')
    hbox.addWidget(self.settings['MilkTaskGoalAngularDistanceTime'])
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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def compute_mean_movement_vector(posHistory):
    posHistory = np.array(posHistory)
    posHistory = posHistory[:, :2]
    posVectors = posHistory[1:, :] - posHistory[:-1, :]
    posVector = np.mean(posVectors, axis=0)

    return posVector

def compute_mean_posHistory(posHistory):
    posHistory = np.array(posHistory)
    posHistory = posHistory[:, :2]
    mean_posHistory = np.mean(posHistory, axis=0)

    return mean_posHistory

def compute_movement_angular_distance_to_target(posHistory, target_location):
    '''
    Computes angular distance between mean movement vector and direct path to target location.
    Outputs None if norm of mean movement vector is 0.
    '''
    posVector = compute_mean_movement_vector(posHistory)
    if np.linalg.norm(posVector) > 0:
        targetVector = target_location - posHistory[-1][:2]
        angle_rad = angle_between(posVector, targetVector)
        angle = np.rad2deg(angle_rad)
    else:
        angle = None

    return angle

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
        self.distance_steps = int(np.round(self.TaskSettings['LastTravelTime'] * self.one_second_steps))
        self.angular_distance_steps = int(np.round(self.TaskSettings['MilkTaskGoalAngularDistanceTime'] * self.one_second_steps))
        self.max_distance_in_arena = int(round(np.hypot(self.TaskSettings['arena_size'][0], self.TaskSettings['arena_size'][1])))
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
        self.lastPelletReward = time()
        self.updatePelletMinSepratation()
        # Set up Milk Rewards
        self.activeMfeeders = []
        for ID in sorted(self.FEEDERs['milk'].keys(), key=int):
            if self.FEEDERs['milk'][ID]['Active']:
                self.activeMfeeders.append(ID)
        self.lastMilkTrial = time()
        self.updateMilkTrialMinSepratation()
        self.milkTrialFailTime = time() - self.TaskSettings['MilkTrialFailPenalty']
        # Initialize FEEDERs
        print('Initializing FEEDERs...')
        T_initFEEDER = []
        self.TaskSettings_Lock = threading.Lock()
        for ID in self.activePfeeders:
            T = threading.Thread(target=self.initFEEDER, args=('pellet', ID))
            T.start()
            T_initFEEDER.append(T)
        for ID in self.activeMfeeders:
            T = threading.Thread(target=self.initFEEDER, args=('milk', ID))
            T.start()
            T_initFEEDER.append(T)
        for T in T_initFEEDER:
            T.join()
        print('Initializing FEEDERs Successful')
        # Initialize counters
        self.game_counters = {'Pellets': {'ID': deepcopy(self.activePfeeders), 
                                          'count': [0] * len(self.activePfeeders)}, 
                              'Milk trials': {'ID': deepcopy(self.activeMfeeders), 
                                              'count': [0] * len(self.activeMfeeders)}, 
                              'Successful': {'ID': deepcopy(self.activeMfeeders), 
                                             'count': [0] * len(self.activeMfeeders)}}
        # Set game speed
        self.responseRate = 60 # Hz
        self.gameRate = 10 # Hz
        # Initialize game
        self.feederID_milkTrial = self.chooseMilkTrialFeeder()
        self.lastRewardLock = threading.Lock()
        self.lastReward = time()
        self.rewardInProgressLock = threading.Lock()
        self.rewardInProgress = []
        self.gameOn = False
        self.game_state = 'interval'
        self.mainLoopActive = True
        self.clock = pygame.time.Clock()
        pygame.mixer.pre_init(48000, -16, 2)  # This is necessary for sound to work
        pygame.init()
        # Check if lightSignal will be used in the task
        if self.TaskSettings['lightSignalIntensity'] > 0:
            self.lightSignalOn = True
        else:
            self.lightSignalOn = False
        # If ambient sound signals are required, create the sound
        if self.TaskSettings['AudioSignalMode'] == 'ambient':
            self.milkTrialSignal = {}
            for ID in self.activeMfeeders:
                self.milkTrialSignal[ID] = createAudioSignal(self.FEEDERs['milk'][ID]['SignalHz'], 
                                                             self.FEEDERs['milk'][ID]['SignalHzWidth'], 
                                                             self.FEEDERs['milk'][ID]['ModulHz'])

    def initFEEDER(self, FEEDER_type, ID):
        with self.TaskSettings_Lock:
            IP = self.FEEDERs[FEEDER_type][ID]['IP']
            username = self.TaskSettings['Username']
            password = self.TaskSettings['Password']
            AudioSignalMode = self.TaskSettings['AudioSignalMode']
            negativeAudioSignal = self.TaskSettings['NegativeAudioSignal']
            lightSignalIntensity = self.TaskSettings['lightSignalIntensity']
        try:
            if FEEDER_type == 'pellet' or AudioSignalMode == 'ambient':
                trialAudioSignal = None
            elif AudioSignalMode == 'localised' and FEEDER_type == 'milk':
                trialAudioSignal = (self.FEEDERs['milk'][ID]['SignalHz'], 
                                    self.FEEDERs['milk'][ID]['SignalHzWidth'], 
                                    self.FEEDERs['milk'][ID]['ModulHz'])
            actuator = RewardControl(FEEDER_type, IP, username, password, 
                                     trialAudioSignal=trialAudioSignal, 
                                     negativeAudioSignal=negativeAudioSignal, 
                                     lightSignalIntensity=lightSignalIntensity)
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
                self.ttlTimes.append(time())

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
                    self.lastReward = time()
                if 'pellet' == FEEDER_type:
                    with self.lastPelletRewardLock:
                        self.lastPelletReward = time()
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
            sleep(0.5)
            button['enabled'] = True

    def buttonManualPellet_callback(self, button):
        button['button_pressed'] = True
        # Update last reward time
        with self.lastPelletRewardLock:
            self.lastPelletReward = time()
        with self.lastRewardLock:
            self.lastReward = time()
        # Send message to Open Ephys GUI
        OEmessage = 'Reward pelletManual'
        self.TaskIO['MessageToOE'](OEmessage)
        # Keep button toggled for another 0.5 seconds
        sleep(0.5)
        button['button_pressed'] = False

    def buttonMilkTrial_callback(self, button):
        if self.game_state == 'interval' or self.game_state == 'pellet' or self.game_state == 'milk':
            self.game_state = 'transition'
            # Starts the trial with specific feeder as goal
            self.feederID_milkTrial = button['callargs'][0]
            threading.Thread(target=self.start_milkTrial, args=('user',)).start()
        else:
            button['enabled'] = False
            sleep(0.5)
            button['enabled'] = True

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
        # Button to start milkTrial
        buttonMilkTrial = []
        buttonMilkTrial.append({'text': 'Milk Trial'})
        for ID in self.activeMfeeders:
            nFeederButton = {'callback': self.buttonMilkTrial_callback, 
                             'callargs': [ID], 
                             'text': ID, 
                             'enabled': self.FEEDERs['milk'][ID]['init_successful'], 
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
                             'enabled': self.FEEDERs['milk'][ID]['init_successful'], 
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
        game_progress = self.game_logic()
        while len(game_progress) == 0:
            game_progress = self.game_logic()
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
        # Display milk feeder IDs
        self.screen.blit(self.renderText('ID'), (columnedges[2], title_topedge))
        for i, ID in enumerate(self.game_counters['Milk trials']['ID']):
            self.screen.blit(self.renderText(ID), (columnedges[2], topedge + i * textSpace))
        # Display milk trial counts
        self.screen.blit(self.renderText('milk trials'), (columnedges[3], title_topedge))
        for i, count in enumerate(self.game_counters['Milk trials']['count']):
            self.screen.blit(self.renderText(str(count)), (columnedges[3], topedge + i * textSpace))
        # Display successful milk trial counts
        self.screen.blit(self.renderText('successful'), (columnedges[4], title_topedge))
        for i, count in enumerate(self.game_counters['Successful']['count']):
            self.screen.blit(self.renderText(str(count)), (columnedges[4], topedge + i * textSpace))

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
            # Find number of successful trials for each feeder
            n_successful_trials = []
            for ID in self.activeMfeeders:
                idx = self.game_counters['Successful']['ID'].index(ID)
                n_successful_trials.append(self.game_counters['Successful']['count'][idx])
            n_successful_trials = np.array(n_successful_trials, dtype=np.float64)
            # Choose feeder with weighted randomness
            if np.any(n_successful_trials > 0):
                feederProbabilityWeights = np.sum(n_successful_trials) - n_successful_trials
                feederProbability = feederProbabilityWeights / np.sum(feederProbabilityWeights)
                n_feeder = np.random.choice(N_feeders, p=feederProbability)
            else:
                n_feeder = np.random.choice(N_feeders)
        else:
            n_feeder = 0
        ID = self.activeMfeeders[n_feeder]

        return ID

    def start_milkTrialAudioSignal(self):
        if self.TaskSettings['AudioSignalMode'] == 'ambient':
            self.milkTrialSignal[self.feederID_milkTrial].play(-1)
        elif self.TaskSettings['AudioSignalMode'] == 'localised':
            self.FEEDERs['milk'][self.feederID_milkTrial]['actuator'].startTrialAudioSignal()

    def stop_milkTrialAudioSignal(self):
        if self.TaskSettings['AudioSignalMode'] == 'ambient':
            self.milkTrialSignal[self.feederID_milkTrial].stop()
        elif self.TaskSettings['AudioSignalMode'] == 'localised':
            self.FEEDERs['milk'][self.feederID_milkTrial]['actuator'].stopTrialAudioSignal()

    def start_milkTrialLightSignal(self):
        self.FEEDERs['milk'][self.feederID_milkTrial]['actuator'].startLightSignal()

    def stop_milkTrialLightSignal(self):
        self.FEEDERs['milk'][self.feederID_milkTrial]['actuator'].stopLightSignal()

    def play_NegativeAudioSignal(self):
        self.FEEDERs['milk'][self.find_closest_feeder_ID()]['actuator'].playNegativeAudioSignal()

    def start_milkTrialSignals(self):
        self.start_milkTrialAudioSignal()
        if self.lightSignalOn:
            sleep(min([self.TaskSettings['lightSignalDelay'], self.TaskSettings['MilkTrialMaxDuration']]))
            if self.game_state == 'milk_trial':
                # This command is only given if milk trial has not yet ended.
                self.start_milkTrialLightSignal()

    def stop_milkTrialSignals(self):
        self.stop_milkTrialAudioSignal()
        if self.lightSignalOn:
            self.stop_milkTrialLightSignal()

    def start_milkTrial(self, action='undefined'):
        # These settings put the game_logic into milkTrial mode
        self.lastMilkTrial = time()
        self.game_state = 'milk_trial'
        # Make process visible on GUI
        feeder_button = self.getButton('buttonMilkTrial', self.feederID_milkTrial)
        feeder_button['button_pressed'] = True
        # Initiate signals
        threading.Thread(target=self.start_milkTrialSignals).start()
        # Send timestamp to Open Ephys GUI
        OEmessage = 'milkTrialStart ' + action + ' ' + self.feederID_milkTrial
        self.TaskIO['MessageToOE'](OEmessage)
        # Update counter
        idx = self.game_counters['Milk trials']['ID'].index(self.feederID_milkTrial)
        self.game_counters['Milk trials']['count'][idx] += 1

    def stop_milkTrial(self, successful, negative_feedback=False):
        # Release reward if successful and update counter
        if successful:
            self.game_state = 'reward_in_progress'
            threading.Thread(target=self.releaseReward, 
                             args=('milk', self.feederID_milkTrial, 'goal_milk', 
                                   self.TaskSettings['MilkQuantity'])
                             ).start()
            # Update counter
            idx = self.game_counters['Successful']['ID'].index(self.feederID_milkTrial)
            self.game_counters['Successful']['count'][idx] += 1
        else:
            self.milkTrialFailTime = time()
            self.game_state = 'interval'
            if negative_feedback and self.TaskSettings['NegativeAudioSignal'] > 0:
                self.play_NegativeAudioSignal()
        # Stop signals
        self.stop_milkTrialSignals()
        # Send timestamp to Open Ephys GUI
        OEmessage = 'milkTrialEnd Success:' + str(successful)
        self.TaskIO['MessageToOE'](OEmessage)
        # Reset GUI signal of trial process
        feeder_button = self.getButton('buttonMilkTrial', self.feederID_milkTrial)
        feeder_button['button_pressed'] = False

    def find_closest_feeder_ID(self):
        # Get animal position history
        with self.TaskIO['RPIPos'].combPosHistoryLock:
            posHistory_one_second_steps = self.TaskIO['RPIPos'].combPosHistory[-self.one_second_steps:]
        # Compute distances to all active milk feeders
        mean_posHistory = compute_mean_posHistory(posHistory_one_second_steps)
        distances = []
        for ID in self.activeMfeeders:
            distances.append(euclidean(mean_posHistory, self.FEEDERs['milk'][ID]['Position']))
        # Find identity ID of closest feeder
        ID = self.activeMfeeders[np.argmin(distances)]

        return ID

    def check_if_reward_in_progress(self):
        '''
        Returns True if any rewards are currently being delivered
        '''
        with self.rewardInProgressLock:
            n_rewards_in_progress = len(self.rewardInProgress)
        reward_in_progress = n_rewards_in_progress > 0

        return reward_in_progress

    def choose_subtask(self):
        if random.uniform(0, 1) < self.TaskSettings['PelletMilkRatio']:
            subtask = 'pellet'
        else:
            subtask = 'milk'

        return subtask

    def get_game_progress(self):
        # Get animal position history
        with self.TaskIO['RPIPos'].combPosHistoryLock:
            posHistory = self.TaskIO['RPIPos'].combPosHistory[-self.distance_steps:]
            posHistory_one_second_steps = self.TaskIO['RPIPos'].combPosHistory[-self.one_second_steps:]
            posHistory_for_angularDistance = self.TaskIO['RPIPos'].combPosHistory[-self.angular_distance_steps:]
        if not (None in posHistory):
            self.lastKnownPos = posHistory[-1]
        else:
            posHistory = [self.lastKnownPos] * self.distance_steps
        # Load variables in thread safe way
        with self.lastRewardLock:
            timeSinceLastReward = time() - self.lastReward
        with self.lastPelletRewardLock:
            lastPelletRewardTime = self.lastPelletReward
        timeSinceLastPelletReward = time() - lastPelletRewardTime
        # Compute distances to all active milk feeders
        mean_posHistory = compute_mean_posHistory(posHistory_one_second_steps)
        distances = []
        for ID in self.activeMfeeders:
            distances.append(euclidean(mean_posHistory, self.FEEDERs['milk'][ID]['Position']))
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
        # Check if milk trial penalty still applies
        game_progress_names.append('fail_penalty')
        timeSinceLastMilkFailedTrial = time() - self.milkTrialFailTime
        game_progress.append({'name': 'Fail Penalty', 
                              'game_states': ['interval'], 
                              'target': self.TaskSettings['MilkTrialFailPenalty'], 
                              'status': int(round(timeSinceLastMilkFailedTrial)), 
                              'complete': timeSinceLastMilkFailedTrial >= self.TaskSettings['MilkTrialFailPenalty'], 
                              'percentage': timeSinceLastMilkFailedTrial / float(self.TaskSettings['MilkTrialFailPenalty'])})
        # Check if enough time as passed since last milk trial
        game_progress_names.append('time_since_last_milk_trial')
        timeSinceLastMilkTrial = time() - self.lastMilkTrial
        game_progress.append({'name': 'Since Trial', 
                              'game_states': ['interval'], 
                              'target': self.TaskSettings['MilkTrialMinSeparation'], 
                              'status': int(round(timeSinceLastMilkTrial)), 
                              'complete': timeSinceLastMilkTrial >= self.TaskSettings['MilkTrialMinSeparation'], 
                              'percentage': timeSinceLastMilkTrial / float(self.TaskSettings['MilkTrialMinSeparation'])})
        # Check if has been moving enough in the last few seconds
        game_progress_names.append('mobility')
        total_distance = compute_distance_travelled(posHistory, self.TaskSettings['LastTravelSmooth'])
        game_progress.append({'name': 'Mobility', 
                              'game_states': ['pellet', 'milk'], 
                              'target': self.TaskSettings['LastTravelDist'], 
                              'status': int(round(total_distance)), 
                              'complete': total_distance >= self.TaskSettings['LastTravelDist'], 
                              'percentage': total_distance / float(self.TaskSettings['LastTravelDist'])})
        # Check if animal is far enough from milk rewards
        game_progress_names.append('distance_from_milk_feeders')
        minDistance = min(distances)
        game_progress.append({'name': 'Milk Distance', 
                              'game_states': ['milk'], 
                              'target': self.TaskSettings['MilkTaskMinStartDistance'], 
                              'status': int(round(minDistance)), 
                              'complete': minDistance >= self.TaskSettings['MilkTaskMinStartDistance'], 
                              'percentage': minDistance / float(self.TaskSettings['MilkTaskMinStartDistance'])})
        # Check if animal not moving towards goal location
        game_progress_names.append('angular_distance_from_goal_feeder')
        target_location = self.FEEDERs['milk'][self.feederID_milkTrial]['Position']
        angularDistance = compute_movement_angular_distance_to_target(posHistory_for_angularDistance, target_location)
        if angularDistance is None:
            angularDistance = 0
        game_progress.append({'name': 'Milk A.Distance', 
                              'game_states': ['milk'], 
                              'target': self.TaskSettings['MilkTaskMinGoalAngularDistance'], 
                              'status': int(round(angularDistance)), 
                              'complete': angularDistance >= self.TaskSettings['MilkTaskMinGoalAngularDistance'], 
                              'percentage': angularDistance / float(self.TaskSettings['MilkTaskMinGoalAngularDistance'])})
        # Check if animal is close enough to goal location
        game_progress_names.append('distance_from_goal_feeder')
        goal_distance = distances[self.activeMfeeders.index(self.feederID_milkTrial)]
        game_progress.append({'name': 'Goal Distance', 
                              'game_states': ['milk_trial'], 
                              'target': self.TaskSettings['MilkTaskMinGoalDistance'], 
                              'status': int(round(goal_distance)), 
                              'complete': goal_distance <= self.TaskSettings['MilkTaskMinGoalDistance'], 
                              'percentage': 1 - (goal_distance - self.TaskSettings['MilkTaskMinGoalDistance']) / float(self.max_distance_in_arena)})
        # Check if animal is too close to goal incorrect location
        game_progress_names.append('distance_from_other_feeders')
        other_distances = min([distances[i] for i in range(len(self.activeMfeeders)) if self.activeMfeeders[i] != self.feederID_milkTrial])
        game_progress.append({'name': 'Other Distance', 
                              'game_states': ['milk_trial'], 
                              'target': self.TaskSettings['MilkTaskMinGoalDistance'], 
                              'status': int(round(other_distances)), 
                              'complete': other_distances <= self.TaskSettings['MilkTaskMinGoalDistance'], 
                              'percentage': 1 - (other_distances - self.TaskSettings['MilkTaskMinGoalDistance']) / float(self.max_distance_in_arena)})
        # Check if trial has been running for too long
        game_progress_names.append('milk_trial_duration')
        trial_run_time = time() - self.lastMilkTrial
        game_progress.append({'name': 'Trial Duration', 
                              'game_states': ['milk_trial'], 
                              'target': self.TaskSettings['MilkTrialMaxDuration'], 
                              'status': int(round(trial_run_time)), 
                              'complete': trial_run_time > self.TaskSettings['MilkTrialMaxDuration'], 
                              'percentage': trial_run_time / float(self.TaskSettings['MilkTrialMaxDuration'])})

        return game_progress, game_progress_names

    def game_logic(self):
        game_progress, game_progress_names = self.get_game_progress()
        # IF IN INTERVAL STATE
        if self.game_state == 'interval':
            # If in interval state, check if conditions met for milk or pellet state transition
            conditions = {'inactivity': game_progress[game_progress_names.index('inactivity')]['complete'], 
                          'chewing': game_progress[game_progress_names.index('chewing')]['complete'], 
                          'pellet_interval': game_progress[game_progress_names.index('time_since_last_pellet')]['complete'], 
                          'milk_trial_penalty': game_progress[game_progress_names.index('fail_penalty')]['complete'], 
                          'milk_trial_interval': game_progress[game_progress_names.index('time_since_last_milk_trial')]['complete']}
            if conditions['inactivity']:
                # If animal has been without any rewards for too long, release pellet reward
                self.game_state = 'reward_in_progress'
                ID = self.choose_pellet_feeder()
                threading.Thread(target=self.releaseReward, 
                                 args=('pellet', ID, 'goal_inactivity', self.TaskSettings['PelletQuantity'])
                                 ).start()
            elif conditions['chewing'] and conditions['pellet_interval'] and conditions['milk_trial_penalty'] and conditions['milk_trial_interval']:
                # If conditions for pellet and milk state are met, choose one based on choose_subtask function
                self.game_state = self.choose_subtask()
            elif conditions['chewing'] and conditions['pellet_interval'] and conditions['milk_trial_penalty'] and (not conditions['milk_trial_interval']):
                # If conditions are met for pellet but not for milk, change game state to pellet
                self.game_state = 'pellet'
            elif (not conditions['chewing']) and conditions['pellet_interval'] and conditions['milk_trial_penalty'] and conditions['milk_trial_interval']:
                # If conditions are met for milk but not for pellet, change game state to milk
                self.game_state = 'milk'
            # Make sure there are feeders for this game state
            if self.game_state == 'pellet' and len(self.activePfeeders) == 0:
                self.game_state = 'interval'
            if self.game_state == 'milk' and len(self.activeMfeeders) == 0:
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
        # IF IN MILK STATE
        elif self.game_state == 'milk':
            conditions = {'inactivity': game_progress[game_progress_names.index('inactivity')]['complete'], 
                          'mobility': game_progress[game_progress_names.index('mobility')]['complete'], 
                          'distance_from_milk_feeders': game_progress[game_progress_names.index('distance_from_milk_feeders')]['complete'], 
                          'angular_distance_from_goal_feeder': game_progress[game_progress_names.index('angular_distance_from_goal_feeder')]['complete']}
            if conditions['inactivity']:
                # If animal has been without any rewards for too long, release pellet reward
                self.game_state = 'reward_in_progress'
                ID = self.choose_pellet_feeder()
                threading.Thread(target=self.releaseReward, 
                                 args=('pellet', ID, 'goal_inactivity', self.TaskSettings['PelletQuantity'])
                                 ).start()
            elif conditions['distance_from_milk_feeders'] and conditions['mobility'] and conditions['angular_distance_from_goal_feeder']:
                # If animal is far enough from milk feeders and is mobile enough, start milk trial
                self.game_state = 'transition'
                threading.Thread(target=self.start_milkTrial, args=('goal_milkTrialStart',)).start() # changes self.game_state = 'milk_trial'
        # IF IN MILK_TRIAL STATE
        elif self.game_state == 'milk_trial':
            conditions = {'distance_from_goal_feeder': game_progress[game_progress_names.index('distance_from_goal_feeder')]['complete'], 
                          'distance_from_other_feeders': game_progress[game_progress_names.index('distance_from_other_feeders')]['complete'], 
                          'milk_trial_duration': game_progress[game_progress_names.index('milk_trial_duration')]['complete']}
            if conditions['distance_from_goal_feeder']:
                # If subject reached goal location, stop milk trial with positive outcome
                self.game_state = 'transition'
                threading.Thread(target=self.stop_milkTrial, args=(True,)).start() # changes self.game_state = 'reward_in_progress'
            elif conditions['milk_trial_duration']:
                # If time limit for task duration has passed, stop milk trial with negative outcome
                self.game_state = 'transition'
                threading.Thread(target=self.stop_milkTrial, args=(False,)).start() # changes self.game_state = 'interval'
            elif conditions['distance_from_other_feeders']:
                # If subject went to incorrect location, stop milk trial with negative outcome and feedback
                self.game_state = 'transition'
                threading.Thread(target=self.stop_milkTrial, args=(False, True)).start() # changes self.game_state = 'interval'
        # IF IN REWARD_IN_PROGRESS STATE
        elif self.game_state == 'reward_in_progress':
            reward_in_progress = self.check_if_reward_in_progress()
            if not reward_in_progress:
                # If reward is not in progress anymore, change game state to interval
                self.game_state = 'interval'
                # Set new variable timer limits
                self.updatePelletMinSepratation()
                self.updateMilkTrialMinSepratation()
                # Update next milk_trial feeder
                self.feederID_milkTrial = self.chooseMilkTrialFeeder()
                # Recompute game state to with new parameters
                game_progress, game_progress_names = self.get_game_progress()

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
            sleep(0.1)
            with self.TaskIO['RPIPos'].combPosHistoryLock:
                posHistory = self.TaskIO['RPIPos'].combPosHistory
        self.lastKnownPos = posHistory[-1]
        # Initialize GUI elements
        self.screen_Lock = threading.Lock()
        self.screen_size = (1400, 300)
        self.screen_margins = 20
        self.screen_progress_bar_space_start = 0
        self.screen_progress_bar_space_width = 800
        self.screen_button_space_start = 800
        self.screen_button_space_width = 300
        self.screen_text_info_space_start = 1100
        self.screen_text_info_space_width = 300
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
            if lastUpdatedState < (time() - 1 / float(self.gameRate)):
                if T_update_states.is_alive():
                    print('Game State Processing thread pileup! ' + asctime())
                T_update_states.join() # Ensure the previous state update thread has finished
                T_update_states = threading.Thread(target=self.update_states)
                T_update_states.start()
                lastUpdatedState = time()
        # Quit game when out of gameOn loop
        with self.screen_Lock:
            pygame.mixer.quit()
            pygame.quit()

    def run(self):
        self.init_main_loop()
        threading.Thread(target=self.main_loop).start()

    def stop(self):
        # Make sure reward delivery is finished before closing game processes
        while self.game_state == 'reward_in_progress':
            print('Waiting for reward_in_progress to finish...')
            sleep(1)
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
