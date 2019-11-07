# This is a task

import numpy as np
from threading import Lock, Thread
from openEPhys_DACQ.RPiInterface import RewardControl
from time import time, sleep
from scipy.spatial.distance import euclidean
import random
from PyQt5 import QtWidgets, QtGui
import warnings
from copy import copy, deepcopy

from openEPhys_DACQ.audioSignalGenerator import createAudioSignal
from openEPhys_DACQ.sshScripts import ssh
from openEPhys_DACQ.HelperFunctions import show_message

import contextlib
with contextlib.redirect_stdout(None):
    import pygame


def init_pygame():
    pygame.mixer.pre_init(48000, -16, 2)  # This is necessary for sound to work
    pygame.init()


def close_pygame():
    pygame.mixer.quit()
    pygame.quit()


def activate_feeder(feeder_type, RPiIPBox, RPiUsernameBox, RPiPasswordBox, quantityBox):
    ssh_connection = ssh(str(RPiIPBox.text()), str(RPiUsernameBox.text()), str(RPiPasswordBox.text()))
    if feeder_type == 'milk':
        command = 'python milkFeederController.py --openValve ' + str(float(str(quantityBox.text())))
    elif feeder_type == 'pellet':
        command = 'python pelletFeederController.py --releasePellet ' + str(int(str(quantityBox.text())))
    else:
        raise Exception('Unknown feeder_type {}'.format(feeder_type))
    ssh_connection.sendCommand(command)
    ssh_connection.disconnect()


def set_double_h_box_stretch(hbox):
    hbox.setStretch(0, 2)
    hbox.setStretch(1, 1)

    return hbox


def set_triple_h_box_stretch(hbox):
    hbox.setStretch(0, 3)
    hbox.setStretch(1, 1)
    hbox.setStretch(2, 1)

    return hbox


def set_quadruple_h_box_stretch(hbox):
    hbox.setStretch(0, 2)
    hbox.setStretch(1, 1)
    hbox.setStretch(2, 1)
    hbox.setStretch(3, 1)

    return hbox


def play_audio_signal(frequency, frequency_band_width, modulation_frequency, duration=2):
    if type(frequency) == QtWidgets.QLineEdit:
        frequency = np.int64(float(str(frequency.text())))
    if type(frequency_band_width) == QtWidgets.QLineEdit:
        frequency_band_width = np.int64(float(str(frequency_band_width.text())))
    if type(modulation_frequency) == QtWidgets.QLineEdit:
        modulation_frequency = np.int64(float(str(modulation_frequency.text())))
    # Initialize pygame for playing sound
    init_pygame()
    # Get sound
    sound = createAudioSignal(frequency, frequency_band_width, modulation_frequency)
    # Play duration seconds of the sound
    sound.play(-1, maxtime=(duration * 1000))
    sleep(duration)
    close_pygame()


def distance_from_segment(point, seg_p1, seg_p2):
    """
    Computes distance of numpy array point from a segment defined by two numpy array points
    seg_p1 and seg_p2.
    """
    return np.cross(seg_p2 - seg_p1, point - seg_p1) / np.linalg.norm(seg_p2 - seg_p1)


def distance_from_boundaries(point, arena_size):
    # North wall
    seg_1_p1 = np.array([0, 0]).astype(np.float64)
    seg_1_p2 = np.array([arena_size[0], 0]).astype(np.float64)
    # East wall
    seg_2_p1 = np.array([arena_size[0], 0]).astype(np.float64)
    seg_2_p2 = np.array([arena_size[0], arena_size[1]]).astype(np.float64)
    # South wall
    seg_3_p1 = np.array([arena_size[0], arena_size[1]]).astype(np.float64)
    seg_3_p2 = np.array([0, arena_size[1]]).astype(np.float64)
    # West wall
    seg_4_p1 = np.array([0, arena_size[1]]).astype(np.float64)
    seg_4_p2 = np.array([0, 0]).astype(np.float64)
    # List of walls
    segments = [[seg_1_p1, seg_1_p2], [seg_2_p1, seg_2_p2], 
                [seg_3_p1, seg_3_p2], [seg_4_p1, seg_4_p2]]
    # Find minimum distance from all walls
    distances = []
    for seg_p1, seg_p2 in segments:
        distances.append(distance_from_segment(point, seg_p1, seg_p2))
    distance = min(distances)

    return distance


class SettingsGUI(object):

    def __init__(self, main_settings_layout, further_settings_layout, arena_size):
        """
        main_settings_layout    - QtWidgets VBox Layout
        further_settings_layout - QtWidgets HBox Layout
        arena_size              - list or numpy array of x and y size of the arena
        """

        # Create empty settings variables
        self.arena_size = arena_size
        self.settings = {'FEEDERs': {'pellet': [], 'milk': []}}
        # Create GUI size requirements
        self.min_size = [0, 0]
        # Create empty button groups dictionary
        self.button_groups = {}
        # Create settings menu
        self.populate_main_settings_layout(main_settings_layout)
        self.populate_further_settings_layout(further_settings_layout)

    def make_space_for_frame(self, frame):
        self.min_size[0] = self.min_size[0] + frame.minimumWidth()
        self.min_size[1] = max(self.min_size[1], frame.minimumHeight())
 
    def populate_main_settings_layout(self, main_settings_layout):
        vbox = QtWidgets.QVBoxLayout()
        font = QtGui.QFont('SansSerif', 15)
        string = QtWidgets.QLabel('General Settings')
        string.setFont(font)
        string.setMaximumHeight(40)
        vbox.addWidget(string)
        # Specify which game is active Pellet, Milk or both
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Games active'))
        self.settings['games_active'] = {'pellet': QtWidgets.QCheckBox('Pellet'),
                                         'milk': QtWidgets.QCheckBox('Milk')}
        self.settings['games_active']['pellet'].setChecked(True)
        self.settings['games_active']['milk'].setChecked(True)
        hbox.addWidget(self.settings['games_active']['pellet'])
        hbox.addWidget(self.settings['games_active']['milk'])
        vbox.addLayout(set_triple_h_box_stretch(hbox))
        # Add option to specify how far into past to check travel distance
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Last travel time (s)'))
        self.settings['LastTravelTime'] = QtWidgets.QLineEdit('2')
        hbox.addWidget(self.settings['LastTravelTime'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        # Add smoothing factor for calculating last travel distance
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Last travel smoothing (dp)'))
        self.settings['LastTravelSmooth'] = QtWidgets.QLineEdit('3')
        hbox.addWidget(self.settings['LastTravelSmooth'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        # Add minimum distance for last travel
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Last travel min distance (cm)'))
        self.settings['LastTravelDist'] = QtWidgets.QLineEdit('50')
        hbox.addWidget(self.settings['LastTravelDist'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        # Specify pellet vs milk reward ratio
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Pellet vs Milk Reward ratio'))
        self.settings['PelletMilkRatio'] = QtWidgets.QLineEdit('0.25')
        hbox.addWidget(self.settings['PelletMilkRatio'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        # Specify raspberry pi username
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Raspberry Pi usernames'))
        self.settings['Username'] = QtWidgets.QLineEdit('pi')
        hbox.addWidget(self.settings['Username'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        # Specify raspberry pi password
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Raspberry Pi passwords'))
        self.settings['Password'] = QtWidgets.QLineEdit('raspberry')
        hbox.addWidget(self.settings['Password'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        # Put these settings into a frame
        frame = QtWidgets.QFrame()
        frame.setLayout(vbox)
        frame.setFrameStyle(3)
        # Set minimum size for frame
        frame.setFixedSize(300, 300)
        # Put frame into main settings layout
        main_settings_layout.addWidget(frame)
        self.make_space_for_frame(frame)

    def populate_further_settings_layout(self, further_settings_layout):
        pellet_settings_frame = self.create_pellet_task_settings()
        further_settings_layout.addWidget(pellet_settings_frame)
        self.make_space_for_frame(pellet_settings_frame)
        milk_settings_frame = self.create_milk_task_settings()
        further_settings_layout.addWidget(milk_settings_frame)
        self.make_space_for_frame(milk_settings_frame)

    def create_pellet_task_settings(self):
        # Create Pellet task specific menu items
        vbox = QtWidgets.QVBoxLayout()
        font = QtGui.QFont('SansSerif', 15)
        string = QtWidgets.QLabel('Pellet Game Settings')
        string.setFont(font)
        vbox.addWidget(string)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Initial Pellets'))
        self.settings['InitPellets'] = QtWidgets.QLineEdit('5')
        hbox.addWidget(self.settings['InitPellets'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Reward Quantity'))
        self.settings['PelletQuantity'] = QtWidgets.QLineEdit('1')
        hbox.addWidget(self.settings['PelletQuantity'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Min Separation (s)'))
        self.settings['PelletRewardMinSeparationMean'] = QtWidgets.QLineEdit('10')
        hbox.addWidget(self.settings['PelletRewardMinSeparationMean'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Min Separation variance (%)'))
        self.settings['PelletRewardMinSeparationVariance'] = QtWidgets.QLineEdit('0.5')
        hbox.addWidget(self.settings['PelletRewardMinSeparationVariance'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Chewing Target count'))
        self.settings['Chewing_Target'] = QtWidgets.QLineEdit('4')
        hbox.addWidget(self.settings['Chewing_Target'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Inactivity pellet time (s)'))
        self.settings['MaxInactivityDuration'] = QtWidgets.QLineEdit('90')
        hbox.addWidget(self.settings['MaxInactivityDuration'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        # Specify chewing signal TTL channel
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Chewing TTL channel'))
        self.settings['Chewing_TTLchan'] = QtWidgets.QLineEdit('5')
        hbox.addWidget(self.settings['Chewing_TTLchan'])
        vbox.addLayout(set_double_h_box_stretch(hbox))
        # Create Pellet FEEDER items
        scroll_widget = QtWidgets.QWidget()
        self.pellet_feeder_settings_layout = QtWidgets.QVBoxLayout(scroll_widget)
        self.addPelletFeederButton = QtWidgets.QPushButton('Add FEEDER')
        self.addPelletFeederButton.clicked.connect(lambda: self.addFeedersToList('pellet'))
        self.pellet_feeder_settings_layout.addWidget(self.addPelletFeederButton)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        vbox.addWidget(scroll)
        # Add Pellet Task settings to task specific settings layout
        frame = QtWidgets.QFrame()
        frame.setLayout(vbox)
        frame.setFrameStyle(3)
        # Set minimum size for frame
        frame.setMinimumSize(400, 900)

        return frame

    def create_milk_task_settings(self):
        vbox = QtWidgets.QVBoxLayout()
        # Create Milk task label
        font = QtGui.QFont('SansSerif', 15)
        string = QtWidgets.QLabel('Milk Game Settings')
        string.setFont(font)
        vbox.addWidget(string)
        # Create top grid layout
        grid = QtWidgets.QGridLayout()
        vbox.addLayout(grid)
        # Add initiation milk amount
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Initial Milk'))
        self.settings['InitMilk'] = QtWidgets.QLineEdit('2')
        hbox.addWidget(self.settings['InitMilk'])
        grid.addLayout(set_double_h_box_stretch(hbox), 0, 0)
        # Specify audio signal mode
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Audio Signal Mode'))
        self.settings['AudioSignalMode'] = {'ambient': QtWidgets.QRadioButton('Ambient'),
                                            'localised': QtWidgets.QRadioButton('Localised')}
        self.settings['AudioSignalMode']['ambient'].setChecked(True)
        hbox.addWidget(self.settings['AudioSignalMode']['ambient'])
        hbox.addWidget(self.settings['AudioSignalMode']['localised'])
        self.button_groups['AudioSignalMode'] = QtWidgets.QButtonGroup()
        self.button_groups['AudioSignalMode'].addButton(self.settings['AudioSignalMode']['ambient'])
        self.button_groups['AudioSignalMode'].addButton(self.settings['AudioSignalMode']['localised'])
        grid.addLayout(set_triple_h_box_stretch(hbox), 1, 0)
        # Add Milk reward quantity
        self.settings['MilkQuantity'] = {}
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Reward Quantity'))
        sub_hbox = QtWidgets.QHBoxLayout()
        sub_hbox.addWidget(QtWidgets.QLabel('present'))
        self.settings['MilkQuantity']['presentation'] = QtWidgets.QLineEdit('1')
        sub_hbox.addWidget(self.settings['MilkQuantity']['presentation'])
        hbox.addLayout(set_double_h_box_stretch(sub_hbox))
        sub_hbox = QtWidgets.QHBoxLayout()
        sub_hbox.addWidget(QtWidgets.QLabel('repeat'))
        self.settings['MilkQuantity']['repeat'] = QtWidgets.QLineEdit('1')
        sub_hbox.addWidget(self.settings['MilkQuantity']['repeat'])
        hbox.addLayout(set_double_h_box_stretch(sub_hbox))
        grid.addLayout(set_triple_h_box_stretch(hbox), 2, 0)
        # Specify light signal pins to use, separated by comma
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Light Signal Pin(s)'))
        self.settings['lightSignalPins'] = QtWidgets.QLineEdit('1')
        hbox.addWidget(self.settings['lightSignalPins'])
        grid.addLayout(set_double_h_box_stretch(hbox), 3, 0)
        # Specify light signal settings regarding repeating trials
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Light Signal On'))
        self.settings['LightSignalOnRepetitions'] = {'presentation': QtWidgets.QCheckBox('present'),
                                                     'repeat': QtWidgets.QCheckBox('repeat')}
        self.settings['LightSignalOnRepetitions']['presentation'].setChecked(True)
        hbox.addWidget(self.settings['LightSignalOnRepetitions']['presentation'])
        hbox.addWidget(self.settings['LightSignalOnRepetitions']['repeat'])
        grid.addLayout(set_triple_h_box_stretch(hbox), 4, 0)
        # Specify probability that light signal does turn on
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Light Signal probability (0 - 1)'))
        self.settings['lightSignalProbability'] = QtWidgets.QLineEdit('1')
        hbox.addWidget(self.settings['lightSignalProbability'])
        grid.addLayout(set_double_h_box_stretch(hbox), 5, 0)
        # Specify light signal intensity
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Light Signal intensity (0 - 100)'))
        self.settings['lightSignalIntensity'] = QtWidgets.QLineEdit('100')
        hbox.addWidget(self.settings['lightSignalIntensity'])
        grid.addLayout(set_double_h_box_stretch(hbox), 6, 0)
        # Specify light signal delay relative to trial start
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Light Signal delay (s)'))
        self.settings['lightSignalDelay'] = QtWidgets.QLineEdit('0')
        hbox.addWidget(self.settings['lightSignalDelay'])
        grid.addLayout(set_double_h_box_stretch(hbox), 7, 0)
        # Option to set duration of negative audio feedback
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Negative Audio Feedback (s)'))
        self.settings['NegativeAudioSignal'] = QtWidgets.QLineEdit('0')
        hbox.addWidget(self.settings['NegativeAudioSignal'])
        grid.addLayout(set_double_h_box_stretch(hbox), 8, 0)
        # Specify milk trial fail penalty duration
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Milk Trial Fail Penalty (s)'))
        self.settings['MilkTrialFailPenalty'] = QtWidgets.QLineEdit('10')
        hbox.addWidget(self.settings['MilkTrialFailPenalty'])
        grid.addLayout(set_double_h_box_stretch(hbox), 9, 0)
        # Specify milk trial mean separation
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Min Separation (s)'))
        self.settings['MilkTrialMinSeparationMean'] = QtWidgets.QLineEdit('40')
        hbox.addWidget(self.settings['MilkTrialMinSeparationMean'])
        grid.addLayout(set_double_h_box_stretch(hbox), 0, 1)
        # Specify milk trial separation variance
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Min Separation variance (%)'))
        self.settings['MilkTrialMinSeparationVariance'] = QtWidgets.QLineEdit('0.5')
        hbox.addWidget(self.settings['MilkTrialMinSeparationVariance'])
        grid.addLayout(set_double_h_box_stretch(hbox), 1, 1)
        # Specify minimum distance to feeder for starting a trial
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Minimum Start Distance (cm)'))
        self.settings['MilkTaskMinStartDistance'] = QtWidgets.QLineEdit('50')
        hbox.addWidget(self.settings['MilkTaskMinStartDistance'])
        grid.addLayout(set_double_h_box_stretch(hbox), 2, 1)
        # Specify minimum angular distance to goal for starting a trial
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Minimum Goal angular distance (deg)'))
        self.settings['MilkTaskMinGoalAngularDistance'] = QtWidgets.QLineEdit('45')
        hbox.addWidget(self.settings['MilkTaskMinGoalAngularDistance'])
        grid.addLayout(set_double_h_box_stretch(hbox), 3, 1)
        # Specify position history period for computing goal angular distance
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Goal angular distance time (s)'))
        self.settings['MilkTaskGoalAngularDistanceTime'] = QtWidgets.QLineEdit('2')
        hbox.addWidget(self.settings['MilkTaskGoalAngularDistanceTime'])
        grid.addLayout(set_double_h_box_stretch(hbox), 4, 1)
        # Specify minimum distance to goal feeder for ending the trial
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Minimum Goal Distance (cm)'))
        self.settings['MilkTaskMinGoalDistance'] = QtWidgets.QLineEdit('10')
        hbox.addWidget(self.settings['MilkTaskMinGoalDistance'])
        grid.addLayout(set_double_h_box_stretch(hbox), 5, 1)
        # Specify maximum trial duration
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Maximum Trial Duration (s)'))
        self.settings['MilkTrialMaxDuration'] = QtWidgets.QLineEdit('9')
        hbox.addWidget(self.settings['MilkTrialMaxDuration'])
        grid.addLayout(set_double_h_box_stretch(hbox), 6, 1)
        # Specify method of choosing the next milk feeder
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Next Goal'))
        self.settings['MilkGoalNextFeederMethod'] = {'random': QtWidgets.QRadioButton('Random'),
                                                     'weighted': QtWidgets.QRadioButton('Weighted'),
                                                     'cycle': QtWidgets.QRadioButton('Cycle')}
        self.settings['MilkGoalNextFeederMethod']['cycle'].setChecked(True)
        hbox.addWidget(self.settings['MilkGoalNextFeederMethod']['random'])
        hbox.addWidget(self.settings['MilkGoalNextFeederMethod']['weighted'])
        hbox.addWidget(self.settings['MilkGoalNextFeederMethod']['cycle'])
        self.button_groups['MilkGoalNextFeederMethod'] = QtWidgets.QButtonGroup()
        self.button_groups['MilkGoalNextFeederMethod'].addButton(
            self.settings['MilkGoalNextFeederMethod']['random'])
        self.button_groups['MilkGoalNextFeederMethod'].addButton(
            self.settings['MilkGoalNextFeederMethod']['weighted'])
        self.button_groups['MilkGoalNextFeederMethod'].addButton(
            self.settings['MilkGoalNextFeederMethod']['cycle'])
        grid.addLayout(set_quadruple_h_box_stretch(hbox), 7, 1)
        # Specify number of repretitions of each milk trial goal
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('Milk goal repetitions'))
        self.settings['MilkGoalRepetition'] = QtWidgets.QLineEdit('0')
        hbox.addWidget(self.settings['MilkGoalRepetition'])
        grid.addLayout(set_double_h_box_stretch(hbox), 8, 1)
        # Create Milk FEEDER items
        scroll_widget = QtWidgets.QWidget()
        self.milk_feeder_settings_layout = QtWidgets.QVBoxLayout(scroll_widget)
        self.addMilkFeederButton = QtWidgets.QPushButton('Add FEEDER')
        self.addMilkFeederButton.clicked.connect(lambda: self.addFeedersToList('milk'))
        self.milk_feeder_settings_layout.addWidget(self.addMilkFeederButton)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        vbox.addWidget(scroll)
        # Add Milk Task settings to task specific settings layout
        frame = QtWidgets.QFrame()
        frame.setLayout(vbox)
        frame.setFrameStyle(3)
        # Set minimum size for frame
        frame.setMinimumSize(800, 800)

        return frame

    def autoFeederPosition(self, target_feeder, max_attempts=1000):
        target_feeder_spacing = np.int64(float(target_feeder['Spacing'].text()))
        target_feeder_clearence = np.int64(float(target_feeder['Clearence'].text()))
        # Collect positions and spacing settings of all other feeders
        positions = []
        spacings = []
        for FEEDER in self.settings['FEEDERs']['milk']:
            if not (target_feeder is FEEDER):
                other_pos = np.array(map(float, str(FEEDER['Position'].text()).split(','))).astype(np.float64)
                positions.append(other_pos)
                spacings.append(float(str(FEEDER['Spacing'].text())))
        # Keep looking for new position until one matches criteria
        n_attempt = 0
        position_found = False
        while not position_found and n_attempt < max_attempts:
            n_attempt += 1
            # Pick new position randomly from uniform distribution across the environment
            position = np.array([random.random() * self.arena_size[0], 
                                 random.random() * self.arena_size[1]], dtype=np.int64)
            position_found = True
            # Check if it is too close to the boundaries
            if position_found:
                if distance_from_boundaries(position, self.arena_size) < target_feeder_clearence:
                    position_found = False
            # Check if position is too close to any other feeder
            if position_found:
                for other_pos, other_spacing in zip(positions, spacings):
                    distance = euclidean(position, other_pos)
                    if distance < target_feeder_spacing or distance < other_spacing:
                        position_found = False
        if position_found:
            # Set position to correct format
            position = ','.join(map(str, map(int, list(position))))
            # Pick random orientation
            angle = str(int(round(random.random() * 360.0)))
            # Set position value in the target feeder box
            target_feeder['Position'].setText(position)
            # Set angle value in the target feeder box
            target_feeder['Angle'].setText(angle)
        else:
            show_message('Could not find a position matching the criteria.')

    def addFeedersToList(self, feeder_type, FEEDER_settings=None):
        if FEEDER_settings is None:
            FEEDER_settings = {'ID': '1', 
                               'Present': True, 
                               'Active': True, 
                               'IP': '192.168.0.40', 
                               'Position': np.array([100,50]), 
                               'Angle': np.array(0), 
                               'Spacing': np.array(60), 
                               'Clearence': np.array(20), 
                               'SignalHz': np.array(10000), 
                               'SignalHzWidth': np.array(500), 
                               'ModulHz': np.array(4)}
        # Create interface for interacting with this FEEDER
        FEEDER = {'Type': feeder_type}
        vbox = QtWidgets.QVBoxLayout()
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(QtWidgets.QLabel('ID:'))
        FEEDER['ID'] = QtWidgets.QLineEdit(FEEDER_settings['ID'])
        FEEDER['ID'].setMaximumWidth(40)
        hbox.addWidget(FEEDER['ID'])
        hbox.addWidget(QtWidgets.QLabel('IP:'))
        FEEDER['IP'] = QtWidgets.QLineEdit(FEEDER_settings['IP'])
        FEEDER['IP'].setMinimumWidth(105)
        hbox.addWidget(FEEDER['IP'])
        activateButton = QtWidgets.QPushButton('Activate')
        activateButton.setMinimumWidth(70)
        activateButton.setMaximumWidth(70)
        FEEDER['ReleaseQuantity'] = QtWidgets.QLineEdit('1')
        FEEDER['ReleaseQuantity'].setMaximumWidth(40)
        activateButton.clicked.connect(lambda: activate_feeder(feeder_type, FEEDER['IP'], 
                                                              self.settings['Username'], 
                                                              self.settings['Password'], 
                                                              FEEDER['ReleaseQuantity']))
        hbox.addWidget(activateButton)
        hbox.addWidget(FEEDER['ReleaseQuantity'])
        vbox.addLayout(hbox)
        hbox = QtWidgets.QHBoxLayout()
        FEEDER['Present'] = QtWidgets.QCheckBox('Present')
        FEEDER['Present'].setChecked(FEEDER_settings['Present'])
        hbox.addWidget(FEEDER['Present'])
        FEEDER['Active'] = QtWidgets.QCheckBox('Active')
        FEEDER['Active'].setChecked(FEEDER_settings['Active'])
        hbox.addWidget(FEEDER['Active'])
        hbox.addWidget(QtWidgets.QLabel('Position:'))
        FEEDER['Position'] = QtWidgets.QLineEdit(','.join(map(str, FEEDER_settings['Position'])))
        FEEDER['Position'].setMinimumWidth(70)
        FEEDER['Position'].setMaximumWidth(70)
        hbox.addWidget(FEEDER['Position'])
        vbox.addLayout(hbox)
        if feeder_type == 'milk':
            hbox = QtWidgets.QHBoxLayout()
            # Add minimum spacing betwen feeders
            hbox.addWidget(QtWidgets.QLabel('Spacing:'))
            FEEDER['Spacing'] = QtWidgets.QLineEdit(str(FEEDER_settings['Spacing']))
            FEEDER['Spacing'].setMinimumWidth(40)
            FEEDER['Spacing'].setMaximumWidth(40)
            hbox.addWidget(FEEDER['Spacing'])
            # Add minimum clearence from boundaries
            hbox.addWidget(QtWidgets.QLabel('Clearence:'))
            FEEDER['Clearence'] = QtWidgets.QLineEdit(str(FEEDER_settings['Clearence']))
            FEEDER['Clearence'].setMinimumWidth(40)
            FEEDER['Clearence'].setMaximumWidth(40)
            hbox.addWidget(FEEDER['Clearence'])
            # Add angular position to specify feeder orientation
            hbox.addWidget(QtWidgets.QLabel('Angle:'))
            FEEDER['Angle'] = QtWidgets.QLineEdit(str(FEEDER_settings['Angle']))
            FEEDER['Angle'].setMinimumWidth(60)
            FEEDER['Angle'].setMaximumWidth(60)
            hbox.addWidget(FEEDER['Angle'])
            # Add a button to automatically select feeder orientation and angle
            autoPosButton = QtWidgets.QPushButton('AutoPos')
            autoPosButton.setMinimumWidth(70)
            autoPosButton.setMaximumWidth(70)
            autoPosButton.clicked.connect(lambda: self.autoFeederPosition(FEEDER))
            hbox.addWidget(autoPosButton)
            # Finish this row of options
            vbox.addLayout(hbox)
            # Add sound signal values
            hbox = QtWidgets.QHBoxLayout()
            hbox.addWidget(QtWidgets.QLabel('Signal (Hz):'))
            FEEDER['SignalHz'] = QtWidgets.QLineEdit(str(FEEDER_settings['SignalHz']))
            hbox.addWidget(FEEDER['SignalHz'])
            hbox.addWidget(QtWidgets.QLabel('W:'))
            if not 'SignalHzWidth' in FEEDER_settings.keys():
                print('Remove this section in Pellets_and_Milk_Task.py when settings resaved!')
                FEEDER_settings['SignalHzWidth'] = np.array(500)
            FEEDER['SignalHzWidth'] = QtWidgets.QLineEdit(str(FEEDER_settings['SignalHzWidth']))
            hbox.addWidget(FEEDER['SignalHzWidth'])
            hbox.addWidget(QtWidgets.QLabel('M:'))
            FEEDER['ModulHz'] = QtWidgets.QLineEdit(str(FEEDER_settings['ModulHz']))
            hbox.addWidget(FEEDER['ModulHz'])
            playSignalButton = QtWidgets.QPushButton('Play')
            playSignalButton.setMaximumWidth(40)
            playSignalButton.clicked.connect(lambda: play_audio_signal(FEEDER['SignalHz'], 
                                                                 FEEDER['SignalHzWidth'], 
                                                                 FEEDER['ModulHz']))
            hbox.addWidget(playSignalButton)
            vbox.addLayout(hbox)
        frame = QtWidgets.QFrame()
        frame.setLayout(vbox)
        frame.setFrameStyle(3)
        if feeder_type == 'milk':
            frame.setMaximumHeight(160)
        else:
            frame.setMaximumHeight(90)
        if feeder_type == 'pellet':
            self.pellet_feeder_settings_layout.addWidget(frame)
        elif feeder_type == 'milk':
            self.milk_feeder_settings_layout.addWidget(frame)
        self.settings['FEEDERs'][feeder_type].append(FEEDER)

    def export_settings_from_gui(self):
        # Get task settings from text boxes
        TaskSettings = {'LastTravelTime': np.float64(str(self.settings['LastTravelTime'].text())),
                        'LastTravelSmooth': np.int64(float(str(self.settings['LastTravelSmooth'].text()))),
                        'LastTravelDist': np.int64(float(str(self.settings['LastTravelDist'].text()))),
                        'PelletMilkRatio': np.float64(str(self.settings['PelletMilkRatio'].text())),
                        'Chewing_TTLchan': np.int64(float(str(self.settings['Chewing_TTLchan'].text()))),
                        'MilkGoalRepetition': np.int64(float(str(self.settings['MilkGoalRepetition'].text()))),
                        'Username': str(self.settings['Username'].text()),
                        'Password': str(self.settings['Password'].text()),
                        'NegativeAudioSignal': np.float64(str(self.settings['NegativeAudioSignal'].text())),
                        'lightSignalProbability': np.float64(str(self.settings['lightSignalProbability'].text())),
                        'lightSignalIntensity': np.int64(str(self.settings['lightSignalIntensity'].text())),
                        'lightSignalDelay': np.float64(str(self.settings['lightSignalDelay'].text())),
                        'lightSignalPins': str(self.settings['lightSignalPins'].text()),
                        'InitPellets': np.int64(float(str(self.settings['InitPellets'].text()))),
                        'PelletQuantity': np.int64(float(str(self.settings['PelletQuantity'].text()))),
                        'PelletRewardMinSeparationMean': np.int64(float(str(self.settings['PelletRewardMinSeparationMean'].text()))),
                        'PelletRewardMinSeparationVariance': np.float64(str(self.settings['PelletRewardMinSeparationVariance'].text())),
                        'Chewing_Target': np.int64(float(str(self.settings['Chewing_Target'].text()))),
                        'MaxInactivityDuration': np.int64(float(str(self.settings['MaxInactivityDuration'].text()))),
                        'MilkTrialFailPenalty': np.int64(float(str(self.settings['MilkTrialFailPenalty'].text()))),
                        'InitMilk': np.float64(str(self.settings['InitMilk'].text())),
                        'MilkTrialMinSeparationMean': np.int64(float(str(self.settings['MilkTrialMinSeparationMean'].text()))),
                        'MilkTrialMinSeparationVariance': np.float64(str(self.settings['MilkTrialMinSeparationVariance'].text())),
                        'MilkTaskMinStartDistance': np.int64(float(str(self.settings['MilkTaskMinStartDistance'].text()))),
                        'MilkTaskMinGoalDistance': np.int64(float(str(self.settings['MilkTaskMinGoalDistance'].text()))),
                        'MilkTaskMinGoalAngularDistance': np.int64(float(str(self.settings['MilkTaskMinGoalAngularDistance'].text()))),
                        'MilkTaskGoalAngularDistanceTime': np.float64(float(str(self.settings['MilkTaskGoalAngularDistanceTime'].text()))),
                        'MilkTrialMaxDuration': np.int64(float(str(self.settings['MilkTrialMaxDuration'].text())))}
        # Get boolean selection for Active Game settings
        TaskSettings['games_active'] = {}
        for key in self.settings['games_active'].keys():
            state = self.settings['games_active'][key].isChecked()
            TaskSettings['games_active'][key] = np.array(state)
        # Get milk reward quantity options
        TaskSettings['MilkQuantity'] = {}
        for key in self.settings['MilkQuantity']:
            TaskSettings['MilkQuantity'][key] = np.float64(str(self.settings['MilkQuantity'][key].text()))
        # Get radio button selection
        for key in self.settings['AudioSignalMode'].keys():
            if self.settings['AudioSignalMode'][key].isChecked():
                TaskSettings['AudioSignalMode'] = key
        for key in self.settings['MilkGoalNextFeederMethod'].keys():
            if self.settings['MilkGoalNextFeederMethod'][key].isChecked():
                TaskSettings['MilkGoalNextFeederMethod'] = key
        # Get boolean selection for LightSignal repetition trial settings
        TaskSettings['LightSignalOnRepetitions'] = {}
        for key in self.settings['LightSignalOnRepetitions'].keys():
            state = self.settings['LightSignalOnRepetitions'][key].isChecked()
            TaskSettings['LightSignalOnRepetitions'][key] = np.array(state)
        # Get FEEDER specific information
        FEEDERs = {}
        for feeder_type in self.settings['FEEDERs'].keys():
            if len(self.settings['FEEDERs'][feeder_type]) > 0:
                FEEDERs[feeder_type] = {}
                IDs = []
                for feeder in self.settings['FEEDERs'][feeder_type]:
                    IDs.append(str(int(str(feeder['ID'].text()))))
                    FEEDERs[feeder_type][IDs[-1]] = {'ID': IDs[-1], 
                                                     'Present': np.array(feeder['Present'].isChecked()), 
                                                     'Active': np.array(feeder['Active'].isChecked()), 
                                                     'IP': str(feeder['IP'].text()), 
                                                     'Position': np.array(list(map(int, str(feeder['Position'].text()).split(','))))}
                    if feeder_type == 'milk':
                        FEEDERs[feeder_type][IDs[-1]]['Spacing'] = np.int64(float(str(feeder['Spacing'].text())))
                        FEEDERs[feeder_type][IDs[-1]]['Clearence'] = np.int64(float(str(feeder['Clearence'].text())))
                        FEEDERs[feeder_type][IDs[-1]]['Angle'] = np.int64(float(str(feeder['Angle'].text())))
                        FEEDERs[feeder_type][IDs[-1]]['SignalHz'] = np.int64(float(str(feeder['SignalHz'].text())))
                        FEEDERs[feeder_type][IDs[-1]]['SignalHzWidth'] = np.int64(float(str(feeder['SignalHzWidth'].text())))
                        FEEDERs[feeder_type][IDs[-1]]['ModulHz'] = np.int64(float(str(feeder['ModulHz'].text())))
                # Check if there are duplicates of FEEDER IDs
                if any(IDs.count(ID) > 1 for ID in IDs):
                    raise ValueError('Duplicates of IDs in ' + feeder_type + ' feeders!')
            else:
                show_message('No ' + feeder_type + ' FEEDERs entered.')
        TaskSettings['FEEDERs'] = FEEDERs

        return TaskSettings

    def import_settings_to_gui(self, TaskSettings):
        # Load all settings
        for key in TaskSettings.keys():
            try:
                if isinstance(TaskSettings[key], np.ndarray) and TaskSettings[key].dtype == 'bool':
                    self.settings[key].setChecked(TaskSettings[key])
                elif key == 'games_active':
                    for repeat_key in TaskSettings['games_active'].keys():
                        state = TaskSettings['games_active'][repeat_key]
                        self.settings['games_active'][repeat_key].setChecked(state)
                elif key == 'AudioSignalMode':
                    for mode_key in self.settings['AudioSignalMode'].keys():
                        if TaskSettings['AudioSignalMode'] == mode_key:
                            self.settings['AudioSignalMode'][mode_key].setChecked(True)
                elif key == 'MilkGoalNextFeederMethod':
                    for mode_key in self.settings['MilkGoalNextFeederMethod'].keys():
                        if TaskSettings['MilkGoalNextFeederMethod'] == mode_key:
                            self.settings['MilkGoalNextFeederMethod'][mode_key].setChecked(True)
                elif key == 'LightSignalOnRepetitions':
                    for repeat_key in TaskSettings['LightSignalOnRepetitions'].keys():
                        state = TaskSettings['LightSignalOnRepetitions'][repeat_key]
                        self.settings['LightSignalOnRepetitions'][repeat_key].setChecked(state)
                elif key == 'FEEDERs':
                    for feeder_type in TaskSettings['FEEDERs'].keys():
                        for ID in sorted(TaskSettings['FEEDERs'][feeder_type].keys(), key=int):
                            FEEDER_settings = TaskSettings['FEEDERs'][feeder_type][ID]
                            self.addFeedersToList(feeder_type, FEEDER_settings)
                elif isinstance(TaskSettings[key], dict):
                    for sub_key in TaskSettings[key]:
                        self.settings[key][sub_key].setText(str(TaskSettings[key][sub_key]))
                elif key in self.settings.keys():
                    self.settings[key].setText(str(TaskSettings[key]))
            except:
                print('Failed to load setting: ' + str(key))


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
    posVector = np.nanmean(posVectors, axis=0)

    return posVector

def compute_mean_posHistory(posHistory):
    posHistory = np.array(posHistory)
    posHistory = posHistory[:, :2]
    mean_posHistory = np.nanmean(posHistory, axis=0)

    return mean_posHistory

def compute_movement_angular_distance_to_target(posHistory, target_location):
    """
    Computes angular distance between mean movement vector and direct path to target location.
    Outputs None if norm of mean movement vector is 0.
    """
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


class PelletChoice(object):
    """
    Selects the next feeder using positional occupancy.

    Query read PelletChoice.ID for current feeder.
    Use PelletChoice.next() method to choose next feeder.
    """

    def __init__(self, PelletRewardDevices, position_histogram_dict, arena_size):
        self.PelletRewardDevices = PelletRewardDevices
        self.position_histogram_dict = position_histogram_dict
        self.arena_size = arena_size
        self.next()

    def _compute_position_histogram_nearest_feeders(self, IDs_active):
        self.old_IDs_active = IDs_active
        # Get feeder locations
        FEEDER_Locs = []
        for ID in IDs_active:
            FEEDER_Locs.append(np.array(self.PelletRewardDevices.positions[ID], dtype=np.float32))
        # Get occupancy histogram information from position_histogram_dict
        self.old_PelletHistogramParameters = deepcopy(self.position_histogram_dict['parameters'])
        self.histogramPfeederMap = {}
        # Convert histogram edges to bin centers
        histXedges = self.position_histogram_dict['edges']['x']
        histYedges = self.position_histogram_dict['edges']['y']
        histXbin = (histXedges[1:] + histXedges[:-1]) / 2
        histYbin = (histYedges[1:] + histYedges[:-1]) / 2
        # Crop data to only include parts inside the arena boundaries
        idx_X = np.logical_and(0 < histXbin, histXbin < self.arena_size[0])
        idx_Y = np.logical_and(0 < histYbin, histYbin < self.arena_size[1])
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
                dists = np.zeros(len(IDs_active), dtype=np.float32)
                for n_feeder in range(len(IDs_active)):
                    dists[n_feeder] = np.linalg.norm(FEEDER_Locs[n_feeder] - pos)
                histFeeder[ypos, xpos] = np.argmin(dists)
        self.histogramPfeederMap['feeder_map'] = histFeeder

    def _ensure_nearest_feeder_map_valid(self, IDs_active):
        histparam = deepcopy(self.position_histogram_dict['parameters'])
        histogram_same = hasattr(self, 'old_PelletHistogramParameters') and \
                                 self.old_PelletHistogramParameters == histparam
        feeders_same = hasattr(self, 'old_IDs_active') and \
                               self.old_IDs_active == IDs_active
        if not histogram_same or not feeders_same:
            self._compute_position_histogram_nearest_feeders(IDs_active)

    @staticmethod
    def weighted_randomness(weights):
        relative_weights = (np.sum(weights) - weights) ** 2
        probability = relative_weights / np.sum(relative_weights)
        
        return np.random.choice(len(probability), p=probability)

    def next(self):
        """
        Uses relative mean occupancy in bins closest to each feeder
        to increase probability of selecting feeder with lower mean occupancy.
        """
        IDs_active = copy(self.PelletRewardDevices.IDs_active)
        self._ensure_nearest_feeder_map_valid(IDs_active)
        if len(IDs_active) > 1:
            # Get occupancy information from position_histogram_dict
            histmap = self.position_histogram_dict['data']
            # Crop histogram to relavant parts
            histmap = histmap[self.histogramPfeederMap['idx_crop_Y'], self.histogramPfeederMap['idx_crop_X']]
            # Find mean occupancy in bins nearest to each feeder
            feeder_bin_occupancy = np.zeros(len(IDs_active), dtype=np.float64)
            for n_feeder in range(len(IDs_active)):
                bin_occupancies = histmap[self.histogramPfeederMap['feeder_map'] == n_feeder]
                feeder_bin_occupancy[n_feeder] = np.mean(bin_occupancies)
            # Choose feeder with weighted randomness if any parts occupied
            if np.any(feeder_bin_occupancy > 0):
                n_feeder = PelletChoice.weighted_randomness(feeder_bin_occupancy)
            else:
                n_feeder = np.random.choice(len(IDs_active))
        else:
            n_feeder = 0
        # Update current feeder ID
        self.ID = IDs_active[n_feeder]

        return self.ID


class MilkGoal(object):
    """
    Determines the sequence of milk feeder goal decisions.

    Query read MilkGoal.ID for current feeder.
    Use MilkGoal.next() method to choose next feeder.
    """
    def __init__(self, activeMfeeders, next_feeder_method='random', repetitions=0):
        """
        activeMfeeders - list - elements are returned with next() method as choices
        next_feeder_method - str - 'random' (default) or 'cycle'
        repetitions - int - number of repetitions to do per each feeder
        """
        self.activeMfeeders = activeMfeeders
        self.next_feeder_method = next_feeder_method
        self.repetitions = repetitions
        if self.next_feeder_method == 'cycle':
            self._initialize_sequence()
            self.next()
        elif self.next_feeder_method == 'weighted':
            self.ID = self.activeMfeeders[MilkGoal.choose_randomly(self.activeMfeeders)]
        elif self.next_feeder_method == 'random':
            self.next()
        else:
            raise ValueError('Unexpected next_feeder_method argument.')

    def _initialize_sequence(self):
        """
        Initializes the sequence of feeders and initial position.
        """
        if self.next_feeder_method == 'cycle':
            self.sequence = range(len(self.activeMfeeders))
            np.random.shuffle(self.sequence)
            # Set repetition counter and position to very last in sequence,
            # so that first call to next() method would start sequence from beginning.
            self.repetition_counter = self.repetitions
            self.sequence_position = len(self.activeMfeeders)

    def re_init(self, activeMfeeders=None, next_feeder_method=None, repetitions=None):
        """
        Allows re-initializing the class with any subset of input variables.
        """
        if not (activeMfeeders is None):
            self.activeMfeeders = activeMfeeders
        if not (next_feeder_method is None):
            self.next_feeder_method = next_feeder_method
        if not (repetitions is None):
            self.repetitions = repetitions
        if self.next_feeder_method == 'cycle':
            self._initialize_sequence()
        self.next()

    @staticmethod
    def choose_with_weighted_randomness(activeMfeeders, game_counters):
        """
        Chooses feeder from list with weighted randomness that is based on
        performance in the task. Feeders that where there has been
        fewer successful trials are more likely to be chosen.
        """
        # Find number of successful trials for each feeder
        n_successful_trials = []
        for ID in activeMfeeders:
            idx = game_counters['Successful']['ID'].index(ID)
            n_successful_trials.append(game_counters['Successful']['count'][idx])
        n_successful_trials = np.array(n_successful_trials, dtype=np.float64)
        # Choose feeder with weighted randomness if any trial has been successful
        if np.any(n_successful_trials > 0):
            feederProbabilityWeights = np.sum(n_successful_trials) - n_successful_trials
            feederProbability = feederProbabilityWeights / np.sum(feederProbabilityWeights)
            n_feeder = np.random.choice(len(activeMfeeders), p=feederProbability)
        else:
            # If not trial has been successful, pick feeder randomly
            n_feeder = np.random.choice(len(activeMfeeders))

    @staticmethod
    def choose_randomly(activeMfeeders):
        return np.random.choice(len(activeMfeeders))

    def check_if_first_repetition(self):
        if self.repetitions > 0:
            return self.repetition_counter == 0
        else:
            return True

    def copy_ID(self):
        return copy(self.ID)

    def next(self, game_counters=None):
        """
        Selects the next feeder chosen from the activeMfeeders list elements, 
        using the choice method provided during initialization.

        game_counters - dict with specific structure (see choose_with_weighted_randomness() method)
                        only required for weighted random choice.
        """
        if self.next_feeder_method == 'random':
            n_feeder = MilkGoal.choose_randomly(self.activeMfeeders)
        elif self.next_feeder_method == 'weighted':
            if game_counters is None:
                raise Exception('game_counters input required for weigthed randomness ' \
                                + 'in next MilkGoal decision.')
            n_feeder = MilkGoal.choose_with_weighted_randomness(self.activeMfeeders, 
                                                                game_counters)
        elif self.next_feeder_method == 'cycle':
            self.repetition_counter += 1
            if self.repetition_counter > self.repetitions:
                # If counter has maxed out, move up in sequence position and reset counter
                self.sequence_position = np.mod(self.sequence_position + 1, len(self.sequence))
                self.repetition_counter = 0
            n_feeder = self.sequence[self.sequence_position]
        # Set current ID of milk goal
        self.ID = self.activeMfeeders[n_feeder]


class RewardDevices(object):

    def __init__ (self, FEEDERs, feeder_type, username, password, 
                  feeder_kwargs={}, inactivation_signal=None):
        """
        FEEDERs            - dict - key (ID) and value (feeder specific parameters)
        feeder_type        - str - passed on to RPiInterface.RewardControl
        username           - str - username for all the Raspberry Pis in FEEDERs dict
        password           - str - password for all the Raspberry Pis in FEEDERs dict
        feeder_kwargs      - dict - key (ID) and value (feeder specific kwargs to be
                           passed to RPiInterfaces.RwardControl)
        inactivation_signal - this method is called with feeder_type and ID as argument 
                              when a feeder is being inactivated.
        """
        # Parse input arguments
        self.FEEDERs = FEEDERs
        self.feeder_type = feeder_type
        self.username = username
        self.password = password
        self.FEEDER_kwargs = feeder_kwargs
        self._inactivation_signal = inactivation_signal
        # Create Locks for all feeders
        self.FEEDERs_Locks = {}
        for ID in self.FEEDERs.keys():
            self.FEEDERs_Locks[ID] = Lock()
        # Create a list of feeders present
        self.IDs_present = []
        for ID in self.FEEDERs.keys():
            if self.FEEDERs[ID]['Present']:
                self.IDs_present.append(ID)
        self.IDs_present = sorted(self.IDs_present, key=int)
        # Create dictionary of feeder positions
        self.positions = {}
        for ID in self.FEEDERs.keys():
            self.positions[ID] = copy(self.FEEDERs[ID]['Position'])
        # Initialize all feeders concurrently
        T_initFEEDER = []
        for ID in self.FEEDERs.keys():
            if (self.FEEDERs[ID]['Active'] 
                    or (ID in feeder_kwargs 
                        and 'negativeAudioSignal' in feeder_kwargs[ID] 
                        and feeder_kwargs[ID]['negativeAudioSignal'] > 0)):
                T = Thread(target=self._initFEEDER, args=(ID,))
                T.start()
                T_initFEEDER.append(T)
        for T in T_initFEEDER:
            T.join()
        # Create a list of active feeder IDs
        self.IDs_active = []
        for ID in self.FEEDERs.keys():
            if self.FEEDERs[ID]['Active'] and self.FEEDERs[ID]['init_successful']:
                self.IDs_active.append(ID)
        self.IDs_active = sorted(self.IDs_active, key=int)

    def _initFEEDER(self, ID):
        with self.FEEDERs_Locks[ID]:
            IP = self.FEEDERs[ID]['IP']
        try:
            kwargs = self.FEEDER_kwargs[ID] if ID in self.FEEDER_kwargs.keys() else {}
            actuator = RewardControl(self.feeder_type, IP, self.username, 
                                     self.password, **kwargs)
            with self.FEEDERs_Locks[ID]:
                self.FEEDERs[ID]['actuator'] = actuator
                self.FEEDERs[ID]['init_successful'] = True
        except Exception as e:
            from inspect import currentframe, getframeinfo
            frameinfo = getframeinfo(currentframe())
            print('Error in ' + frameinfo.filename + ' line ' + str(frameinfo.lineno - 3))
            print('initFEEDER failed for: ' + IP)
            print(e)
            with self.FEEDERs_Locks[ID]:
                self.FEEDERs[ID]['init_successful'] = False

    def actuator_method_call(self, ID, method_name, *args, **kwargs):
        """
        ID          - str - identifies which feeder to use
        method_name - str - name of the method to call

        Method will be called with any following args and kwargs
        """
        with self.FEEDERs_Locks[ID]:
            if 'actuator' in self.FEEDERs[ID].keys():
                return getattr(self.FEEDERs[ID]['actuator'], method_name)(*args, **kwargs)
            else:
                warnings.warn('FEEDER ' + str(ID) + ' does not have an active acutator.')

    def inactivate_feeder(self, ID):
        # Remove feeder from active feeder list
        self.IDs_active.remove(ID)
        # Invoke the deactivation signal if provided during initialization
        if not (self._inactivation_signal is None):
            self._inactivation_signal(self.feeder_type, ID)
        # Close actuator for this feeder
        with self.FEEDERs_Locks[ID]:
            if 'actuator' in self.FEEDERs[ID].keys():
                actuator = self.FEEDERs[ID].pop('actuator')
                if hasattr(actuator, 'close'):
                    # Deactivate in a separate thread in case it crashes
                    Thread(target=actuator.close).start()

    def close(self):
        for ID in self.FEEDERs.keys():
            if 'actuator' in self.FEEDERs[ID].keys():
                if hasattr(self.FEEDERs[ID]['actuator'], 'close'):
                    self.FEEDERs[ID]['actuator'].close()


class ChewingTracker(object):

    def __init__(self, chewing_ttl_chan):
        self.chewing_ttl_chan = chewing_ttl_chan
        self.ttlTimes = np.array([], dtype=np.float64)
        self.ttlTimesLock = Lock()

    def check_for_chewing_message(self, message):
        parts = message.split()
        if parts[2] == str(self.chewing_ttl_chan) and parts[3] == str(1):
            with self.ttlTimesLock:
                self.ttlTimes = np.append(self.ttlTimes, time())

    def number_of_chewings(self, time_from):
        with self.ttlTimesLock:
            chewing_times = copy(self.ttlTimes)
        n_chewings = np.sum(chewing_times > time_from)

        return n_chewings


class Abstract_Variables(object):
    """
    If self._variable_state_update_pending is set True, get() and get_all()
    methods only return after a new update has taken place.
    """

    def __init__(self):
        """
        Can be reimplemented but must be called as well with super.
        """
        # Create variable states
        self._variable_states = []
        self._variable_state_names = []
        self._variable_states_Lock = Lock()
        self._variable_state_update_pending = True
        # Compute variable states for first time
        self._recompute_dynamic_variables()
        # Create a list of relevance time markers
        self._last_relevance = [time()] * len(self._variable_state_names)
        # Start updating loop in the background
        VARIABLE_UPDATE_RATE = 10
        self._start_updating(VARIABLE_UPDATE_RATE)

    def _update_variable_states(self, variable_states, variable_state_names):
        # Update variable states
        with self._variable_states_Lock:
            self._variable_states = variable_states
        self._variable_state_names = variable_state_names
        # Mark that no update is pending
        self._variable_state_update_pending = False

    def _wait_until_pending_update_complete(self):
        while self._variable_state_update_pending:
            sleep(0.005)

    def _recompute_dynamic_variables(self):
        """
        Must be re-implemented.

        This method must finish by calling self._update_variable_states method.
        This method is called iteratively to keep track of dynamic variables.
        """
        raise NotImplementedError
        self._update_variable_states(variable_states, variable_state_names)

    def _update_loop(self, update_rate):
        loop_clock = pygame.time.Clock()
        while self._loop_active:
            self._recompute_dynamic_variables()
            loop_duration = loop_clock.tick(update_rate)
            if loop_duration > (25 + 1000 / update_rate):
                warnings.warn('Method _recompute_dynamic_variables in ' + str(type(self)) + \
                              ' runs slower than assigned ' + \
                              'update rate ' + str(update_rate) + ' Hz. Duration ' + \
                              str(loop_duration) + ' ms.', RuntimeWarning)

    def _start_updating(self, update_rate):
        self._loop_active = True
        self._T_loop = Thread(target=self._update_loop, 
                              args=(update_rate,))
        self._T_loop.start()

    def _stop_updating(self):
        self._loop_active = False
        self._T_loop.join()

    def get(self, name, key, set_relevant=False):
        self._wait_until_pending_update_complete()
        if name in self._variable_state_names:
            if set_relevant:
                self._set_relevant(name)
            with self._variable_states_Lock:
                return copy(self._variable_states[self._variable_state_names.index(name)][key])
        else:
            return None

    def get_all(self):
        self._wait_until_pending_update_complete()
        with self._variable_states_Lock:
            return copy(self._variable_states)

    def get_all_relevance(self):
        """
        Returns list of time in seconds for each variable in _variable_states list
        since this was last set as relevant.
        """
        return [time() - rel for rel in self._last_relevance]

    def _set_relevant(self, name):
        self._last_relevance[self._variable_state_names.index(name)] = time()

    def close(self):
        self._stop_updating()


class GenericGame_Variables(Abstract_Variables):

    def __init__(self, TaskSettings, processed_position_list):
        # Parse inputs
        self.processed_position_list = processed_position_list
        # Parse TaskSettings
        self._distance_steps = TaskSettings['distance_steps']
        self._last_travel_smoothing = TaskSettings['LastTravelSmooth']
        self._last_travel_min_dist = TaskSettings['LastTravelDist']
        # Initialize position data for use in computing variable states
        self._initialize_position_data_for_update_variable_states()
        # Proceed with original __init__ method
        super(GenericGame_Variables, self).__init__()

    def _initialize_position_data_for_update_variable_states(self):
        posHistory = [None]
        while None in posHistory:
            if len(self.processed_position_list) > self._distance_steps:
                posHistory = copy(self.processed_position_list[-self._distance_steps:])
            else:
                posHistory = [None]
            if not (None in posHistory):
                self._lastKnownPosHistory = posHistory
            else:
                sleep(0.01)

    def _recompute_dynamic_variables(self):
        # Get animal position history
        posHistory = self.processed_position_list[-self._distance_steps:]
        if not (None in posHistory):
            self._lastKnownPosHistory = posHistory[-1]
        else:
            posHistory = [self._lastKnownPosHistory] * self._distance_steps
        # Compute all game progress variables
        variable_states = []
        variable_state_names = []
        # Check if has been moving enough in the last few seconds
        variable_state_names.append('mobility')
        total_distance = compute_distance_travelled(posHistory, self._last_travel_smoothing)
        variable_states.append({'name': 'Mobility', 
                                'target': self._last_travel_min_dist, 
                                'status': int(round(total_distance)), 
                                'complete': total_distance >= self._last_travel_min_dist, 
                                'percentage': total_distance / float(self._last_travel_min_dist)})
        # Update variable states
        self._update_variable_states(variable_states, variable_state_names)


class PelletGame_Variables(Abstract_Variables):

    def __init__(self, TaskSettings, ChewingTracker):
        # Parse inputs
        self._ChewingTracker = ChewingTracker
        # Parse TaskSettings
        self._distance_steps = TaskSettings['distance_steps']
        self._one_second_steps = TaskSettings['one_second_steps']
        self._chewing_target = TaskSettings['Chewing_Target']
        self._reward_min_separation_mean = TaskSettings['PelletRewardMinSeparationMean']
        self._reward_min_separation_variance = TaskSettings['PelletRewardMinSeparationVariance']
        self._max_inactivity_duration = TaskSettings['MaxInactivityDuration']
        # Create timers
        self._last_reward = time()
        self.update_reward_min_separation()
        # Proceed with original __init__ method
        super(PelletGame_Variables, self).__init__()

    def update_reward_min_separation(self):
        mean_val = self._reward_min_separation_mean
        var_val = self._reward_min_separation_variance
        jitter = [int(- mean_val * var_val), int(mean_val * var_val)]
        jitter = random.randint(jitter[0], jitter[1])
        new_val = int(mean_val + jitter)
        self._reward_min_separation = new_val
        # Ensure variables are updated using this new value before they are observed
        self._variable_state_update_pending = True

    def update_last_reward(self):
        self._last_reward = time()
        # Ensure variables are updated using this new value before they are observed
        self._variable_state_update_pending = True

    def _recompute_dynamic_variables(self):
        # Compute all game progress variables
        variable_states = []
        variable_state_names = []
        # Check if animal has been without pellet reward for too long
        timeSinceLastReward = time() - self._last_reward
        variable_state_names.append('inactivity')
        variable_states.append({'name': 'Inactivity', 
                                'target': self._max_inactivity_duration, 
                                'status': int(round(timeSinceLastReward)), 
                                'complete': timeSinceLastReward >= self._max_inactivity_duration, 
                                'percentage': timeSinceLastReward / float(self._max_inactivity_duration)})
        # Check if animal has been chewing enough since last reward
        if self._chewing_target > 0:
            variable_state_names.append('chewing')
            n_chewings = self._ChewingTracker.number_of_chewings(self._last_reward)
            variable_states.append({'name': 'Chewing', 
                                    'target': self._chewing_target, 
                                    'status': n_chewings, 
                                    'complete': n_chewings >= self._chewing_target, 
                                    'percentage': n_chewings / float(self._chewing_target)})
        else:
            variable_state_names.append('chewing')
            variable_states.append({'name': 'Chewing', 
                                    'target': self._chewing_target, 
                                    'status': 0, 
                                    'complete': True, 
                                    'percentage': 0})
        # Check if enough time as passed since last pellet reward
        timeSinceLastPelletReward = time() - self._last_reward
        variable_state_names.append('time_since_last_pellet')
        variable_states.append({'name': 'Since Pellet', 
                                'target': self._reward_min_separation, 
                                'status': int(round(timeSinceLastPelletReward)), 
                                'complete': timeSinceLastPelletReward >= self._reward_min_separation, 
                                'percentage': timeSinceLastPelletReward / float(self._reward_min_separation)})
        # Update variable states
        self._update_variable_states(variable_states, variable_state_names)


class MilkGame_Variables(Abstract_Variables):

    def __init__(self, TaskSettings, processed_position_list, MilkRewardDevices, MilkGoal):
        # Parse inputs
        self.processed_position_list = processed_position_list
        self._MilkRewardDevices = MilkRewardDevices
        self._MilkGoal = MilkGoal
        # Parse TaskSettings
        self._distance_steps = TaskSettings['distance_steps']
        self._one_second_steps = TaskSettings['one_second_steps']
        self._angular_distance_steps = TaskSettings['angular_distance_steps']
        self._max_distance_in_arena = TaskSettings['max_distance_in_arena']
        self._min_trial_separation_mean = TaskSettings['MilkTrialMinSeparationMean']
        self._min_trial_separation_variance = TaskSettings['MilkTrialMinSeparationVariance']
        self._min_start_distance = TaskSettings['MilkTaskMinStartDistance']
        self._min_angular_distance = TaskSettings['MilkTaskMinGoalAngularDistance']
        self._min_goal_distance = TaskSettings['MilkTaskMinGoalDistance']
        self._max_trial_duration = TaskSettings['MilkTrialMaxDuration']
        # Create timers
        self._last_trial = time()
        self.update_min_trial_separation()
        # Initialize position data for use in computing variable states
        self._initialize_position_data_for_update_variable_states()
        # Proceed with original __init__ method
        super(MilkGame_Variables, self).__init__()

    def _initialize_position_data_for_update_variable_states(self):
        posHistory = [None]
        while None in posHistory:
            if len(self.processed_position_list) > self._distance_steps:
                posHistory = copy(self.processed_position_list[-self._distance_steps:])
            else:
                posHistory = [None]
            if not (None in posHistory):
                self._lastKnownPosHistory = posHistory
            else:
                sleep(0.01)

    def _update_feeder_distances(self, posHistory_one_second_steps):
        mean_posHistory = compute_mean_posHistory(posHistory_one_second_steps)
        distances = []
        for ID in self._MilkRewardDevices.IDs_present:
            distances.append(euclidean(mean_posHistory, self._MilkRewardDevices.positions[ID]))
        self._feeder_distances = distances

    def closest_feeder_ID(self):
        return self._MilkRewardDevices.IDs_present[np.argmin(self._feeder_distances)]

    def update_min_trial_separation(self):
        mean_val = self._min_trial_separation_mean
        var_val = self._min_trial_separation_variance
        jitter = [int(- mean_val * var_val), int(mean_val * var_val)]
        jitter = random.randint(jitter[0], jitter[1])
        new_val = int(mean_val + jitter)
        self._min_trial_separation = new_val
        # Ensure variables are updated using this new value before they are observed
        self._variable_state_update_pending = True

    def update_last_trial(self):
        self._last_trial = time()
        # Ensure variables are updated using this new value before they are observed
        self._variable_state_update_pending = True

    def _recompute_dynamic_variables(self):
        # Get animal position history
        max_len = max([self._distance_steps, self._one_second_steps, self._angular_distance_steps])
        posHistory_max_length = self.processed_position_list[-max_len:]
        posHistory = posHistory_max_length[-self._distance_steps:]
        posHistory_one_second_steps = posHistory_max_length[-self._one_second_steps:]
        posHistory_for_angularDistance = posHistory_max_length[-self._angular_distance_steps:]
        # If animal position history is flaulty, use last known position as current static position
        if None in posHistory or None in posHistory_one_second_steps or None in posHistory_for_angularDistance:
            posHistory_one_second_steps = [self._lastKnownPosHistory] * self._one_second_steps
            posHistory_for_angularDistance = [self._lastKnownPosHistory] * self._angular_distance_steps
        else:
            self._lastKnownPosHistory = posHistory[-1]
        # Compute distances to all active milk feeders
        self._update_feeder_distances(posHistory_one_second_steps)
        # Compute all game progress variables
        variable_states = []
        variable_state_names = []
        # Check if enough time as passed since last milk trial
        variable_state_names.append('time_since_last_milk_trial')
        timeSinceLastMilkTrial = time() - self._last_trial
        variable_states.append({'name': 'Since Trial', 
                                'target': self._min_trial_separation, 
                                'status': int(round(timeSinceLastMilkTrial)), 
                                'complete': timeSinceLastMilkTrial >= self._min_trial_separation, 
                                'percentage': timeSinceLastMilkTrial / float(self._min_trial_separation)})
        # Check if animal is far enough from milk rewards
        variable_state_names.append('distance_from_milk_feeders')
        minDistance = min(self._feeder_distances)
        variable_states.append({'name': 'Milk Distance', 
                                'target': self._min_start_distance, 
                                'status': int(round(minDistance)), 
                                'complete': minDistance >= self._min_start_distance, 
                                'percentage': minDistance / float(self._min_start_distance)})
        # Check if animal not moving towards goal location
        variable_state_names.append('angular_distance_from_goal_feeder')
        target_location = self._MilkRewardDevices.positions[self._MilkGoal.copy_ID()]
        angularDistance = compute_movement_angular_distance_to_target(posHistory_for_angularDistance, 
                                                                      target_location)
        if angularDistance is None:
            angularDistance = 0
        variable_states.append({'name': 'Milk A.Distance', 
                                'target': self._min_angular_distance, 
                                'status': int(round(angularDistance)), 
                                'complete': angularDistance >= self._min_angular_distance, 
                                'percentage': angularDistance / float(self._min_angular_distance)})
        # Check if animal is close enough to goal location
        variable_state_names.append('distance_from_goal_feeder')
        if self._MilkGoal.copy_ID() in self._MilkRewardDevices.IDs_active:
            # This may not be the case if the goal milk feeder has just been deactivated
            goal_distance = self._feeder_distances[self._MilkRewardDevices.IDs_present.index(self._MilkGoal.copy_ID())]
        else:
            goal_distance = self._max_distance_in_arena
        variable_states.append({'name': 'Goal Distance', 
                                'target': self._min_goal_distance, 
                                'status': int(round(goal_distance)), 
                                'complete': goal_distance <= self._min_goal_distance, 
                                'percentage': 1 - (goal_distance - self._min_goal_distance) / float(self._max_distance_in_arena)})
        # Check if animal is too close to goal incorrect location
        if len(self._MilkRewardDevices.IDs_present) > 1:
            variable_state_names.append('distance_from_other_feeders')
            other_distances = min([self._feeder_distances[i] for i in range(len(self._MilkRewardDevices.IDs_present)) if self._MilkRewardDevices.IDs_present[i] != self._MilkGoal.copy_ID()])
            variable_states.append({'name': 'Other Distance', 
                                    'target': self._min_goal_distance, 
                                    'status': int(round(other_distances)), 
                                    'complete': other_distances <= self._min_goal_distance, 
                                    'percentage': 1 - (other_distances - self._min_goal_distance) / float(self._max_distance_in_arena)})
        # Check if trial has been running for too long
        variable_state_names.append('milk_trial_duration')
        trial_run_time = time() - self._last_trial
        variable_states.append({'name': 'Trial Duration', 
                                'target': self._max_trial_duration, 
                                'status': int(round(trial_run_time)), 
                                'complete': trial_run_time > self._max_trial_duration, 
                                'percentage': trial_run_time / float(self._max_trial_duration)})
        # Update variable states
        self._update_variable_states(variable_states, variable_state_names)


class Variables(object):
    def __init__(self, TaskSettings, processed_position_list, ChewingTracker=None,
                 MilkRewardDevices=None, MilkGoal=None):
        self._names = []
        self._instances = {}
        # Instantiate Generic variables
        self._instances['GenericGame_Variables'] = GenericGame_Variables(TaskSettings, processed_position_list)
        self._names.append('GenericGame_Variables')
        # Instantiate Pellet Game variables if pellet game active
        if TaskSettings['games_active']['pellet']:
            self._instances['PelletGame_Variables'] = PelletGame_Variables(TaskSettings, ChewingTracker)
            self._names.append('PelletGame_Variables')
        # Instantiate Milk Game variables if milk game active
        if TaskSettings['games_active']['milk']:
            self._instances['MilkGame_Variables'] = MilkGame_Variables(TaskSettings, processed_position_list,
                                                                       MilkRewardDevices, MilkGoal)
            self._names.append('MilkGame_Variables')

    def get(self, name):
        return self._instances[name]

    def full_list(self):
        full_list = []
        for names in self._names:
            full_list += self._instances[names].get_all()

        return full_list

    def full_list_relevance(self):
        full_list_relevance = []
        for names in self._names:
            full_list_relevance += self._instances[names].get_all_relevance()

        return full_list_relevance

    def set_dict(self):
        return self._instances

    def close(self):
        for name in self._names:
            self._instances[name].close()


class VariableDisplay(object):

    def __init__(self, position_on_screen, renderText, Variables):
        """
        position_on_screen - dict - ['top', 'bottom', 'left', 'right'] border of the data on screen
        renderText         - pygame.font.SysFont.render method for an existing font instace
        Variables          - Variables instance
        """
        self._renderText = renderText
        self._Variables = Variables
        self._pre_render(position_on_screen)

    def _get_all_variable_states(self):
        return self._Variables.full_list(), self._Variables.full_list_relevance()

    def _pre_render(self, position_on_screen):
        """
        Pre-computes variable state display positions and renders static text
        
        position_on_screen - dict - ['top', 'bottom', 'left', 'right'] border of the data on screen
                             provided to VariableDisplay.render method
        """
        variable_states, _ = self._get_all_variable_states()
        # Initialise random color generation
        random.seed(1232)
        rcolor = lambda: random.randint(0,255)
        # Compute progress bar locations
        textSpacing = 3
        textSpace = 10 + textSpacing
        textLines = 3
        xpos = np.linspace(position_on_screen['left'], position_on_screen['right'], 2 * len(variable_states))
        xlen = xpos[1] - xpos[0]
        xpos = xpos[::2]
        ybottompos = position_on_screen['bottom'] - textSpace * textLines
        ymaxlen = position_on_screen['bottom'] - position_on_screen['top'] - textSpace * textLines
        progress_bars = []
        for i, vs in enumerate(variable_states):
            progress_bars.append({'name_text': self._renderText(vs['name']), 
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

        self.progress_bars = progress_bars

    def render(self, screen, max_relevance_latency=0.2):
        """
        screen                - pygame.display.set_mode output
        max_relevance_latency - float - if more seconds since last relevance, colored dark gray
        """
        variable_states, relevance = self._get_all_variable_states()
        for vs, pb, rel in zip(variable_states, self.progress_bars, relevance):
            screen.blit(pb['name_text'], pb['name_position'])
            screen.blit(self._renderText('T: ' + str(vs['target'])), pb['target_position'])
            screen.blit(self._renderText('C: ' + str(vs['status'])), pb['value_position'])
            if rel < max_relevance_latency:
                color = pb['color'] # This could be turned into a condition to have state dependent color
            else:
                color = (30, 30, 30)
            if vs['complete']:
                ylen = int(round(pb['Position']['ymaxlen']))
                ypos = int(round(pb['Position']['ybottompos'] - pb['Position']['ymaxlen']))
                position = (pb['Position']['xpos'], ypos, pb['Position']['xlen'], ylen)
                draw_rect_with_border(screen, color, (255, 255, 255), position, border=2)
            else:
                ylen = int(round(vs['percentage'] * pb['Position']['ymaxlen']))
                ypos = pb['Position']['ybottompos'] - ylen
                position = (pb['Position']['xpos'], ypos, pb['Position']['xlen'], ylen)
                pygame.draw.rect(screen, color, position, 0)


class GameState(object):

    def __init__(self, **kwargs):
        """
        Can be re-reimplemented, but must accept **kwargs. These can be used
        further but must not be changed.
        """

    def pre_logic(self, **kwargs):
        """
        Is called once before GameState.repeat_logic method calls, using same input kwargs.

        Can be re-implemented.
        """
        pass

    def repeat_logic(self, **kwargs):
        """
        Must be re-implemented and must accept **kwargs.

        This method should utilize kwargs provided to __init__ and this method.

        The method is called repeatedly until it does not return None.
        
        Must return:
            next_state - str - name of the next state class
            kwargs     - dict - at minimum must be an empty dicitonary.
                         This is passed to the next state class repeat_logic method.
        """
        raise NotImplementedError

        return next_state, kwargs

    def enter(self, game_rate, **kwargs):
        """
        This should be called immediately after instantiation.
        This method blocks, calls GameState.repeat_logic repeatedly, 
        until it outputs the next_state and kwargs.

        """
        self._continue = True
        # Call GameState.pre_logic before successive GameState.repeat_logic method calls.
        self.pre_logic(**kwargs)
        # Keep calling GameState.repeat_logic until it does not return None
        ret = None
        loop_clock = pygame.time.Clock()
        while self._continue and ret is None:
            ret = self.repeat_logic(**kwargs)
            loop_duration = loop_clock.tick(game_rate)
            if loop_duration > (25 + 1000 / game_rate):
                warnings.warn('Method repeat_logic in ' + str(type(self)) + \
                              ' runs slower than assigned ' + \
                              'rate ' + str(game_rate) + ' Hz. Duration ' + \
                              str(loop_duration) + ' ms.', RuntimeWarning)
        # Expand output and return
        if self._continue:
            next_state, kwargs = ret
            return next_state, kwargs

    def exit(self):
        """
        Interrupts repeated GameState.repeat_logic calls and 
        causes GameState.enter method to return None.
        """
        self._continue = False


class GameState_Init_Rewards(GameState):
    """
    This state dispenses rewards if any are required during initialization.
    """
    def __init__(self, **kwargs):
        # Parse input arguments
        self._GameState_kwargs = kwargs
        self.MessageToOE = kwargs['MessageToOE']
        self.init_pellets = kwargs['InitPellets']
        self.init_milk = kwargs['InitMilk']
        if 'PelletRewardDevices' in kwargs.keys() and hasattr(kwargs['PelletRewardDevices'], 'IDs_active'):
            self.pellet_IDs_active = kwargs['PelletRewardDevices'].IDs_active
        else:
            self.pellet_IDs_active = []
        if 'MilkRewardDevices' in kwargs.keys() and hasattr(kwargs['MilkRewardDevices'], 'IDs_active'):
            self.milk_IDs_active = kwargs['MilkRewardDevices'].IDs_active
        else:
            self.milk_IDs_active = []

    def _release_pellet_reward(self, ID, quantity):
        game_state = GameState_PelletReward(**self._GameState_kwargs)
        state_specific_kwargs = {'action': 'GameState_Init_Rewards', 'ID': ID, 'quantity': quantity}
        game_state.enter(1, action='GameState_Init_Rewards', ID=ID, 
                         quantity=quantity, suppress_MessageToOE=True)

    def _release_milk_reward(self, ID, quantity):
        game_state = GameState_MilkReward(**self._GameState_kwargs)
        state_specific_kwargs = {'action': 'GameState_Init_Rewards', 'ID': ID, 'quantity': quantity}
        game_state.enter(1, action='GameState_Init_Rewards', ID=ID, 
                         quantity=quantity, suppress_MessageToOE=True)

    def _compute_pellet_rewards(self):
        minPellets = int(np.floor(float(self.init_pellets) / len(self.pellet_IDs_active)))
        extraPellets = np.mod(self.init_pellets, len(self.pellet_IDs_active))
        feeder_pellet_count = minPellets * np.ones(len(self.pellet_IDs_active), dtype=np.int16)
        feeder_pellet_count[:extraPellets] = feeder_pellet_count[:extraPellets] + 1

        return feeder_pellet_count

    def _dispense_pellet_rewards(self, feeder_pellet_count):
        T_rewards = []
        for ID, n_pellets in zip(self.pellet_IDs_active, feeder_pellet_count):
            if n_pellets > 0:
                T = Thread(target=self._release_pellet_reward, args=(ID, n_pellets))
                T.start()
                T_rewards.append(T)

        return T_rewards

    def _dispense_milk_rewards(self, quantity):
        T_rewards = []
        for ID in self.milk_IDs_active:
            T = Thread(target=self._release_milk_reward, args=(ID, quantity))
            T.start()
            T_rewards.append(T)

        return T_rewards

    def pre_logic(self, **kwargs):
        self.MessageToOE('GameState_Init_Rewards')
        # Collect threads into this list
        T_rewards = []
        # Dispense pellet rewards
        if len(self.pellet_IDs_active) > 0 and self.init_pellets > 0:
            feeder_pellet_count = self._compute_pellet_rewards()
            T_rewards += self._dispense_pellet_rewards(feeder_pellet_count)
        # Dispense milk rewards
        if len(self.milk_IDs_active) > 0 and self.init_milk > 0:
            T_rewards += self._dispense_milk_rewards(self.init_milk)
        # Ensure all threads have finished
        for T in T_rewards:
            T.join()

    def repeat_logic(self, **kwargs):
        return 'GameState_Interval', {}


class GameState_Interval_Pellet(GameState):

    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.PelletGame_Variables = kwargs['PelletGame_Variables']
        self.pellet_feeder_ID = kwargs['PelletChoice'].next()

    def pre_logic(self, **kwargs):
        self.MessageToOE('GameState_Interval_Pellet')

    def repeat_logic(self, **kwargs):
        # Acquire relavant variable states
        conditions = {'inactivity': self.PelletGame_Variables.get('inactivity', 'complete', set_relevant=True), 
                      'chewing': self.PelletGame_Variables.get('chewing', 'complete', set_relevant=True), 
                      'pellet_interval': self.PelletGame_Variables.get('time_since_last_pellet', 'complete', set_relevant=True)}
        if conditions['inactivity']:
            # If animal has been without any rewards for too long, release pellet reward
            return 'GameState_PelletReward', {'action': 'goal_inactivity', 'ID': self.pellet_feeder_ID}
        elif conditions['chewing'] and conditions['pellet_interval']:
            # If animal has been chewing and there has been sufficient time since last pellet reward
            return 'GameState_Pellet', {}


class GameState_Interval_Milk(GameState):

    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.MilkGame_Variables = kwargs['MilkGame_Variables']

    def pre_logic(self, **kwargs):
        self.MessageToOE('GameState_Interval_Milk')

    def repeat_logic(self, **kwargs):
        # Acquire relavant variable states
        conditions = {'milk_trial_interval': self.MilkGame_Variables.get('time_since_last_milk_trial', 'complete', set_relevant=True)}
        if conditions['milk_trial_interval']:
            # If sufficient time has passed since last milk trial
            return 'GameState_Milk', {}


class GameState_Interval_Pellet_And_Milk(GameState):

    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.PelletGame_Variables = kwargs['PelletGame_Variables']
        self.MilkGame_Variables = kwargs['MilkGame_Variables']
        self.pellet_milk_ratio = kwargs['PelletMilkRatio']
        self.pellet_feeder_ID = kwargs['PelletChoice'].next()

    def pre_logic(self, **kwargs):
        self.MessageToOE('GameState_Interval_Pellet_And_Milk')

    def _choose_subtask(self):
        if random.uniform(0, 1) < self.pellet_milk_ratio:
            subtask = 'GameState_Pellet'
        else:
            subtask = 'GameState_Milk'

        return subtask

    def repeat_logic(self, **kwargs):
        # Acquire relavant variable states
        conditions = {'inactivity': self.PelletGame_Variables.get('inactivity', 'complete', set_relevant=True), 
                      'chewing': self.PelletGame_Variables.get('chewing', 'complete', set_relevant=True), 
                      'pellet_interval': self.PelletGame_Variables.get('time_since_last_pellet', 'complete', set_relevant=True), 
                      'milk_trial_interval': self.MilkGame_Variables.get('time_since_last_milk_trial', 'complete', set_relevant=True)}
        if conditions['inactivity']:
            # If animal has been without any rewards for too long, release pellet reward
            return 'GameState_PelletReward', {'action': 'goal_inactivity', 'ID': self.pellet_feeder_ID}
        elif conditions['chewing'] and conditions['pellet_interval'] and conditions['milk_trial_interval']:
            # If conditions for pellet and milk state are met, choose one based on _choose_subtask method
            return self._choose_subtask(), {}
        elif conditions['chewing'] and conditions['pellet_interval'] and (not conditions['milk_trial_interval']):
            # If conditions are met for pellet but not for milk, change game state to pellet
            return 'GameState_Pellet', {}
        elif (not conditions['chewing']) and conditions['pellet_interval'] and conditions['milk_trial_interval']:
            # If conditions are met for milk but not for pellet, change game state to milk
            return 'GameState_Milk', {}


def GameState_Interval(**kwargs):
    games_active = kwargs['games_active']
    if games_active['pellet']and not games_active['milk']:
        return GameState_Interval_Pellet(**kwargs)
    elif games_active['milk'] and not games_active['pellet']:
        return GameState_Interval_Milk(**kwargs)
    elif games_active['pellet'] and games_active['milk']:
        return GameState_Interval_Pellet_And_Milk(**kwargs)


class GameState_Pellet(GameState):
    
    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.GenericGame_Variables = kwargs['GenericGame_Variables']
        self.PelletGame_Variables = kwargs['PelletGame_Variables']
        self.game_counters = kwargs['game_counters']
        self.feeder_ID = kwargs['PelletChoice'].next()

    def update_game_counters(self, ID):
        idx = self.game_counters['Pellets']['ID'].index(ID)
        self.game_counters['Pellets']['count'][idx] += 1

    def pre_logic(self, **kwargs):
        self.MessageToOE('GameState_Pellet')

    def repeat_logic(self, **kwargs):
        # Acquire relavant variable states
        conditions = {'inactivity': self.PelletGame_Variables.get('inactivity', 'complete', set_relevant=True), 
                      'mobility': self.GenericGame_Variables.get('mobility', 'complete', set_relevant=True)}
        if conditions['inactivity']:
            # If animal has been without any rewards for too long, release pellet reward
            self.update_game_counters(self.feeder_ID)
            return 'GameState_PelletReward', {'action': 'GameState_Pellet.inactivity', 'ID': self.feeder_ID}
        elif conditions['mobility']:
            # If the animal is mobile enough, release pellet reward
            self.update_game_counters(self.feeder_ID)
            return 'GameState_PelletReward', {'action': 'GameState_Pellet.mobility', 'ID': self.feeder_ID}


class GameState_PelletReward(GameState):
    """
    This state just means pellet reward is released.
    This is implemented in pre_logic method to avoid
    warnings about too slow processing in GameState.repeat_logic method.
    """
    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.quantity = kwargs['PelletQuantity']
        self.reward_device = kwargs['PelletRewardDevices']
        self.PelletGame_Variables = kwargs['PelletGame_Variables']

    def pre_logic(self, **kwargs):
        if not ('suppress_MessageToOE' in kwargs.keys()) or not kwargs['suppress_MessageToOE']:
            self.MessageToOE('GameState_PelletReward ' + kwargs['action'])
        # Parse input arguments
        ID = kwargs['ID']
        if 'quantity' in kwargs.keys():
            self.quantity = kwargs['quantity']
        # Send command to release reward and wait for positive feedback
        feedback = self.reward_device.actuator_method_call(ID, 'release', self.quantity)
        if feedback:
            # Send message to Open Ephys GUI
            OEmessage = 'Reward pellet ' + ID + ' ' + str(self.quantity)
            self.MessageToOE(OEmessage)
            # Reset last reward timer
            self.PelletGame_Variables.update_last_reward()
        else:
            # If failed, remove feeder from game and change button(s) red
            self.reward_device.inactivate_feeder(ID)
            # Send message to Open Ephys GUI
            OEmessage = 'FEEDER pellet ' + ID + ' inactivated'
            self.MessageToOE(OEmessage)

    def repeat_logic(self, **kwargs):
        return 'GameState_Interval', {}


class GameState_Milk(GameState):
    
    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.GenericGame_Variables = kwargs['GenericGame_Variables']
        self.MilkGame_Variables = kwargs['MilkGame_Variables']

    def pre_logic(self, **kwargs):
        self.MessageToOE('GameState_Milk')

    def _check_duration(self):
        if not hasattr(self, '_time_of_first_check_duration'):
            self._time_of_first_check_duration = time()
        return time() - self._time_of_first_check_duration

    def repeat_logic(self, **kwargs):
        # Acquire relavant variable states
        conditions = {'mobility': self.GenericGame_Variables.get('mobility', 'complete', set_relevant=True), 
                      'distance_from_milk_feeders': self.MilkGame_Variables.get('distance_from_milk_feeders', 'complete', set_relevant=True), 
                      'angular_distance_from_goal_feeder': self.MilkGame_Variables.get('angular_distance_from_goal_feeder', 'complete', set_relevant=True)}
        if conditions['distance_from_milk_feeders'] and conditions['mobility'] and conditions['angular_distance_from_goal_feeder']:
            # If animal is far enough from milk feeders and is mobile enough but not running to the goal, start milk trial
            return 'GameState_MilkTrial', {'action': 'GameState_Milk'}
        elif self._check_duration() > 60:
            # If this game state has been active for more than 60 seconds, move back to interval
            return 'GameState_Interval', {}


class GameState_MilkTrial(GameState):
    
    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.MilkGoal = kwargs['MilkGoal']
        self.MilkTrialSignals = kwargs['MilkTrialSignals']
        self.GenericGame_Variables = kwargs['GenericGame_Variables']
        self.MilkGame_Variables = kwargs['MilkGame_Variables']
        self.game_counters = kwargs['game_counters']

    def update_game_counters_started(self, ID):
        idx = self.game_counters['Milk trials']['ID'].index(ID)
        self.game_counters['Milk trials']['count'][idx] += 1

    def update_game_counters_successful(self, ID):
        idx = self.game_counters['Successful']['ID'].index(ID)
        self.game_counters['Successful']['count'][idx] += 1

    def pre_logic(self, **kwargs):
        # Check if this is the first repetition
        first_repetition = self.MilkGoal.check_if_first_repetition()
        # Send timestamp to Open Ephys GUI
        OEmessage = 'GameState_MilkTrial ' + kwargs['action'] + ' ' + self.MilkGoal.copy_ID()
        if first_repetition:
            OEmessage += ' presentation_trial'
        else:
            OEmessage += ' repeat_trial'
        self.MessageToOE(OEmessage)
        # Reset milk trial timers
        self.MilkGame_Variables.update_min_trial_separation()
        self.MilkGame_Variables.update_last_trial()
        # Start milk trial signals
        Thread(target=self.MilkTrialSignals.start, 
               args=(self.MilkGoal.copy_ID(), first_repetition)).start()
        # Update game counters
        self.update_game_counters_started(self.MilkGoal.copy_ID())

    def logic_first_trial(self, **kwargs):
        # Acquire relavant variable states
        conditions = {'distance_from_goal_feeder': self.MilkGame_Variables.get('distance_from_goal_feeder', 'complete', set_relevant=True), 
                      'milk_trial_duration': self.MilkGame_Variables.get('milk_trial_duration', 'complete', set_relevant=True)}
        if conditions['distance_from_goal_feeder']:
            # If subject reached goal location, proceed with reward
            self.update_game_counters_successful(self.MilkGoal.copy_ID())
            self.MilkTrialSignals.stop(self.MilkGoal.copy_ID())
            ID = self.MilkGoal.copy_ID()
            self.MilkGoal.next(game_counters=self.game_counters)
            return 'GameState_MilkReward', {'action': 'GameState_MilkTrial', 'ID': ID, 'trial_type': 'presentation'}
        elif conditions['milk_trial_duration']:
            # If time limit for task duration has passed, stop milk trial without reward.
            # Milk Trial goal is not updated if first trial fails.
            self.MilkTrialSignals.stop(self.MilkGoal.copy_ID())
            return 'GameState_MilkTrial_Fail', {'reason': 'timeout'}

    def logic_other_trial(self, **kwargs):
        # Acquire relavant variable states
        conditions = {'distance_from_goal_feeder': self.MilkGame_Variables.get('distance_from_goal_feeder', 'complete', set_relevant=True), 
                      'distance_from_other_feeders': self.MilkGame_Variables.get('distance_from_other_feeders', 'complete', set_relevant=True), 
                      'milk_trial_duration': self.MilkGame_Variables.get('milk_trial_duration', 'complete', set_relevant=True)}
        if conditions['distance_from_goal_feeder']:
            # If subject reached goal location, proceed with reward
            self.update_game_counters_successful(self.MilkGoal.copy_ID())
            self.MilkTrialSignals.stop(self.MilkGoal.copy_ID())
            ID = self.MilkGoal.copy_ID()
            self.MilkGoal.next(game_counters=self.game_counters)
            return 'GameState_MilkReward', {'action': 'GameState_MilkTrial', 'ID': ID, 'trial_type': 'repeat'}
        elif conditions['milk_trial_duration']:
            # If time limit for task duration has passed, stop milk trial without reward
            self.MilkTrialSignals.stop(self.MilkGoal.copy_ID())
            self.MilkGoal.next(game_counters=self.game_counters)
            return 'GameState_MilkTrial_Fail', {'reason': 'timeout'}
        elif not (conditions['distance_from_other_feeders'] is None) and conditions['distance_from_other_feeders']:
            # If subject went to incorrect location, stop milk trial with negative feedback
            self.MilkTrialSignals.stop(self.MilkGoal.copy_ID())
            self.MilkTrialSignals.fail(self.MilkGame_Variables.closest_feeder_ID())
            self.MilkGoal.next(game_counters=self.game_counters)
            return 'GameState_MilkTrial_Fail', {'reason': 'incorrect_feeder'}
        
    def repeat_logic(self, **kwargs):
        if self.MilkGoal.check_if_first_repetition():
            return self.logic_first_trial(**kwargs)
        else:
            return self.logic_other_trial(**kwargs)


class GameState_MilkTrial_Fail(GameState):
    """
    In this state the game halts for the duration of milk trial penalty.
    This is implemented in pre_logic to avoid warnings regarding slow processing
    in GameState.repeat_logic method.
    """
    
    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.penalty_duration = kwargs['MilkTrialFailPenalty']

    def pre_logic(self, **kwargs):
        self.MessageToOE('GameState_MilkTrial_Fail ' + kwargs['reason'])
        # Start penalty timer
        sleep(self.penalty_duration)

    def repeat_logic(self, **kwargs):
        return 'GameState_Interval', {}


class GameState_MilkReward(GameState):
    """
    This state just means milk reward is released.
    This is implemented in pre_logic method to avoid
    warnings about too slow processing in GameState.repeat_logic method.
    """
    def __init__(self, **kwargs):
        # Parse input arguments
        self.MessageToOE = kwargs['MessageToOE']
        self.MilkTrialSignals = kwargs['MilkTrialSignals']
        self.quantity = kwargs['MilkQuantity']
        self.reward_device = kwargs['MilkRewardDevices']

    def pre_logic(self, **kwargs):
        if not ('suppress_MessageToOE' in kwargs.keys()) or not kwargs['suppress_MessageToOE']:
            self.MessageToOE('GameState_MilkReward ' + kwargs['action'])
        # Parse input arguments
        ID = kwargs['ID']
        if 'quantity' in kwargs.keys():
            quantity = kwargs['quantity']
        else:
            if 'trial_type' in kwargs:
                quantity = self.quantity[kwargs['trial_type']]
            else:
                quantity = 0
                for key in self.quantity:
                    quantity = max(quantity, self.quantity[key])
        # Send command to release reward and wait for positive feedback
        feedback = self.reward_device.actuator_method_call(ID, 'release', quantity)
        if feedback:
            # Send message to Open Ephys GUI
            OEmessage = 'Reward milk ' + ID + ' ' + str(quantity)
            self.MessageToOE(OEmessage)
        else:
            # If failed, remove feeder from game and change button(s) red
            self.reward_device.inactivate_feeder(ID)
            # Send message to Open Ephys GUI
            OEmessage = 'FEEDER milk ' + ID + ' inactivated'
            self.MessageToOE(OEmessage)

    def repeat_logic(self, **kwargs):
        return 'GameState_Interval', {}


class GameStateOperator(object):

    def __init__(self, TaskSettings, MessageToOE, Variables, game_counters, game_state_display_update, 
                 PelletRewardDevices=None, PelletChoice=None, MilkRewardDevices=None, 
                 MilkTrialSignals=None, MilkGoal=None):
        """
        TaskSettings  - dict - see below which values are required
        MessageToOE   - method - this method is called with string to log messages
        Variables     - dict - instance of same name classes
        game_counters - dict - 
        game_state_display_update - method is called with string specifying current game state

        The following input arguments are necessary depending
        if pellet or milk game is active, based on TaskSettings['games_active'].
        These input arguments should be instances of the same name classes:

            PelletRewardDevices
            PelletChoice
            MilkRewardDevices
            MilkTrialSignals
            MilkGoal
        """
        # Specify game update rate
        self.game_rate = 5 # Game update rate in Hz
        # Parse game state display input
        self._game_state_display_update = game_state_display_update
        # Start parsing input
        kwargs = {}
        # Parse TaskSettings
        kwargs['games_active'] = TaskSettings['games_active']
        kwargs['PelletMilkRatio'] = TaskSettings['PelletMilkRatio']
        kwargs['PelletQuantity'] = TaskSettings['PelletQuantity']
        kwargs['MilkTrialFailPenalty'] = TaskSettings['MilkTrialFailPenalty']
        kwargs['MilkQuantity'] = TaskSettings['MilkQuantity']
        kwargs['InitPellets'] = TaskSettings['InitPellets']
        kwargs['InitMilk'] = TaskSettings['InitMilk']
        # Parse generic inputs
        kwargs['game_counters'] = game_counters
        kwargs['MessageToOE'] = MessageToOE
        kwargs['GenericGame_Variables'] = Variables.set_dict()['GenericGame_Variables']
        # Parse pellet game inputs
        if kwargs['games_active']['pellet']:
            kwargs['PelletGame_Variables'] = Variables.set_dict()['PelletGame_Variables']
            kwargs['PelletRewardDevices'] = PelletRewardDevices
            kwargs['PelletChoice'] = PelletChoice
        # Parse milk game inputs
        if kwargs['games_active']['milk']:
            kwargs['MilkGame_Variables'] = Variables.set_dict()['MilkGame_Variables']
            kwargs['MilkRewardDevices'] = MilkRewardDevices
            kwargs['MilkTrialSignals'] = MilkTrialSignals
            kwargs['MilkGoal'] = MilkGoal
        # Store for later use
        self._GameState_kwargs = kwargs

    def _display_update(self, game_state):
        if game_state.startswith('GameState_'):
            game_state_name = game_state[10:]
        else:
            game_state_name = game_state
        self._game_state_display_update(game_state_name)

    def _process(self):
        self._active = True
        self._display_update('GameState_Init_Rewards')
        self._GameState = GameState_Init_Rewards(**self._GameState_kwargs)
        ret = self._GameState.enter(self.game_rate, **{})
        while self._active:
            if ret is None:
                self._active = False
            else:
                next_state, next_state_kwargs = ret
                self._display_update(next_state)
                next_state_class = globals()[next_state]
                self._GameState = next_state_class(**self._GameState_kwargs)
                ret = self._GameState.enter(self.game_rate, **next_state_kwargs)

    def start(self):
        self._T__process = Thread(target=self._process)
        self._T__process.start()

    def close(self):
        self._active = False
        self._GameState.exit()
        self._T__process.join()


class MilkTrial_AudioSignal(object):

    def __init__(self, actuator_method_call, MessageToOE, AudioSignalMode, FEEDERsettings):
        """
        actuator_method_call - MilkRewardDevices.actuator_method_call method
        MessageToOE          - method is called when signal is started with a message string
        AudioSignalMode - str - 'ambient' or 'localised'
        FEEDERsettings  - dict - key (ID) and value (feeder specific parameters)
        """
        # Parse input
        self.actuator_method_call = actuator_method_call
        self.MessageToOE = MessageToOE
        self.AudioSignalMode = AudioSignalMode
        if self.AudioSignalMode == 'ambient':
            self._init_ambient_sounds(FEEDERsettings)

    def _init_ambient_sounds(self, FEEDERsettings):
        self.ambient_sounds = {}
        for ID in FEEDERsettings.keys():
            self.ambient_sounds[ID] = createAudioSignal(FEEDERsettings[ID]['SignalHz'], 
                                                        FEEDERsettings[ID]['SignalHzWidth'], 
                                                        FEEDERsettings[ID]['ModulHz'])

    def start(self, ID):
        OEmessage = 'AudioSignal Start'
        self.MessageToOE(OEmessage)
        if self.AudioSignalMode == 'ambient':
            self.ambient_sounds[ID].play(-1)
        else:
            self.actuator_method_call(ID, 'startTrialAudioSignal')

    def stop(self, ID):
        OEmessage = 'AudioSignal Stop'
        self.MessageToOE(OEmessage)
        if self.AudioSignalMode == 'ambient':
            self.ambient_sounds[ID].stop()
        else:
            self.actuator_method_call(ID, 'stopTrialAudioSignal')

    def play_negative(self, ID):
        OEmessage = 'NegativeAudioSignal Play'
        self.MessageToOE(OEmessage)
        self.actuator_method_call(ID, 'playNegativeAudioSignal')


class MilkTrial_LightSignal(object):

    def __init__(self, actuator_method_call, MessageToOE):
        """
        actuator_method_call - MilkRewardDevices.actuator_method_call method
        MessageToOE          - method is called when signal is started with a message string
        """
        # Parse input
        self.actuator_method_call = actuator_method_call
        self.MessageToOE = MessageToOE
        # Prepare internal variables
        self.light_on = False
        self._waiting_on_delay = False
        self._cancel_delay = False

    def start(self, ID):
        OEmessage = 'LightSignal Start'
        self.MessageToOE(OEmessage)
        self.actuator_method_call(ID, 'startLightSignal')
        self.light_on = True

    def stop(self, ID):
        if self._waiting_on_delay:
            self._cancel_delay = True
        else:
            OEmessage = 'LightSignal Stop'
            self.MessageToOE(OEmessage)
            self.actuator_method_call(ID, 'stopLightSignal')
            self.light_on = False

    def _delayed_starter(self, ID, delay):
        """
        delay - float - time to wait in seconds
        """
        self._waiting_on_delay = True
        sleep(delay)
        self._waiting_on_delay = False
        if not self._cancel_delay:
            self.start(ID)
        self._cancel_delay = False

    def start_delayed(self, ID, delay):
        Thread(target=self._delayed_starter, args=(ID, delay)).start()


class MilkTrialSignals(object):

    def __init__(self, TaskSettings, actuator_method_call, MessageToOE, FEEDERsettings=None):
        """
        TaskSettings    - dict - General Task settings. See below for what is used.
        actuator_method_call - MilkRewardDevices.actuator_method_call method
        MessageToOE          - method is called when signal is started with a message string
        FEEDERsettings  - dict - key (ID) and value (feeder specific parameters)
        """
        # Parse TaskSettings
        self.FirstTrialLightOn      = TaskSettings['LightSignalOnRepetitions']['presentation']
        self.OtherTrialLightOn      = TaskSettings['LightSignalOnRepetitions']['repeat']
        self.lightSignalDelay       = TaskSettings['lightSignalDelay']
        self.lightSignalProbability = TaskSettings['lightSignalProbability']
        AudioSignalMode             = TaskSettings['AudioSignalMode']
        # Initialize signals
        self.MilkTrial_AudioSignal = MilkTrial_AudioSignal(actuator_method_call, MessageToOE, 
                                                           AudioSignalMode, FEEDERsettings)
        if self.FirstTrialLightOn or self.OtherTrialLightOn:
            self.MilkTrial_LightSignal = MilkTrial_LightSignal(actuator_method_call, MessageToOE)

    def light_signal_probabilistic_determinant(self):
        if self.lightSignalProbability > 0.99:
            return True
        if random.random() < self.lightSignalProbability:
            return True
        else:
            return False

    def start(self, ID, first_repetition=False):
        self.MilkTrial_AudioSignal.start(ID)
        # Show light signal ONLY 
        # if its this goal has been achieved and other repetitions are set to have light signal 
        # OR 
        # if this goal has not been achieved and first repetition is set to have light signal.
        if first_repetition and self.FirstTrialLightOn:
            start_light_signal = True
        elif (not first_repetition) and self.OtherTrialLightOn:
            start_light_signal = True
        else:
            start_light_signal = False
        if start_light_signal:
            start_light_signal = self.light_signal_probabilistic_determinant()
        if start_light_signal:
            if self.lightSignalDelay > 0:
                self.MilkTrial_LightSignal.start_delayed(ID, self.lightSignalDelay)
            else:
                self.MilkTrial_LightSignal.start(ID)

    def stop(self, ID):
        self.MilkTrial_AudioSignal.stop(ID)
        if hasattr(self, 'MilkTrial_LightSignal') and self.MilkTrial_LightSignal.light_on:
            self.MilkTrial_LightSignal.stop(ID)

    def fail(self, ID):
        self.MilkTrial_AudioSignal.play_negative(ID)


class Buttons(object):

    def __init__(self, position_on_screen, renderText, 
                 PelletRewardDevices=None, MilkRewardDevices=None):
        """
        position_on_screen  - dict - ['top', 'bottom', 'left', 'right'] border of the buttons 
                              on screen provided to Buttons.render method
        renderText          - pygame.font.SysFont.render method for an existing font instace
        PelletRewardDevices - instance of class with same name.
                              If not provided, relevant buttons will not be created.
        MilkRewardDevices   - instance of class with same name.
                              If not provided, relevant buttons will not be created.
        """
        self._PelletRewardDevices = PelletRewardDevices
        self._MilkRewardDevices = MilkRewardDevices
        self._define()
        self.list = Buttons._pre_render(self.list, position_on_screen, renderText)

    def buttonGameOnOff_callback(self, button):
        pass

    def buttonReleaseReward_callback(self, button):
        if False:
            pass
        else:
            button['enabled'] = False
            sleep(0.5)
            button['enabled'] = True

    def buttonManualPellet_callback(self, button):
        button['button_pressed'] = True
        sleep(0.5)
        button['button_pressed'] = False

    def buttonMilkTrial_callback(self, button):
        if False:
            pass
        else:
            button['enabled'] = False
            sleep(0.5)
            button['enabled'] = True

    def get(self, name, FEEDER_ID=None):
        button = self.list[self.names.index(name)]
        if isinstance(button, list): 
            if FEEDER_ID is None:
                raise ValueError('get needs FEEDER_ID for this button_name.')
            else:
                for subbutton in button:
                    if 'text' in subbutton.keys() and subbutton['text'] == FEEDER_ID:
                        button = subbutton
                        break

        return button

    def _inactivate_device(self, device_type, ID):
        """
        This method is called whenever a device is inactivated
        """
        if device_type == 'pellet':
            # Inactivate pellet reward button
            self.get('buttonReleasePellet', ID)['enabled'] = False
        elif device_type == 'milk':
            # Inactivate milk reward button
            self.get('buttonReleaseMilk', ID)['enabled'] = False
            # Inactivate milk trial button
            self.get('buttonMilkTrial', ID)['enabled'] = False
        else:
            raise ValueError('Unknown device_type.')

    def _define(self):
        """
        Add or remove buttons in this function
        Create new callbacks for new button if necessary
        Callbacks are called at button click in a new thread with button dictionary as argument
        Note default settings applied in Buttons._addDefaultProperties method
        """
        self.list = []
        self.names = []
        # Game On/Off button
        buttonGameOnOff = {'callback': self.buttonGameOnOff_callback, 
                           'text': 'Game Off', 
                           'toggled': {'text': 'Game On', 
                                       'color': (0, 128, 0)}}
        self.list.append(buttonGameOnOff)
        self.names.append('buttonGameOnOff')
        # Button to mark manually released pellet
        buttonManualPellet = {'callback': self.buttonManualPellet_callback, 
                              'text': 'Manual Pellet', 
                              'toggled': {'text': 'Manual Pellet', 
                                          'color': (0, 128, 0)}}
        self.list.append(buttonManualPellet)
        self.names.append('buttonManualPellet')
        # Button to release pellet
        if not (self._PelletRewardDevices is None):
            buttonReleasePellet = []
            buttonReleasePellet.append({'text': 'Release Pellet'})
            for ID in self._PelletRewardDevices.IDs_active:
                nFeederButton = {'callback': self.buttonReleaseReward_callback, 
                                 'callargs': ['pellet', ID], 
                                 'text': ID, 
                                 'enabled': True, 
                                 'toggled': {'text': ID, 
                                             'color': (0, 128, 0)}}
                buttonReleasePellet.append(nFeederButton)
            self.list.append(buttonReleasePellet)
            self.names.append('buttonReleasePellet')
        # Button to start milkTrial
        if not (self._MilkRewardDevices is None):
            buttonMilkTrial = []
            buttonMilkTrial.append({'text': 'Milk Trial'})
            for ID in self._MilkRewardDevices.IDs_active:
                nFeederButton = {'callback': self.buttonMilkTrial_callback, 
                                 'callargs': [ID], 
                                 'text': ID, 
                                 'enabled': True, 
                                 'toggled': {'text': ID, 
                                             'color': (0, 128, 0)}}
                buttonMilkTrial.append(nFeederButton)
            self.list.append(buttonMilkTrial)
            self.names.append('buttonMilkTrial')
        # Button to release milk
        if not (self._MilkRewardDevices is None):
            buttonReleaseMilk = []
            buttonReleaseMilk.append({'text': 'Deposit Milk'})
            for ID in self._MilkRewardDevices.IDs_active:
                nFeederButton = {'callback': self.buttonReleaseReward_callback, 
                                 'callargs': ['milk', ID], 
                                 'text': ID, 
                                 'enabled': True, 
                                 'toggled': {'text': ID, 
                                             'color': (0, 128, 0)}}
                buttonReleaseMilk.append(nFeederButton)
            self.list.append(buttonReleaseMilk)
            self.names.append('buttonReleaseMilk')
        # Add default properties to all buttons
        self.list = Buttons._addDefaultProperties(self.list)

    @staticmethod
    def _addDefaultProperties(buttons):
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

        return buttons

    @staticmethod
    def _pre_render(buttons, position_on_screen, renderText):
        # Compute button locations
        xpos = position_on_screen['left']
        xlen = position_on_screen['right'] - position_on_screen['left']
        ypos = np.linspace(position_on_screen['top'], position_on_screen['bottom'], 2 * len(buttons))
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
                buttons[i]['textRendered'] = renderText(button['text'])
                if 'toggled' in button.keys():
                    buttons[i]['toggled']['textRendered'] = renderText(button['toggled']['text'])
            elif isinstance(button, list):
                for j, subbutton in enumerate(button):
                    buttons[i][j]['textRendered'] = renderText(subbutton['text'])
                    if 'toggled' in subbutton.keys():
                        buttons[i][j]['toggled']['textRendered'] = renderText(subbutton['toggled']['text'])

        return buttons

    @staticmethod
    def _draw_button(screen, button):
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
        pygame.draw.rect(screen, color, button['Position'], 0)
        screen.blit(textRendered, button['Position'][:2])

    def render(self, screen):
        # Draw all self.list here
        for i, button in enumerate(self.list):
            if isinstance(button, dict):
                Buttons._draw_button(screen, button)
            elif isinstance(button, list):
                # Display name for button group
                screen.blit(button[0]['textRendered'], button[0]['Position'][:2])
                for j, subbutton in enumerate(button[1:]):
                    Buttons._draw_button(screen, subbutton)

    def click(self, pos):
        """
        Checks if pos matches collidepoint of any buttons in self.list
        If there is a match, callback function is called for that button, 
        if it is enabled (['enabled'] = True).
        """
        for button in self.list:
            if isinstance(button, dict):
                if button['Rect'].collidepoint(pos) and button['enabled']:
                    Thread(target=button['callback'], args=(button,)).start()
            elif isinstance(button, list):
                for subbutton in button[1:]:
                    if subbutton['Rect'].collidepoint(pos) and subbutton['enabled']:
                        Thread(target=subbutton['callback'], args=(subbutton,)).start()


class InfoDisplay(object):

    def __init__(self, renderText, position, game_counters):
        self._renderText = renderText
        self._position = position
        self._game_counters = game_counters
        self._game_state = ''
        self._game_state_Lock = Lock()

    def update_game_state(self, game_state):
        with self._game_state_Lock:
            self._game_state = game_state

    def get_game_state(self):
        with self._game_state_Lock:
            return self._game_state

    def render(self, screen):
        # Compute window borders
        xborder = (self._position['left'], self._position['right'])
        yborder = (self._position['top'], self._position['bottom'])
        # Compute text spacing
        textSpacing = 3
        textSpace = 10 + textSpacing
        # Display game state
        game_state_pos = (xborder[0], yborder[0])
        screen.blit(self._renderText('Game State:'), game_state_pos)
        game_state_current_pos = (xborder[0], yborder[0] + textSpace)
        screen.blit(self._renderText(str(self.get_game_state().upper())), game_state_current_pos)
        # Split rest of screen in 5 columns
        title_topedge = game_state_pos[1] + 3 * textSpace
        topedge = game_state_pos[1] + 4 * textSpace
        columnedges = np.linspace(xborder[0], xborder[1], 10)
        columnedges = columnedges[::2]
        # Display Pellet Game info
        if 'Pellets' in self._game_counters.keys():
            # Display pellet feeder IDs
            screen.blit(self._renderText('ID'), (columnedges[0], title_topedge))
            for i, ID in enumerate(self._game_counters['Pellets']['ID']):
                screen.blit(self._renderText(ID), (columnedges[0], topedge + i * textSpace))
            # Display pellet counts
            screen.blit(self._renderText('pellets'), (columnedges[1], title_topedge))
            for i, count in enumerate(self._game_counters['Pellets']['count']):
                screen.blit(self._renderText(str(count)), (columnedges[1], topedge + i * textSpace))
        # Display Milk Game info
        if 'Milk trials' in self._game_counters.keys():
            # Display milk feeder IDs
            screen.blit(self._renderText('ID'), (columnedges[2], title_topedge))
            for i, ID in enumerate(self._game_counters['Milk trials']['ID']):
                screen.blit(self._renderText(ID), (columnedges[2], topedge + i * textSpace))
            # Display milk trial counts
            screen.blit(self._renderText('milk trials'), (columnedges[3], title_topedge))
            for i, count in enumerate(self._game_counters['Milk trials']['count']):
                screen.blit(self._renderText(str(count)), (columnedges[3], topedge + i * textSpace))
            # Display successful milk trial counts
            screen.blit(self._renderText('successful'), (columnedges[4], title_topedge))
            for i, count in enumerate(self._game_counters['Successful']['count']):
                screen.blit(self._renderText(str(count)), (columnedges[4], topedge + i * textSpace))


class TextRenderer(object):

    def __init__(self):
        self._font = pygame.font.SysFont('Arial', 10)
        self._color = (255, 255, 255)

    def render(self, text):
        return self._font.render(text, True, self._color)


class Display(object):

    def __init__(self, size, variable_display, button_display, info_display, update_rate=20):
        """
        size             - tuple - (width, height) of screen as int
        variable_display - VariableDisplay instance render method
        button_display   - Buttons instance render method
        info_display     - InfoDisplay instance render method
        """
        self._render_methods = []
        self._render_methods.append(variable_display)
        self._render_methods.append(button_display)
        self._render_methods.append(info_display)
        self._screen = pygame.display.set_mode(size)
        self._update_rate = update_rate

    def _update(self):
        self._screen.fill((0, 0, 0))
        for renderer in self._render_methods:
            renderer(self._screen)
        pygame.display.update()

    def _process(self):
        self._continue = True
        loop_clock = pygame.time.Clock()
        while self._continue:
            self._update()
            loop_duration = loop_clock.tick(self._update_rate)
            if loop_duration > (25 + 1000 / self._update_rate):
                warnings.warn('Method logic in ' + str(type(self)) + \
                              ' runs slower than assigned ' + \
                              'rate ' + str(self._update_rate) + ' Hz. Duration ' + \
                              str(loop_duration) + ' ms.', RuntimeWarning)

    def start(self):
        self._T__process = Thread(target=self._process)
        self._T__process.start()

    def close(self):
        if hasattr(self, '_T__process'):
            self._continue = False
            self._T__process.join()

    @staticmethod
    def Positions(n_variables):
        # Set constants for screen shape
        margins = 20
        height = 300
        variables_width = n_variables * 80
        buttons_width = 300
        text_width = 300
        # Compute screen size
        size = (6 * margins + variables_width  + buttons_width + text_width , 300)
        # Compute position of variable bars
        variables = {'top': margins, 
                     'bottom': size[1] - margins, 
                     'left': margins, 
                     'right': margins + variables_width}
        # Compute position for buttons
        buttons = {'top': margins, 
                   'bottom': size[1] - margins, 
                   'left': 3 * margins + variables_width, 
                   'right': 3 * margins + variables_width + buttons_width}
        # Compute position for counters
        counters = {'top': margins, 
                    'bottom': size[1] - margins, 
                    'left': 5 * margins + variables_width + buttons_width, 
                    'right': 5 * margins + variables_width + buttons_width + text_width}

        return size, variables, buttons, counters


class UserEventHandler(object):

    def __init__(self, button_click_method, KillSwitch, response_rate=60):
        self._button_click_method = button_click_method
        self._KillSwitch = KillSwitch
        self._response_rate = response_rate

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._KillSwitch()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Process click with Buttons class click detector
                    Thread(target=self._button_click_method, args=(event.pos,)).start()

    def _process(self):
        self._continue = True
        loop_clock = pygame.time.Clock()
        while self._continue:
            self._check_events()
            loop_duration = loop_clock.tick(self._response_rate)
            if loop_duration > (25 + 1000 / self._response_rate):
                warnings.warn('Method logic in ' + str(type(self)) + \
                              ' runs slower than assigned ' + \
                              'rate ' + str(self._response_rate) + ' Hz. Duration ' + \
                              str(self._response_rate) + ' ms.', RuntimeWarning)

    def start(self):
        self._T__process = Thread(target=self._process)
        self._T__process.start()

    def close(self):
        if hasattr(self, '_T__process'):
            self._continue = False
            self._T__process.join()


class Core(object):

    def __init__(self, TaskSettings, open_ephys_message_pipe, processed_position_list,
                 processed_position_update_interval, position_histogram_dict):
        """Initializes all possible devices to prepare for start command.

        :param TaskSettings:
        :param multiprocessing.connection open_ephys_message_pipe:
        :param multiprocessing.managers.List processed_position_list:
        :param int processed_position_update_interval:
        :param multiprocessing.managers.Dict position_histogram_dict:
        """

        # Parse input
        FEEDERs = TaskSettings.pop('FEEDERs')
        self.TaskSettings = TaskSettings

        self.open_ephys_message_pipe = open_ephys_message_pipe
        self.processed_position_list = processed_position_list
        self.processed_position_update_interval = processed_position_update_interval
        self.position_histogram_dict = position_histogram_dict

        self._closed = False

        # Initialize pygame engine
        init_pygame()

        # Initialize Pellet Rewards
        if self.TaskSettings['games_active']['pellet']:
            print('Initializing Pellet FEEDERs...')
            self.PelletRewardDevices = RewardDevices(FEEDERs['pellet'], 'pellet', 
                                                     self.TaskSettings['Username'], 
                                                     self.TaskSettings['Password'], 
                                                     inactivation_signal=self._inactivation_signal)
            print('Initializing Pellet FEEDERs Successful')
        else:
            self.PelletRewardDevices = None

        # Initialize Milk Rewards
        if self.TaskSettings['games_active']['milk']:
            print('Initializing Milk FEEDERs ...')
            feeder_kwargs = Core.prepare_milk_feeder_kwargs(FEEDERs['milk'], TaskSettings)
            self.MilkRewardDevices = RewardDevices(FEEDERs['milk'], 'milk', 
                                                   self.TaskSettings['Username'], 
                                                   self.TaskSettings['Password'], 
                                                   feeder_kwargs=feeder_kwargs, 
                                                   inactivation_signal=self._inactivation_signal)
            print('Initializing Milk FEEDERs Successful')
        else:
            self.MilkRewardDevices = None

        # Initialize counters
        self.game_counters = {}
        if self.TaskSettings['games_active']['pellet']:
            self.game_counters['Pellets'] = {'ID': deepcopy(self.PelletRewardDevices.IDs_active), 
                                             'count': [0] * len(self.PelletRewardDevices.IDs_active)}
        if self.TaskSettings['games_active']['milk']:
            self.game_counters['Milk trials'] = {'ID': deepcopy(self.MilkRewardDevices.IDs_active), 
                                                 'count': [0] * len(self.MilkRewardDevices.IDs_active)}
            self.game_counters['Successful'] = {'ID': deepcopy(self.MilkRewardDevices.IDs_active), 
                                                'count': [0] * len(self.MilkRewardDevices.IDs_active)}

        # Initialize Pellet Game
        if self.TaskSettings['games_active']['pellet']:
            self.PelletChoice = PelletChoice(self.PelletRewardDevices, position_histogram_dict,
                                             self.TaskSettings['arena_size'])
        else:
            self.PelletChoice = None

        # Initialize Milk Game
        if self.TaskSettings['games_active']['milk']:
            self.MilkTrialSignals = MilkTrialSignals(self.TaskSettings, 
                                                     self.MilkRewardDevices.actuator_method_call, 
                                                     self.send_message_to_open_ephys, FEEDERs['milk'])
            self.MilkGoal = MilkGoal(self.MilkRewardDevices.IDs_active, 
                                     next_feeder_method=self.TaskSettings['MilkGoalNextFeederMethod'], 
                                     repetitions=self.TaskSettings['MilkGoalRepetition'])
        else:
            self.MilkTrialSignals = None
            self.MilkGoal = None

        # Prepare chewing counter
        if self.TaskSettings['games_active']['pellet']:
            self.ChewingTracker = ChewingTracker(self.TaskSettings['Chewing_TTLchan'])
            self.chewing_tracker_update_method_thread = Thread(target=self.chewing_tracker_update_method)
            self.chewing_tracker_update_method_thread.start()
        else:
            self.ChewingTracker = None

    @property
    def closed(self):
        return self._closed

    def send_message_to_open_ephys(self, message):
        self.open_ephys_message_pipe.send(message)

    def chewing_tracker_update_method(self):
        while not self.closed:
            if self.open_ephys_message_pipe.poll(0.1):
                message = self.open_ephys_message_pipe.recv()
                self.ChewingTracker.check_for_chewing_message(message)

    @staticmethod
    def prepare_milk_feeder_kwargs(FEEDERs, TaskSettings):
        """
        Prepares specific milk feeder kwargs for use in Reward Devices class

        FEEDERs      - dict - key (ID) and value (feeder specific parameters)
        TaskSettings - dict - see below for used keys
        """
        # Grab relavant values from TaskSettings
        AudioSignalMode = TaskSettings['AudioSignalMode']
        negativeAudioSignal = TaskSettings['NegativeAudioSignal']
        lightSignalIntensity = TaskSettings['lightSignalIntensity']
        lightSignalPins = map(int, TaskSettings['lightSignalPins'].split(','))
        # Create settings for each feeder ID
        feeder_kwargs = {}
        for ID in FEEDERs.keys():
            if AudioSignalMode == 'ambient':
                trialAudioSignal = None
            elif AudioSignalMode == 'localised':
                trialAudioSignal = (FEEDERs[ID]['SignalHz'], 
                                    FEEDERs[ID]['SignalHzWidth'], 
                                    FEEDERs[ID]['ModulHz'])
            feeder_kwargs[ID] = {'trialAudioSignal': trialAudioSignal, 
                                 'negativeAudioSignal': negativeAudioSignal, 
                                 'lightSignalIntensity': lightSignalIntensity, 
                                 'lightSignalPins': lightSignalPins}

        return feeder_kwargs

    def _final_initialization(self):
        """
        Final initialization that depends on task inputs to be active (e.g. tracking).
        """
        # Initialize text rendering for faster display
        self.TextRenderer = TextRenderer()
        # Initialize variables first to find out how many are in use
        self.Variables = Core.init_Variables(self.TaskSettings, self.processed_position_list,
                                             self.processed_position_update_interval, self.ChewingTracker,
                                             self.MilkRewardDevices, self.MilkGoal)
        # Get positions of all display elements based on number of variables
        display_size, variable_pos, buttons_pos, info_pos = Display.Positions(len(self.Variables.full_list()))
        # Initialize variable display
        self.VariableDisplay = VariableDisplay(variable_pos, self.TextRenderer.render, 
                                               self.Variables)
        # Initialize buttons along with button rendering
        self.Buttons = Buttons(buttons_pos, self.TextRenderer.render, 
                               self.PelletRewardDevices, self.MilkRewardDevices)
        # Initialize info display
        self.InfoDisplay = InfoDisplay(self.TextRenderer.render, info_pos, self.game_counters)
        # Initialize display
        self.Display = Display(display_size, self.VariableDisplay.render, 
                               self.Buttons.render, self.InfoDisplay.render)
        # Initialize Game State Process
        self.GameStateOperator = GameStateOperator(self.TaskSettings, self.send_message_to_open_ephys,
                                                   self.Variables, self.game_counters, 
                                                   self.InfoDisplay.update_game_state, 
                                                   self.PelletRewardDevices, self.PelletChoice, 
                                                   self.MilkRewardDevices, self.MilkTrialSignals, 
                                                   self.MilkGoal)
        # Initialize user event detection and handling
        self.UserEventHandler = UserEventHandler(self.Buttons.click, self.KillSwitch)

    @staticmethod
    def init_Variables(TaskSettings, processed_position_list, processed_position_update_interval,
                       ChewingTracker=None, MilkRewardDevices=None, MilkGoal=None):
        # Pre-compute variables
        TaskSettings['one_second_steps'] = \
            int(np.round(1 / processed_position_update_interval))
        TaskSettings['max_distance_in_arena'] = \
            int(round(np.hypot(TaskSettings['arena_size'][0], TaskSettings['arena_size'][1])))
        TaskSettings['distance_steps'] = \
            int(np.round(TaskSettings['LastTravelTime'] * TaskSettings['one_second_steps']))
        TaskSettings['angular_distance_steps'] = \
            int(np.round(TaskSettings['MilkTaskGoalAngularDistanceTime'] * TaskSettings['one_second_steps']))
        # Start Variable class
        return Variables(TaskSettings, processed_position_list, ChewingTracker, MilkRewardDevices, MilkGoal)

    def _inactivation_signal(self, device_type, ID):
        """
        This method is passed to reward devices and invoked each time a device is inactivated.
        """
        self.Buttons._inactivate_device(device_type, ID)
        if device_type == 'pellet':
            self.PelletChoice._ensure_nearest_feeder_map_valid(self.PelletRewardDevices.IDs_active)
        elif device_type == 'milk':
            self.MilkGoal.re_init(self.MilkRewardDevices.IDs_active)
        else:
            raise ValueError('Unknown device_type: ' + str(device_type))

    def run(self):
        # Perform final initialization steps
        self._final_initialization()
        # Start Display
        self.Display.start()
        # Start Game State Process
        self.GameStateOperator.start()
        # Start Display event detection
        self.UserEventHandler.start()

    def KillSwitch(self):
        """
        Allows shutting down all downstream and upstream processes.
        Can be called by child proceses.
        """
        self.stop()

    def stop(self):
        print('Closing Task processes ...')

        # Stop Display event detection
        if hasattr(self, 'UserEventHandler'):
            self.UserEventHandler.close()

        # Stop Game State Process
        if hasattr(self, 'GameStateOperator'):
            self.GameStateOperator.close()

        # Stop updating display
        if hasattr(self, 'Display'):
            self.Display.close()

        # Stop updating ingame variables
        if hasattr(self, 'Variables'):
            self.Variables.close()

        print('Closing Task processes successful')

        # Close FEEDER connections
        if hasattr(self, 'PelletRewardDevices'):
            if hasattr(self.PelletRewardDevices, 'close'):
                print('Closing Pellet FEEDER connections...')
                self.PelletRewardDevices.close()
                print('Closing Pellet FEEDER connections successful.')
        if hasattr(self, 'MilkRewardDevices'):
            if hasattr(self.MilkRewardDevices, 'close'):
                print('Closing Milk FEEDER connections...')
                self.MilkRewardDevices.close()
                print('Closing Milk FEEDER connections successful.')

        # Close pygame engine
        close_pygame()

        self._closed = True
