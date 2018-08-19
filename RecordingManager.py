### This program has a GUI and handles the data flows synchrony of
### tracking with the electrophysiological recording of OpenEphysGUI.

### By Sander Tanni, May 2017, UCL

from PyQt4 import QtGui
from PyQt4.QtCore import QThread, pyqtSignal
import RecordingManagerDesign
import sys
from datetime import datetime
import os
import RPiInterface as rpiI
import cPickle as pickle
import subprocess
from shutil import copyfile, copytree, rmtree, move
import csv
import time
from CumulativePosPlot import PosPlot
from ZMQcomms import SendOpenEphysSingleMessage, SubscribeToOpenEphys
import HelperFunctions as hfunct
import threading
from copy import deepcopy
from scipy.io import savemat
from tempfile import mkdtemp
import numpy as np
import NWBio

def show_message(message, message_more=None):
    # This function is used to display a message in a separate window
    msg = QtGui.QMessageBox()
    msg.setIcon(QtGui.QMessageBox.Information)
    msg.setText(message)
    if message_more:
        msg.setInformativeText(message_more)
    msg.setWindowTitle('Message')
    msg.setStandardButtons(QtGui.QMessageBox.Ok)
    msg.exec_()

def list_window(items):
    '''
    Creates a new window with a list of string items to select from.
    Double clicking returns the string on the item selected

    Input:  list of strings
    Output: selected list element
    '''
    listWindow = QtGui.QDialog()
    listWindow.setWindowTitle('Choose option:')
    listWindow.resize(200, 200)
    itemsList = QtGui.QListWidget()
    itemsList.addItems(items)
    itemsList.itemDoubleClicked.connect(listWindow.accept)
    scroll = QtGui.QScrollArea()
    scroll.setWidget(itemsList)
    scroll.setWidgetResizable(True)
    vbox = QtGui.QVBoxLayout()
    vbox.addWidget(scroll)
    listWindow.setLayout(vbox)
    listWindow.show()
    if listWindow.exec_():
        selected_item = items[itemsList.currentRow()]
    else:
        selected_item = None

    return selected_item


def findLatestTimeFolder(path):
    # Get subdirectory names and find folder with latest date time as folder name
    dir_items = os.listdir(path)
    dir_times = []
    item_nrs = []
    for n_item in range(len(dir_items)):
        try: # Use only items for which the name converst to correct date time format
            item_time = datetime.strptime(dir_items[n_item], '%Y-%m-%d_%H-%M-%S')
            item_nrs.append(n_item)
            dir_times.append(item_time)
        except:
            tmp = []
    # Find the latest time based on folder names
    latest_time = max(dir_times)
    latest_folder = dir_items[item_nrs[dir_times.index(latest_time)]]

    return latest_folder

def get_recording_file_path(recording_folder_root):
    recording_file_name = 'experiment_1.nwb'
    if os.path.isdir(recording_folder_root):
        # Get the name of the folder with latest date time as name
        latest_folder = findLatestTimeFolder(recording_folder_root)
        foldertime = datetime.strptime(latest_folder, '%Y-%m-%d_%H-%M-%S')
        # If the folder time is less than 5 seconds old
        if (datetime.now() - foldertime).total_seconds() < 5:
            recording_file = os.path.join(recording_folder_root, latest_folder, recording_file_name)
        else: # Display error in text box
            recording_file = False
    else: # Display error in text box
        recording_file = False

    return recording_file

def check_if_nwb_recording(dataFilePath):
    if os.path.isfile(dataFilePath):
        # Check for file size at intervals to see if it is changing in size. Stop checking if constant.
        lastsize = os.stat(dataFilePath).st_size
        time.sleep(0.1)
        currsize = os.stat(dataFilePath).st_size
        file_recording = lastsize != currsize
    else:
        file_recording = False

    return file_recording

def update_specific_camera(TrackingSettings, n_rpi, tmpFolder):
    # Use up to date scripts
    copytree('RaspberryPi', tmpFolder)
    # Store TrackingSettings
    with open(os.path.join(tmpFolder, 'TrackingSettings.p'),'wb') as file:
        pickle.dump(TrackingSettings, file)
    # Specify RPi number
    with open(os.path.join(tmpFolder, 'RPiNumber'), 'wb') as file:
        file.write(str(n_rpi))
    # Store correct calibration data
    if str(n_rpi) in TrackingSettings['calibrationData'].keys():
        calibrationFile = os.path.join(tmpFolder, 'calibrationData.p')
        with open(calibrationFile, 'wb') as file:
            pickle.dump(TrackingSettings['calibrationData'][str(n_rpi)], file)
    # Recreate this folder on the tracking RPi
    tmp_path = os.path.join(tmpFolder, '')
    callstr = 'rsync -qavrP -e "ssh -l server" ' + tmp_path + ' ' + \
              TrackingSettings['username'] + '@' + TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
              TrackingSettings['tracking_folder'] + ' --delete'
    _ = os.system(callstr)
    # Copy over ZMQcomms.py
    src_file = 'ZMQcomms.py'
    dst_file = TrackingSettings['username'] + '@' + TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
               TrackingSettings['tracking_folder']
    callstr = 'scp -q ' + src_file + ' ' + dst_file
    _ = os.system(callstr)

def update_tracking_camera_files(TrackingSettings):
    # Updates settings and scripts on all tracking cameras in use (based on settings)
    # Create temporary working directory
    RPiTempFolder = mkdtemp('RPiTempFolder')
    T_updateRPi = []
    for n_rpi in TrackingSettings['use_RPi_nrs']:
        RPiTempSubFolder = os.path.join(RPiTempFolder, str(n_rpi))
        T = threading.Thread(target=update_specific_camera, args=[TrackingSettings, n_rpi, RPiTempSubFolder])
        T.start()
        T_updateRPi.append(T)
    for T in T_updateRPi:
        T.join()
    rmtree(RPiTempFolder)

def get_RPi_frame_timestamps_in_OEtimes(OEtimes, RPtimes, RPFtimes):
    # Find closest RPi GlobalClock timestamp to each RPi frame timestamp
    RPtimes_idx = []
    for RPFtime in RPFtimes:
        RPtimes_idx.append(np.argmin(np.abs(RPtimes - RPFtime)))
    # Compute difference from the GlobalClock timestamps for each frame
    RPtimes_full = RPtimes[RPtimes_idx]
    RPF_RP_times_delta = RPFtimes - RPtimes_full
    # Convert this difference to seconds as Open Ephys timestamps
    RPF_RP_times_delta_in_seconds = RPF_RP_times_delta / float(10 ** 6)
    # Use frame and GlobalClock timestamp diffence to estimate OpenEphys timepoints
    OEtimes_full = OEtimes[RPtimes_idx]
    RPFinOEtimes = OEtimes_full + RPF_RP_times_delta_in_seconds

    return RPFinOEtimes

def process_single_tracking_data(n_rpi, TrackingLog_File, TTLpulse_CameraTime_File, data_events):
    '''
    This function processes TrackingLog*.csv files.
    Offset between actual frame time and TTL pulses are corrected.
    If TrackingLog*.csv has more datapoints than TTL pulses recorded, the TrackingLog datapoints from the end are dropped.
    '''
    GlobalClock_TTL_Channel = 1
    RPiTime2Sec = 10 ** 6 # This values is used to convert RPi times to seconds
    # Read position data from this camera
    trackingData = np.genfromtxt(TrackingLog_File, delimiter=',')
    # Read GlobalClock TTL pulse times from this camera
    RPi_GC_times = np.genfromtxt(TTLpulse_CameraTime_File, delimiter=',')
    # Read OpenEphys timestamps of GlobalClock TTL pulses (in seconds)
    OE_GC_times = data_events['timestamps'][np.array(data_events['eventID']) == GlobalClock_TTL_Channel]
    # Notify when data incorrect length
    if OE_GC_times.size > RPi_GC_times.size:
        OE_GC_times = OE_GC_times[:RPi_GC_times.size]
        print('[ Warning ] OpenEphys recorded more GlobalClock TTL pulses than RPi ' + str(n_rpi) + '.\n' + 
              'Dumping extra timestamps from the end.')
    elif OE_GC_times.size < RPi_GC_times.size:
        RPi_GC_times = RPi_GC_times[:OE_GC_times.size]
        print('[ Warning ] RPi ' + str(n_rpi) + ' recorded more GlobalClock TTL pulses than Open Ephys.\n' + 
              'Dumping extra timestamps from the end.')
    # Get RPi frame timestamps in Open Ephys time
    times = get_RPi_frame_timestamps_in_OEtimes(OE_GC_times, RPi_GC_times, trackingData[:,1])
    # Combine corrected timestamps with position data
    TrackingData = np.concatenate((np.expand_dims(times, axis=1), trackingData[:,2:]), axis=1).astype(np.float64)

    return TrackingData

def retrieve_specific_camera_tracking_data(n_rpi, TrackingSettings, RPiTempFolder, data_events, TrackingData, TrackingDataLock):
    src_file = TrackingSettings['username'] + '@' + TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
               TrackingSettings['tracking_folder'] + '/logfile.csv'
    TrackingLog_File = os.path.join(RPiTempFolder, 'TrackingLog' + str(n_rpi) + '.csv')
    callstr = 'scp -q ' + src_file + ' ' + TrackingLog_File
    _ = os.system(callstr)
    src_file = TrackingSettings['username'] + '@' + TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
               TrackingSettings['tracking_folder'] + '/TTLpulse_CameraTime.csv'
    TTLpulse_CameraTime_File = os.path.join(RPiTempFolder, 'TTLpulse_CameraTime' + str(n_rpi) + '.csv')
    callstr = 'scp -q ' + src_file + ' ' + TTLpulse_CameraTime_File
    _ = os.system(callstr)
    tmp_data = process_single_tracking_data(n_rpi, TrackingLog_File, TTLpulse_CameraTime_File, data_events)
    with TrackingDataLock:
        TrackingData[str(n_rpi)] = tmp_data
    if TrackingSettings['save_frames']:
        src = TrackingSettings['username'] + '@' + TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
              TrackingSettings['tracking_folder'] + '/frames'
        dst = os.path.join(RPiTempFolder, 'frames_' + str(n_rpi))
        callstr = 'scp -q -r ' + src + ' ' + dst
        _ = os.system(callstr)

def store_frames_to_recording_folder(TrackingSettings, rec_file_path, RPiTempFolder):
    for n_rpi in TrackingSettings['use_RPi_nrs']:
        src = os.path.join(RPiTempFolder, 'frames_' + str(n_rpi))
        dst = os.path.join(os.path.dirname(rec_file_path), 'frames_' + str(n_rpi))
        move(src, dst)

def store_tracking_data_to_recording_file(TrackingSettings, rec_file_path):
    data_events = NWBio.load_events(rec_file_path)
    # Copy all tracking data from active RPis and load to memory
    TrackingData = {'ColumnLabels': ['timestamps', 'X1', 'Y1', 'X2', 'Y2', 'Luminance_1', 'Luminance_2']}
    RPiTempFolder = mkdtemp('RPiTempFolder')
    T_retrievePosLogsRPi = []
    TrackingDataLock = threading.Lock()
    T_args = [TrackingSettings, RPiTempFolder, data_events, TrackingData, TrackingDataLock]
    for n_rpi in TrackingSettings['use_RPi_nrs']:
        T = threading.Thread(target=retrieve_specific_camera_tracking_data, args=[n_rpi] + T_args)
        T.start()
        T_retrievePosLogsRPi.append(T)
    for T in T_retrievePosLogsRPi:
        T.join()
    if TrackingSettings['save_frames']:
        store_frames_to_recording_folder(TrackingSettings, rec_file_path, RPiTempFolder)
    rmtree(RPiTempFolder)
    # Save position data from all sources to recording file
    NWBio.save_tracking_data(rec_file_path, TrackingData)

def list_general_settings_history(path):
    '''
    Assumes all files on the path are NWB files with General settings stored

    Returns:
        dictionary - just as General settings, but each key contains
                     a list of values for that key in files on the path.
        list       - full paths to all settings files
        
        All lists are sorted starting from the most recent timestamp in filename.
        None is entered if key is missing in settings file.
    '''
    dir_items = os.listdir(path)
    filetimes = []
    filenames = []
    for item in dir_items:
        if item.endswith('.settings.nwb'):
            filename = os.path.join(path, item)
            filenames.append(filename)
            filetimes.append(datetime.strptime(item[:19],'%Y-%m-%d_%H-%M-%S'))
    # Sort filenames based on filetimes
    filenames = [x for _,x in sorted(zip(filetimes, filenames))][::-1]
    # Load all general settings to memory and build a list of keys
    settings_keys = []
    settings_list = []
    for filename in filenames:
        settings = NWBio.load_settings(filename, path='/General/')
        for key in settings.keys():
            if not (key in settings_keys):
                settings_keys.append(key)
        settings_list.append(settings)
    # Create lists for all keys
    general_settings_history = {}
    for key in settings_keys:
        general_settings_history[key] = []
    for settings in settings_list:
        for key in settings_keys:
            if key in settings.keys():
                general_settings_history[key].append(settings[key])
            else:
                general_settings_history[key].append(None)

    return general_settings_history, filenames

def find_latest_matching_settings_filepath(path, settings=None):
    general_settings_history, filenames = list_general_settings_history(path)
    if settings is None or len(settings.keys()) == 0:
        # If no specific settings were requested, return the path to latest settings file
        filepath = filenames[0]
    else:
        filepaths = []
        # Loop through all settings files
        for nfile in range(len(filenames)):
            # Check if all specified settings match to the one in the file
            settings_correct = []
            for key in settings.keys():
                settings_correct.append(general_settings_history[key][nfile] == settings[key])
            if all(settings_correct):
                filepaths.append(filenames[nfile])
        if len(filepaths) > 0: # Use the most recent settings file path
            filepath = filepaths[0]
        else: # If no settings matched, return None
            filepath = None

    return filepath

def check_if_device_available_thread(address, output_list, output_pos, output_lock):
    # Function to allow multi-threading check_if_devices_available function
    output = hfunct.test_pinging_address(address)
    with output_lock:
        output_list[output_pos] = output

def check_if_devices_available(address_list, device_names=[], output='r'):
    '''
    Creates a list of devices that fail to ping back.
    address_list = list of IP address strings
    device_names = list of strings corresponding to IP. If omitted, becomes string list of range(len(address_list))
    output='r' returns the list
    output='d' displays it in a dialog window
    output='dr' does both
    '''
    if len(device_names) != len(address_list):
        device_names = [str(x + 1) for x in range(len(address_list))]
    # Check each device in a seperate thread to speed things up
    output_lock = threading.Lock()
    address_ping_outcome = [None] * len(address_list)
    T_address_pings = []
    for n_address, address in enumerate(address_list):
        T = threading.Thread(target=check_if_device_available_thread, 
                             args=[address, address_ping_outcome, n_address, output_lock])
        T.start()
        T_address_pings.append(T)
    for T in T_address_pings:
        T.join()
    # List devices that failed to ping back
    disconnected_devices = []
    for successful, address, device_name in zip(address_ping_outcome, address_list, device_names):
        if not successful:
            disconnected_devices.append({'device_name': device_name, 'address': address})
    # Provide output as requested
    if 'd' in output:
        if len(disconnected_devices) == 0:
            message = 'All ' + str(len(address_list)) + ' devices available.'
            message_more = None
        else:
            message = str(len(disconnected_devices)) + ' of ' + str(len(address_list)) + ' devices not available!'
            message_more = ''
            for device in disconnected_devices:
                message_more += device['device_name'] + ' @ ' + device['address'] + '\n'
        show_message(message, message_more=message_more)
    if 'r' in output:
        return disconnected_devices


class QThread_with_completion_callback(QThread):
    finishedSignal = pyqtSignal()
    '''
    Allows calling funcitons in separate threads with a callback function executed at completion.

    Note! Make sure QThread_with_completion_callback instance does not go out of scope during execution.
    '''
    def __init__(self, callback_function, function, *args, **kwargs):
        '''
        The callback_function is called when function has finished.
        The function is called with any input arguments that follow it.
        '''
        super(QThread_with_completion_callback, self).__init__()
        self.finishedSignal.connect(callback_function)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start()

    def run(self):
        self.function(*self.args, **self.kwargs)
        self.finishedSignal.emit()

class QThread_thread(QThread):
    '''
    Allows initating a function in a QThread.
    '''
    def __init__(self, function, *args, **kwargs):
        '''
        The function is called with any input arguments that follow it.
        '''
        super(QThread_thread, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.start()

    def run(self):
        self.function(*self.args, **self.kwargs)


class RecordingManager(QtGui.QMainWindow, RecordingManagerDesign.Ui_MainWindow):

    def __init__(self, parent=None):
        super(RecordingManager, self).__init__(parent=parent)
        self.setupUi(self)
        # Set GUI environment
        RecordingDataPath = os.path.expanduser('~') + '/RecordingData'
        self.pt_root_folder.setPlainText(RecordingDataPath)
        self.RecordingManagerSettingsFolder = 'RecordingManagerSettings'
        # Create empty settings dictonary
        self.Settings = {}
        # Set GUI interaction connections
        self.pb_experimenter.clicked.connect(lambda:self.show_options('experimenter', self.pt_experimenter))
        self.pb_animal.clicked.connect(lambda:self.show_options('animal', self.pt_animal))
        self.pb_experiment_id.clicked.connect(lambda:self.show_options('experiment_id', self.pt_experiment_id))
        self.pb_load_last.clicked.connect(lambda:self.load_last_settings())
        self.pb_load.clicked.connect(lambda:self.load_settings())
        self.pb_save.clicked.connect(lambda:self.save_settings())
        self.pb_root_folder.clicked.connect(lambda:self.root_folder_browse())
        self.pb_cam_set.clicked.connect(lambda:self.camera_settings())
        self.pb_task_set.clicked.connect(lambda:self.task_settings())
        self.pb_test_devices.clicked.connect(lambda:self.test_devices())
        self.pb_init_devices.clicked.connect(lambda:self.pb_init_devices_callback())
        self.pb_start_rec.clicked.connect(lambda:self.pb_start_rec_callback())
        self.pb_stop_rec.clicked.connect(lambda:self.pb_stop_rec_callback())
        self.pb_process_data.clicked.connect(lambda:self.pb_process_data_callback())
        self.pb_sync_server.clicked.connect(lambda:self.sync_server())
        self.pb_open_rec_folder.clicked.connect(lambda:self.open_rec_folder())
        # Store original stylesheets for later use
        self.original_stylesheets = {'pb_init_devices': self.pb_init_devices.styleSheet(), 
                                     'pb_start_rec': self.pb_start_rec.styleSheet(), 
                                     'pb_stop_rec': self.pb_stop_rec.styleSheet()}

    def openFolderDialog(self, caption='Select folder'):
        # Pops up a GUI to select a folder
        dialog = QtGui.QFileDialog(self, caption=caption, directory=str(self.pt_root_folder.toPlainText()))
        dialog.setFileMode(QtGui.QFileDialog.Directory)
        dialog.setOption(QtGui.QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            # Get paths of selected folder
            tmp = dialog.selectedFiles()
            selected_folder = str(tmp[0])
        else:
            selected_folder = 'No folder selected'

        return selected_folder

    def get_RecordingGUI_settings(self):
        # Get channel mapping info
        channel_map = {}
        if len(str(self.pt_chan_map_1.toPlainText())) > 0:
            channel_map_1 = str(self.pt_chan_map_1.toPlainText())
            chanString = str(self.pt_chan_map_1_chans.toPlainText())
            chan_from = int(chanString[:chanString.find('-')])
            chan_to = int(chanString[chanString.find('-') + 1:])
            chan_list = np.arange(chan_from - 1, chan_to, dtype=np.int64)
            channel_map[channel_map_1] = {'string': chanString, 'list': chan_list}
            if len(str(self.pt_chan_map_2.toPlainText())) > 0:
                channel_map_2 = str(self.pt_chan_map_2.toPlainText())
                chanString = str(self.pt_chan_map_2_chans.toPlainText())
                chan_from = int(chanString[:chanString.find('-')])
                chan_to = int(chanString[chanString.find('-') + 1:])
                chan_list = np.arange(chan_from - 1, chan_to, dtype=np.int64)
                channel_map[channel_map_2] = {'string': chanString, 'list': chan_list}
        # Grabs all options in the Recording Manager and puts them into a dictionary
        RecGUI_Settings = {'experimenter': str(self.pt_experimenter.toPlainText()), 
                           'root_folder': str(self.pt_root_folder.toPlainText()), 
                           'animal': str(self.pt_animal.toPlainText()), 
                           'experiment_id': str(self.pt_experiment_id.toPlainText()), 
                           'arena_size': np.array([str(self.pt_arena_size_x.toPlainText()), str(self.pt_arena_size_y.toPlainText())], dtype=np.float64), 
                           'badChan': str(self.pt_badChan.toPlainText()), 
                           'rec_file_path': str(self.pt_rec_file.toPlainText()), 
                           'Tracking': np.array(self.rb_tracking_yes.isChecked()), 
                           'channel_map': channel_map, 
                           'TaskActive': np.array(self.rb_task_yes.isChecked())}

        return RecGUI_Settings

    def show_options(self, setting_name, setting_textbox):
        RecGUI_Settings = self.get_RecordingGUI_settings()
        RecordingManagerSettingsPath = os.path.join(RecGUI_Settings['root_folder'], 
                                                    self.RecordingManagerSettingsFolder)
        general_settings_history, _ = list_general_settings_history(RecordingManagerSettingsPath)
        items = sorted(list(set(general_settings_history[setting_name])))
        selected_item = list_window(items)
        if not (selected_item is None):
            setting_textbox.setPlainText(selected_item)

    def load_settings(self, filename=None):
        if filename is None: # Get user to select settings to load if not given
            filename = hfunct.openSingleFileDialog('load', suffix='nwb', caption='Select file to load')
        self.Settings = NWBio.load_settings(filename)
        # Put Recording Manager General settings into GUI
        RecGUI_Settings = self.Settings['General']
        self.pt_experimenter.setPlainText(RecGUI_Settings['experimenter'])
        self.pt_root_folder.setPlainText(RecGUI_Settings['root_folder'])
        self.pt_animal.setPlainText(RecGUI_Settings['animal'])
        self.pt_experiment_id.setPlainText(RecGUI_Settings['experiment_id'])
        self.pt_arena_size_x.setPlainText(str(RecGUI_Settings['arena_size'][0]))
        self.pt_arena_size_y.setPlainText(str(RecGUI_Settings['arena_size'][1]))
        self.pt_badChan.setPlainText(RecGUI_Settings['badChan'])
        self.pt_rec_file.setPlainText(RecGUI_Settings['rec_file_path'])
        if RecGUI_Settings['Tracking']:
            self.rb_tracking_yes.setChecked(True)
        else:
            self.rb_tracking_no.setChecked(True)
        if RecGUI_Settings['TaskActive']:
            self.rb_task_yes.setChecked(True)
        else:
            self.rb_task_no.setChecked(True)
        channel_locations = RecGUI_Settings['channel_map'].keys()
        if len(channel_locations) > 0:
            self.pt_chan_map_1.setPlainText(channel_locations[0])
            self.pt_chan_map_1_chans.setPlainText(RecGUI_Settings['channel_map'][channel_locations[0]]['string'])
            if len(RecGUI_Settings['channel_map']) > 1:
                self.pt_chan_map_2.setPlainText(channel_locations[1])
                self.pt_chan_map_2_chans.setPlainText(RecGUI_Settings['channel_map'][channel_locations[1]]['string'])
            else:
                self.pt_chan_map_2.setPlainText('')
                self.pt_chan_map_2_chans.setPlainText('')
        else:
            self.pt_chan_map_1.setPlainText('')
            self.pt_chan_map_1_chans.setPlainText('')
            self.pt_chan_map_2.setPlainText('')
            self.pt_chan_map_2_chans.setPlainText('')

    def save_settings(self, filename=None):
        # Saves current settings into a file
        if filename is None:
            filename = hfunct.openSingleFileDialog('save', suffix='nwb', caption='Save file name and location')
        self.Settings['General'] = self.get_RecordingGUI_settings()
        self.Settings['Time'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        NWBio.save_settings(filename, self.Settings)

    def load_last_settings(self):
        RecGUI_Settings = self.get_RecordingGUI_settings()
        # Hard-code which settings to use for searching the latest settings
        settings = {}
        if len(RecGUI_Settings['animal']) > 0:
            settings['animal'] = RecGUI_Settings['animal']
        if len(RecGUI_Settings['experiment_id']) > 0:
            settings['experiment_id'] = RecGUI_Settings['experiment_id']
        # Find the latest settings file
        RecordingManagerSettingsPath = os.path.join(RecGUI_Settings['root_folder'], 
                                                    self.RecordingManagerSettingsFolder)
        filename = find_latest_matching_settings_filepath(RecordingManagerSettingsPath, settings)
        if filename is None:
            # If no matching file is found, show error message
            criteria = ''
            for key in sorted(settings.keys()):
                criteria += str(key) + ': ' + str(settings[key]) + '\n'
            show_message('No settings found for criteria:', message_more=criteria)
        else:
            self.load_settings(filename=filename)

    def root_folder_browse(self):
        # Pops up a new window to select a folder and then inserts the path to the text box
        self.pt_root_folder.setPlainText(self.openFolderDialog())

    def camera_settings(self):
        # Opens up a new window for Camera Settings
        from CameraSettingsGUI import CameraSettings
        self.CamSet = CameraSettings(parent=self)
        # Check if TrackingSettings available, if so Load them.
        if 'TrackingSettings' in self.Settings.keys():
            self.CamSet.load(deepcopy(self.Settings['TrackingSettings']))

    def task_settings(self):
        from TaskSettingsGUI import TaskSettingsGUI
        self.TaskSet = TaskSettingsGUI(parent=self)
        # If TaskSettings available, load them
        if 'TaskSettings' in self.Settings.keys():
            self.TaskSet.loadSettings(deepcopy(self.Settings['TaskSettings']))

    def test_devices(self):
        address_list = []
        device_names = []
        # Add tracking cameras to list
        if 'TrackingSettings' in self.Settings.keys():
            for Camera_ID in sorted(self.Settings['TrackingSettings']['RPiInfo'].keys(), key=int):
                if self.Settings['TrackingSettings']['RPiInfo'][Camera_ID]['active']:
                    address_list.append(self.Settings['TrackingSettings']['RPiInfo'][Camera_ID]['IP'])
                    device_names.append('Tracking Camera ' + str(Camera_ID))
        # Add feeders to list
        if 'TaskSettings' in self.Settings.keys():
            for FEEDER_type in self.Settings['TaskSettings']['FEEDERs'].keys():
                for FEEDER_ID in sorted(self.Settings['TaskSettings']['FEEDERs'][FEEDER_type].keys(), key=int):
                    if self.Settings['TaskSettings']['FEEDERs'][FEEDER_type][FEEDER_ID]['Active']:
                        address_list.append(self.Settings['TaskSettings']['FEEDERs'][FEEDER_type][FEEDER_ID]['IP'])
                        device_names.append('Feeder ' + FEEDER_type + ' '  + FEEDER_ID)
        # Test devices and display results in dialog box
        check_if_devices_available(address_list, device_names, output='d')

    def pb_init_devices_callback(self):
        # Change Init_devices button style to red and disable the button
        self.pb_init_devices.setStyleSheet('background-color: red') # Change button to red
        self.pb_init_devices.setEnabled(False)
        # Disable data handling buttons
        self.pb_process_data.setEnabled(False)
        self.pb_sync_server.setEnabled(False)
        # Do background work
        self.main_worker = QThread_with_completion_callback(self.init_devices_completed, self.init_devices)

    def init_devices_completed(self):
        # Change Init_devices button style to green and enable start_rec button
        self.pb_init_devices.setStyleSheet('background-color: green') # Change button to red
        self.pb_start_rec.setEnabled(True)

    def init_devices(self):
        # Load settings
        self.Settings['General'] = self.get_RecordingGUI_settings()
        if self.Settings['General']['Tracking']:
            if 'TrackingSettings' in self.Settings.keys():
                self.Settings['TrackingSettings']['arena_size'] = self.Settings['General']['arena_size']
            else:
                show_message('No camera settings for Tracking.')
                raise ValueError('No camera settings for Tracking.')
        if self.Settings['General']['TaskActive']:
            if 'TaskSettings' in self.Settings.keys():
                self.Settings['TaskSettings']['arena_size'] = self.Settings['General']['arena_size']
            else:
                show_message('No task settings for active task.')
                raise ValueError('No task settings for active task.')
            if not self.Settings['General']['Tracking']:
                show_message('Tracking must be active for task to work')
                raise ValueError('Tracking must be active for task to work')
        # Initialize tracking
        if self.Settings['General']['Tracking']:
            # Connect to tracking RPis
            print('Connecting to tracking RPis...')
            update_tracking_camera_files(self.Settings['TrackingSettings'])
            self.trackingControl = rpiI.TrackingControl(self.Settings['TrackingSettings'])
            print('Connecting to tracking RPis Successful')
            # Initialize onlineTrackingData class
            print('Initializing Online Tracking Data...')
            self.histogramParameters = {'margins': 10, # histogram data margins in centimeters
                                        'binSize': 2, # histogram binSize in centimeters
                                        'speedLimit': 10}# centimeters of distance in last second to be included
            self.RPIpos = rpiI.onlineTrackingData(self.Settings['TrackingSettings'], HistogramParameters=deepcopy(self.histogramParameters), SynthData=False)
            print('Initializing Online Tracking Data Successful')
            print('Connecting to GlobalClock RPi...')
            self.GlobalClockControl = rpiI.GlobalClockControl(self.Settings['TrackingSettings'])
            print('Connecting to GlobalClock RPi Successful')
        # Initialize listening to Open Ephys GUI messages
        print('Connecting to Open Ephys GUI via ZMQ...')
        self.OEmessages = SubscribeToOpenEphys(verbose=False)
        self.OEmessages.connect()
        print('Connecting to Open Ephys GUI via ZMQ Successful')
        # Initialize Task
        if self.Settings['General']['TaskActive']:
            print('Initializing Task...')
            # Put input streams in a default dictionary that can be used by task as needed
            TaskIO = {'RPIPos': self.RPIpos, 
                      'OEmessages': self.OEmessages, 
                      'MessageToOE': SendOpenEphysSingleMessage}
            TaskModule = hfunct.import_subdirectory_module('Tasks', self.Settings['TaskSettings']['name'])
            self.current_task = TaskModule.Core(deepcopy(self.Settings['TaskSettings']), TaskIO)
            print('Initializing Task Successful')  

    def pb_start_rec_callback(self):
        # Change Start button style to red and init_devices button back to default
        self.pb_start_rec.setStyleSheet('background-color: red') # Change button to red
        self.pb_init_devices.setStyleSheet(self.original_stylesheets['pb_init_devices'])
        # Disable and Enable Start and Stop buttons, respectively
        self.pb_start_rec.setEnabled(False)
        self.pb_stop_rec.setEnabled(True)
        # Do background work
        self.start_rec()

    def start_rec(self):
        # Start Open Ephys GUI recording
        print('Starting Open Ephys GUI Recording...')
        recording_folder_root = os.path.join(self.Settings['General']['root_folder'], str(self.pt_animal.toPlainText()))
        command = 'StartRecord RecDir=' + recording_folder_root + ' CreateNewDir=1'
        SendOpenEphysSingleMessage(command)
        # Make sure OpenEphys is recording
        recording_file = get_recording_file_path(recording_folder_root)
        while not recording_file:
            time.sleep(0.1)
            recording_file = get_recording_file_path(recording_folder_root)
        while not check_if_nwb_recording(recording_file):
            time.sleep(0.1)
        self.pt_rec_file.setPlainText(recording_file)
        self.Settings['General']['rec_file_path'] = recording_file
        print('Starting Open Ephys GUI Recording Successful')
        # Start the tracking scripts on all RPis
        if self.Settings['General']['Tracking']:
            print('Starting tracking RPis...')
            self.trackingControl.start()
            print('Starting tracking RPis Successful')
            print('Starting GlobalClock RPi...')
            self.GlobalClockControl.start()
            print('Starting GlobalClock RPi Successful')
        # Start cumulative plot
        if self.Settings['General']['Tracking']:
            print('Starting Position Plot...')
            self.PosPlot = PosPlot(self.Settings['TrackingSettings'], self.RPIpos, deepcopy(self.histogramParameters))
            print('Starting Position Plot Successful')
        # Start task
        if self.Settings['General']['TaskActive']:
            print('Starting Task...')
            self.current_task.run()
            print('Starting Task Successful')

    def pb_stop_rec_callback(self):
        # Disable Stop button
        self.pb_stop_rec.setEnabled(False)
        # Do background work
        self.main_worker = QThread_with_completion_callback(self.stop_rec_completed, self.stop_rec)

    def stop_rec_completed(self):
        # Change start button colors
        self.pb_start_rec.setStyleSheet(self.original_stylesheets['pb_start_rec']) # Start button to default
        # Enable other buttons
        self.pb_init_devices.setEnabled(True)
        self.pb_process_data.setEnabled(True)
        self.pb_sync_server.setEnabled(True)

    def compile_recording_data(self):
        if self.Settings['General']['Tracking']:
            # Store tracking  data in recording file
            print('Copying over tracking data to Recording File...')
            store_tracking_data_to_recording_file(self.Settings['TrackingSettings'], self.Settings['General']['rec_file_path'])
            print('Copying over tracking data to Recording File Successful')
        # Store recording settings to recording file
        self.save_settings(filename=self.Settings['General']['rec_file_path'])
        print('Settings saved to Recording File')
        # Store settings for RecordingManager history reference
        RecordingManagerSettingsPath = os.path.join(self.Settings['General']['root_folder'], 
                                                    self.RecordingManagerSettingsFolder)
        if not os.path.isdir(RecordingManagerSettingsPath):
            os.mkdir(RecordingManagerSettingsPath)
        RecordingFolderName = os.path.basename(os.path.dirname(self.Settings['General']['rec_file_path']))
        RecordingManagerSettingsFilePath = os.path.join(RecordingManagerSettingsPath, 
                                                        RecordingFolderName + '.settings.nwb')
        self.save_settings(filename=RecordingManagerSettingsFilePath)
        print('Settings saved to Recording Manager Settings Folder')

    def stop_rec(self):# Stop Task
        if self.Settings['General']['TaskActive']:
            print('Stopping Task...')
            self.current_task.stop()
            print('Stopping Task Successful')
        # Stop cumulative plot
        if self.Settings['General']['Tracking']:
            if hasattr(self, 'PosPlot'):
                print('Stopping Position Plot...')
                self.PosPlot.close()
                print('Stopping Position Plot Successful')
            # Stop updating online tracking data
            print('Closing Online Tracking Data...')
            self.RPIpos.close()
            print('Closing Online Tracking Data Successful')
            print('Closing GlobalClock Controller...')
            self.GlobalClockControl.stop()
            self.GlobalClockControl.close()
            print('Closing GlobalClock Controller Successful')
            # Stop tracking
            print('Stopping tracking RPis...')
            self.trackingControl.stop()
            print('Stopping tracking RPis Successful')
        # Stop reading Open Ephys messages
        print('Closing Open Ephys GUI ZMQ connection...')
        self.OEmessages.disconnect()
        print('Closing Open Ephys GUI ZMQ connection Successful')
        # Stop Open Ephys Recording
        while check_if_nwb_recording(self.Settings['General']['rec_file_path']):
            print('Stopping Open Ephys GUI Recording...')
            SendOpenEphysSingleMessage('StopRecord')
            time.sleep(0.1)
        print('Stopping Open Ephys GUI Recording Successful')
        self.compile_recording_data()

    def pb_process_data_callback(self):
        self.pb_init_devices.setEnabled(False)
        self.pb_process_data.setEnabled(False)
        self.pb_sync_server.setEnabled(False)
        # Do background work
        self.main_worker = QThread_with_completion_callback(self.process_data_completed, self.process_data)
    
    def process_data_completed(self):
        self.pb_init_devices.setEnabled(True)
        self.pb_process_data.setEnabled(True)
        self.pb_sync_server.setEnabled(True)

    def process_data(self):
        # Applies KlustaKwik tetrode-wise to recorded data
        import Processing
        Processing.main(self.Settings['General']['rec_file_path'])

    def sync_server(self):
        # Change button colors
        raise NotImplementedError
        # self.original_stylesheets['pb_sync_server'] = self.pb_sync_server.styleSheet() # Save button color
        # self.pb_sync_server.setStyleSheet('background-color: red') # Change button to red
        # # Extract directory tree from root folder to data to append to server folder
        # rec_folder = str(self.pt_rec_file.toPlainText())
        # root_folder = str(self.pt_root_folder.toPlainText())
        # directory_tree = rec_folder[len(rec_folder)-rec_folder[::-1].rfind(root_folder[::-1]):]
        # server_folder = self.file_server_path + directory_tree
        # # Create target folder tree on server
        # os.system('mkdir -p ' + server_folder)
        # # Copy files over to server
        # print('Copying files to server ...')
        # callstr = 'rsync -avzh ' + rec_folder + '/ ' + server_folder + '/'
        # os.system(callstr)
        # # Change button color back to default
        # self.pb_sync_server.setStyleSheet(self.original_stylesheets['pb_sync_server'])

    def open_rec_folder(self):
        # Opens recording folder with Ubuntu file browser
        subprocess.Popen(['xdg-open', os.path.dirname(self.Settings['General']['rec_file_path'])])


# The following is the default ending for a QtGui application script
def main():
    app = QtGui.QApplication(sys.argv)
    form = RecordingManager()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main()
