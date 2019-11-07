
import os
import sys
from copy import deepcopy
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from uuid import uuid4
from datetime import datetime
from time import sleep
import threading
from importlib import import_module
import multiprocessing

from openEPhys_DACQ.package_configuration import package_config
from openEPhys_DACQ import HelperFunctions as HFunc
from openEPhys_DACQ import RPiInterface
from openEPhys_DACQ import NWBio
from openEPhys_DACQ.CameraSettings import CameraSettingsApp, CameraSettingsGUI
from openEPhys_DACQ.TaskSettings import TaskSettingsApp, TaskSettingsGUI
from openEPhys_DACQ.ZMQcomms import SubscribeToOpenEphys, PublishToOpenEphys
from openEPhys_DACQ.CumulativePosPlot import PosPlot


def find_latest_time_folder(path):
    """Looks up subdirectory names and returns folder with latest date time string as folder name
    """
    dir_items = os.listdir(path)
    dir_times = []
    item_nrs = []
    for n_item in range(len(dir_items)):
        try:  # Use only items for which the name can be converted to correct date time format
            item_time = datetime.strptime(dir_items[n_item], '%Y-%m-%d_%H-%M-%S')
            item_nrs.append(n_item)
            dir_times.append(item_time)
        except:
            pass
    # Find the latest time based on folder names
    latest_time = max(dir_times)
    latest_folder = dir_items[item_nrs[dir_times.index(latest_time)]]

    return latest_folder


def get_recording_file_path(recording_folder_root):
    recording_file_name = 'experiment_1.nwb'
    if os.path.isdir(recording_folder_root):
        # Get the name of the folder with latest date time as name
        latest_folder = find_latest_time_folder(recording_folder_root)
        folder_timestamp = datetime.strptime(latest_folder, '%Y-%m-%d_%H-%M-%S')
        # If the folder time is less than 5 seconds old
        if (datetime.now() - folder_timestamp).total_seconds() < 5:
            recording_file = os.path.join(recording_folder_root, latest_folder, recording_file_name)
        else:  # Display error in text box
            recording_file = False
    else:  # Display error in text box
        recording_file = False

    return recording_file


def check_if_nwb_recording(fpath):
    if os.path.isfile(fpath):
        # Check for file size at intervals to see if it is changing in size. Stop checking if constant.
        previous_size = os.stat(fpath).st_size
        sleep(0.1)
        current_size = os.stat(fpath).st_size
        file_recording = previous_size != current_size
    else:
        file_recording = False

    return file_recording


def store_camera_data_to_recording_file(camera_settings, fpath):
    # Initialize file_manager for each camera
    file_managers = {}
    for cameraID in camera_settings['CameraSpecific'].keys():
        print(['DEBUG', HFunc.time_string(), 'Initializing Camera_RPi_file_manager for ID: ' + cameraID])
        address = camera_settings['CameraSpecific'][cameraID]['address']
        username = camera_settings['General']['username']
        file_managers[cameraID] = RPiInterface.Camera_RPi_file_manager(address, username)
        print(
            ['DEBUG', HFunc.time_string(), 'Initializing Camera_RPi_file_manager for ID: ' + cameraID + ' complete.'])
    # Retrieve data concurrently from all cameras
    thread_list = []
    for cameraID in camera_settings['CameraSpecific'].keys():
        t = threading.Thread(target=file_managers[cameraID].retrieve_timestamps_and_OnlineTrackerData)
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()
    # Combine data from cameras and store to recording file
    camera_data = {}
    for cameraID in camera_settings['CameraSpecific'].keys():
        print(['DEBUG', HFunc.time_string(), 'Retrieving camera_data for ID: ' + cameraID])
        camera_data[cameraID] = file_managers[cameraID].get_timestamps_and_OnlineTrackerData(cameraID)
        print(['DEBUG', HFunc.time_string(), 'Retrieving camera_data for ID: ' + cameraID + ' complete.'])
    print(['DEBUG', HFunc.time_string(), 'Saving camera_data to recording file'])
    NWBio.save_tracking_data(fpath, camera_data)
    print(['DEBUG', HFunc.time_string(), 'Saving camera_data to recording file complete.'])
    # Copy video data directly to recording folder
    thread_list = []
    for cameraID in camera_settings['CameraSpecific'].keys():
        t = threading.Thread(target=file_managers[cameraID].copy_over_video_data,
                             args=(os.path.dirname(fpath), cameraID))
        t.start()
        thread_list.append(t)
    for t in thread_list:
        t.join()


def list_general_settings_history(path):
    """
    Assumes all files on the path are NWB files with General settings stored

    Returns:
        dictionary - just as General settings, but each key contains
                     a list of values for that key in files on the path.
        list       - full paths to all settings files

        All lists are sorted starting from the most recent timestamp in filename.
        None is entered if key is missing in settings file.
    """
    dir_items = os.listdir(path)
    filetimes = []
    filenames = []
    for item in dir_items:
        if item.endswith('.settings.nwb'):
            filename = os.path.join(path, item)
            filenames.append(filename)
            filetimes.append(datetime.strptime(item[:19], '%Y-%m-%d_%H-%M-%S'))
    # Sort filenames based on timestamps
    filenames = [x for _, x in sorted(zip(filetimes, filenames))][::-1]
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


class RecordingManagerException(Exception):
    pass


class OpenEphysMessenger(object):

    def __init__(self):

        self._closed = False

        self._pipe, self.open_ephys_message_pipe = multiprocessing.Pipe()

        print('Connecting to Open Ephys GUI via ZMQ...')
        self._subscriber = SubscribeToOpenEphys(verbose=False)
        self._subscriber.connect()
        self._publisher = PublishToOpenEphys()
        print('Connecting to Open Ephys GUI via ZMQ Successful')

        self._subscriber.add_callback(lambda msg: self._pipe.send(msg))
        self._publisher_thread = threading.Thread(target=self.publisher_thread_method)
        self._publisher_thread.start()

    def send_message_to_open_ephys(self, msg):
        self._publisher.sendMessage(msg)

    @property
    def closed(self):
        return self._closed

    def publisher_thread_method(self):
        while not self.closed:
            if self._pipe.poll(0.1):
                self._publisher.sendMessage(self._pipe.recv())

    def close(self):
        self._subscriber.disconnect()
        self._publisher.close()
        self._closed = True
        self._publisher_thread.join()


def format_channel_map(channel_map):
    if not isinstance(channel_map, dict):
        raise ValueError('channel_map must be dict')
    for channel_group in channel_map:
        if not isinstance(channel_group, str):
            raise ValueError('Channel map fields must be strings')
        channel_map[channel_group] = {
            'string': str(channel_map[channel_group]['string']),
            'list': np.array(channel_map[channel_group]['list'])
        }

    return channel_map


class RecordingManager(object):
    """
    Central recording controller. Provides interface to specify settings and start/stop the recording.
    """

    general_settings = {
        'experimenter': '',
        'root_folder': package_config()['root_folder'],
        'animal': '',
        'experiment_id': '',
        'arena_size': np.array([87.5, 125]),
        'badChan': '',
        'rec_file_path': '',
        'Tracking': np.array(False),
        'channel_map': {},
        'TaskActive': np.array(False)
    }

    general_settings_formatting = {
        'experimenter': str,
        'root_folder': str,
        'animal': str,
        'experiment_id': str,
        'arena_size': np.array,
        'badChan': str,
        'rec_file_path': str,
        'Tracking': np.array,
        'channel_map': format_channel_map,
        'TaskActive': np.array,
    }

    camera_settings = None

    task_settings = None

    settings_ready = {
        'general_settings': True,
        'camera_settings': False,
        'task_settings': False
    }

    recording_initialized = False
    recording_active = False
    recording_closing = False

    RecordingManagerSettingsFolder = os.path.join(package_config()['root_folder'],
                                                  package_config()['recording_manager_settings_subfolder'])

    global_clock_controller = None
    tracking_controller = None
    online_tracking_processor = None
    open_ephys_messenger = None
    current_task = None
    position_plot = None

    def update_general_settings(self, key, value):

        if key in ('experimenter', 'root_folder', 'animal', 'experiment_id', 'badChan', 'rec_file_path'):
            assert isinstance(value, str)
        elif key in ('Tracking', 'TaskActive'):
            assert isinstance(value, bool) or (isinstance(value, np.ndarray) and value.dtype == np.bool)
        elif key in ('arena_size',):
            assert isinstance(value, (list, np.ndarray)) and len(value) == 2
            assert all([isinstance(x, (int, float)) for x in value])
        elif key in ('channel_map',):
            assert isinstance(value, dict)
        else:
            raise ValueError('key {} not found.'.format(key))

        self.general_settings[key] = self.general_settings_formatting[key](value)

        if key == 'arena_size':
            self.settings_ready['task_settings'] = False

    def update_camera_settings(self, camera_settings):
        self.camera_settings = camera_settings
        self.settings_ready['camera_settings'] = True

    def update_task_settings(self, task_settings):
        self.task_settings = task_settings
        self.settings_ready['task_settings'] = True

    def camera_settings_app(self):
        return CameraSettingsApp(self.update_camera_settings, self.camera_settings)

    def task_settings_app(self):
        return TaskSettingsApp(self.general_settings['arena_size'], self.update_task_settings,
                               self.task_settings)

    def save_settings(self, fpath):
        """Saves settings into NWB file.
        """
        NWBio.save_settings(fpath, {'General': self.general_settings,
                                    'Time': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                                    'CameraSettings': self.camera_settings,
                                    'TaskSettings': self.task_settings
                                    })

    def load_settings(self, fpath):
        """Loads settings from NWB file.
        """
        settings = NWBio.load_settings(fpath)
        if 'General' in settings:
            for key, value in settings['General'].items():
                self.update_general_settings(key, value)
        if 'CameraSettings' in settings:
            self.update_camera_settings(settings['CameraSettings'])
        if 'TaskSettings' in settings:
            self.update_task_settings(settings['TaskSettings'])

    @staticmethod
    def find_latest_matching_settings_filepath(path, settings=None):

        general_settings_history, filenames = list_general_settings_history(path)

        if len(filenames) == 0:

            return None

        if settings is None or len(settings.keys()) == 0:

            # If no specific settings were requested, return the path to latest settings file
            filepath = filenames[0]

        else:

            # Loop through all settings files
            filepaths = []
            for nfile in range(len(filenames)):
                # Check if all specified settings match to the one in the file
                settings_correct = []
                for key in settings.keys():
                    settings_correct.append(general_settings_history[key][nfile] == settings[key])
                if all(settings_correct):
                    filepaths.append(filenames[nfile])
            if len(filepaths) > 0:  # Use the most recent settings file path
                filepath = filepaths[0]
            else:  # If no settings matched, return None
                filepath = None

        return filepath

    def list_general_setting_options(self, setting_key):
        general_settings_history, _ = list_general_settings_history(self.RecordingManagerSettingsFolder)
        return sorted(list(set(general_settings_history[setting_key])))

    def load_last_settings(self):
        # Hard-code which settings to use for searching the latest settings
        settings = {}
        if len(self.general_settings['animal']) > 0:
            settings['animal'] = self.general_settings['animal']
        if len(self.general_settings['experiment_id']) > 0:
            settings['experiment_id'] = self.general_settings['experiment_id']
        # Find the latest settings file
        fpath = self.find_latest_matching_settings_filepath(self.RecordingManagerSettingsFolder, settings)
        if fpath is None:
            # If no matching file is found, show error message
            criteria = ''
            for key in sorted(settings.keys()):
                criteria += '{}: {}\n'.format(key, settings[key])

            raise ValueError('No settings found for criteria:\n' + criteria)
        else:
            self.load_settings(fpath)

    @staticmethod
    def test_ping_of_ip_addresses(ip_addresses):
        """Returns a list of devices that fail to ping back.

        :param tuple ip_addresses: tuple of list of IP address strings
        :return: tuple of bool values for each element of ip_addresses. True if device can be pinged successfully.
        """

        def test_ping_and_update_locked_list(ip_address, locked_list, list_pos, list_lock):
            ret = HFunc.test_pinging_address(ip_address)
            with list_lock:
                locked_list[list_pos] = ret

        # Check each device in a seperate thread to speed things up
        output_lock = threading.Lock()
        address_ping_outcome = [None] * len(ip_addresses)
        threads_address_pings = []
        for n_address, address in enumerate(ip_addresses):
            t = threading.Thread(target=test_ping_and_update_locked_list,
                                 args=[address, address_ping_outcome, n_address, output_lock])
            t.start()
            threads_address_pings.append(t)
        for t in threads_address_pings:
            t.join()

        return tuple(address_ping_outcome)

    def test_devices(self):
        """Returns status of devices required for recording.

        :return: list of dicts {'name': str, 'available': bool}
        """

        ip_addresses = []
        device_names = []

        # Add tracking cameras to list
        if self.general_settings['Tracking']:
            if self.settings_ready['camera_settings']:
                assert not (self.camera_settings is None)  # camera_settings should not be None if settings_ready

                for cameraID in sorted(self.camera_settings['CameraSpecific'].keys(), key=int):
                    ip_addresses.append(self.camera_settings['CameraSpecific'][cameraID]['address'])
                    device_names.append('Camera {}'.format(cameraID))

            else:

                raise RecordingManagerException('Tracking is turned on, but Camera Settings are not available.')

        # Add feeders to list
        if self.general_settings['TaskActive']:
            if self.settings_ready['task_settings']:
                assert not (self.task_settings is None)  # task_settings should not be None if settings_ready

                for feeder_type in self.task_settings['FEEDERs'].keys():
                    for feeder_id in sorted(self.task_settings['FEEDERs'][feeder_type].keys(), key=int):

                        if self.task_settings['FEEDERs'][feeder_type][feeder_id]['Active']:

                            ip_addresses.append(self.task_settings['FEEDERs'][feeder_type][feeder_id]['IP'])
                            device_names.append('Feeder {} {}'.format(feeder_type, feeder_id))

            else:

                raise RecordingManagerException('Task is turned on, but Task Settings are not available.')

        device_names = tuple(device_names)

        # Test pinging of these devices
        device_availability = self.test_ping_of_ip_addresses(tuple(ip_addresses))

        return tuple([{'name': name,
                       'available': availability}
                      for name, availability in zip(device_names, device_availability)])

    def check_if_ready_to_initialize(self):

        if self.general_settings['Tracking']:
            if not self.settings_ready['camera_settings']:
                raise RecordingManagerException('Tracking is turned on, but Camera Settings are not available.')

        if self.general_settings['TaskActive']:
            if not self.general_settings['Tracking']:
                raise RecordingManagerException('Tracking must be turned on')
            if not self.settings_ready['task_settings']:
                raise RecordingManagerException('Task is turned on, but Task Settings are not available.')

    def create_task_arguments(self):
        return (
            deepcopy(self.task_settings),
            self.open_ephys_messenger.open_ephys_message_pipe,
            self.online_tracking_processor.combPosHistory,
            self.online_tracking_processor.combPos_update_interval,
            self.online_tracking_processor.position_histogram_dict
        )

    def init_devices(self):

        if self.recording_active or self.recording_closing:
            raise RecordingManagerException('Recording already ongoing.')

        if self.recording_initialized:
            raise RecordingManagerException('Recording already initialized.')

        # self.check_if_ready_to_initialize()

        # Initialize tracking
        if self.general_settings['Tracking']:

            # print('Connecting to GlobalClock RPi...')
            # args = (self.camera_settings['General']['global_clock_ip'],
            #         self.camera_settings['General']['ZMQcomms_port'],
            #         self.camera_settings['General']['username'],
            #         self.camera_settings['General']['password'])
            # self.global_clock_controller = RPiInterface.GlobalClockControl(*args)
            # print('Connecting to GlobalClock RPi Successful')
            #
            # # Connect to tracking RPis
            # print('Connecting to tracking RPis...')
            # self.tracking_controller = RPiInterface.TrackingControl(self.camera_settings)
            # print('Connecting to tracking RPis Successful')

            # Initialize onlineTrackingData class
            print('Initializing Online Tracking Data Processor...')
            self.online_tracking_processor = RPiInterface.onlineTrackingData(
                self.camera_settings, self.general_settings['arena_size'],
                HistogramParameters={'margins': 10,  # histogram data margins in centimeters
                                     'binSize': 2,  # histogram binSize in centimeters
                                     'speedLimit': 10}  # centimeters of distance in last second to be included
            )
            print('Initializing Tracking Data Processor Successful')

        # # Initialize talking to OpenEphysGUI
        # self.open_ephys_messenger = OpenEphysMessenger()
        #
        # # Initialize Task
        # if self.general_settings['TaskActive']:
        #
        #     print('Initializing Task...')
        #     task_module = import_module('Tasks.' + self.task_settings['name'] + '.Task')
        #     self.current_task = task_module.Core(*self.create_task_arguments())
        #     print('Initializing Task Successful')

        self.recording_initialized = True
        print('Recording Initialization Successful')

    def cancel_init_devices(self):

        if not self.recording_initialized:
            raise RecordingManagerException('Devices are not initialized.')

        # Initialize tracking
        if self.general_settings['Tracking']:

            print('Closing GlobalClock Controller...')
            self.global_clock_controller.close()
            print('Closing GlobalClock Controller Successful')

            print('Closing tracking RPis...')
            self.tracking_controller.close()
            print('Closing tracking RPis Successful')

            print('Closing Online Tracking Data...')
            self.online_tracking_processor.close()
            print('Closing Online Tracking Data Successful')

        # Close channel to OpenEphysGUI
        self.open_ephys_messenger.close()

        # Initialize Task
        if self.general_settings['TaskActive']:

            print('Closing Task...')
            self.current_task.stop()
            print('Closing Task Successful')

        self.recording_initialized = False
        print('Recording Initialization Cancelled successfully')

    def create_position_plot_arguments(self):
        return (
            self.online_tracking_processor.combPosHistory,
            self.online_tracking_processor.position_histogram_dict,
            self.online_tracking_processor.position_histogram_update_parameters,
            self.online_tracking_processor.position_histogram_dict_updating,
            self.online_tracking_processor.is_alive,
            self.general_settings['arena_size'],
            self.camera_settings['General']['LED_angle']
        )

    def start_rec(self):

        if self.recording_active or self.recording_closing:
            raise RecordingManagerException('Recording already ongoing.')

        if not self.recording_initialized:
            raise RecordingManagerException('Devices must be initialized first')
        else:
            self.recording_initialized = False

        # # Start Open Ephys GUI recording
        # print('Starting Open Ephys GUI Recording...')
        # recording_folder_root = os.path.join(self.general_settings['root_folder'], self.general_settings['animal'])
        # command = 'StartRecord RecDir=' + recording_folder_root + ' CreateNewDir=1'
        # self.open_ephys_messenger.send_message_to_open_ephys(command)
        #
        # # Make sure OpenEphys is recording
        # recording_file = get_recording_file_path(recording_folder_root)
        # while not recording_file:
        #     sleep(0.1)
        #     recording_file = get_recording_file_path(recording_folder_root)
        # while not check_if_nwb_recording(recording_file):
        #     sleep(0.1)
        # self.general_settings['rec_file_path'] = recording_file
        # print('Starting Open Ephys GUI Recording Successful')
        #
        # # Start the tracking scripts on all RPis
        # if self.general_settings['Tracking']:
        #     print('Starting tracking RPis...')
        #     self.tracking_controller.start()
        #     print('Starting tracking RPis Successful')
        #     # Tracking controller start command should not complete before RPis have really executed the command
        #     print('Starting GlobalClock RPi...')
        #     self.global_clock_controller.start()
        #     print('Starting GlobalClock RPi Successful')

        # Start cumulative plot
        if self.general_settings['Tracking']:
            print('Starting Position Plot...')
            self.position_plot_process = PosPlot(*self.create_position_plot_arguments())
            print('Starting Position Plot Successful')

        # # Start task
        # if self.general_settings['TaskActive']:
        #     print('Starting Task...')
        #     self.current_task.run()
        #     print('Starting Task Successful')

        self.recording_active = True

    def compile_recording_data(self):

        # Store recording settings to recording file
        self.save_settings(self.general_settings['rec_file_path'])
        print('Settings saved to Recording File')

        # Store settings for RecordingManager history reference
        RecordingManagerSettingsPath = os.path.join(self.general_settings['root_folder'],
                                                    self.RecordingManagerSettingsFolder)
        if not os.path.isdir(RecordingManagerSettingsPath):
            os.mkdir(RecordingManagerSettingsPath)
        RecordingFolderName = os.path.basename(os.path.dirname(self.general_settings['rec_file_path']))
        RecordingManagerSettingsFilePath = os.path.join(RecordingManagerSettingsPath,
                                                        RecordingFolderName + '.settings.nwb')
        self.save_settings(RecordingManagerSettingsFilePath)
        print('Settings saved to Recording Manager Settings Folder')

        if self.general_settings['Tracking']:
            # Store tracking  data in recording file
            print('Copying over tracking data to Recording File...')
            store_camera_data_to_recording_file(self.camera_settings, self.general_settings['rec_file_path'])
            print('Copying over tracking data to Recording File Successful')

    def stop_rec(self):

        if self.recording_closing:
            raise RecordingManagerException('Recording closing already in progress.')
        elif not self.recording_active or self.recording_initialized:
            raise RecordingManagerException('Recording is not currently ongoing.')
        else:
            self.recording_active = False
            self.recording_closing = True

        # Stop Task
        if self.general_settings['TaskActive']:
            print('Stopping Task...')
            self.current_task.stop()
            print('Stopping Task Successful')

        # Stop updating tracking system
        if self.general_settings['Tracking']:
            print('Closing Online Tracking Data...')
            self.online_tracking_processor.close()
            print('Closing Online Tracking Data Successful')
            print('Closing GlobalClock Controller...')
            self.global_clock_controller.stop()
            self.global_clock_controller.close()
            print('Closing GlobalClock Controller Successful')
            # Stop cameras
            print('Stopping tracking RPis...')
            self.tracking_controller.close()
            print('Stopping tracking RPis Successful')

        # Stop Open Ephys Recording
        while check_if_nwb_recording(self.general_settings['rec_file_path']):
            print('Stopping Open Ephys GUI Recording...')
            self.open_ephys_messenger.send_message_to_open_ephys('StopRecord')
            sleep(0.1)
        print('Stopping Open Ephys GUI Recording Successful')

        self.compile_recording_data()

        # Close connection with OpenEphysGUI
        self.open_ephys_messenger.close()

        # Position plot should have close automatically after tracking was closed
        if self.general_settings['Tracking']:
            self.position_plot_process.PosPlot.close()

        self.recording_closing = False

    def close(self):
        if self.recording_initialized:
            self.cancel_init_devices()
        elif self.recording_active:
            self.stop_rec()
        elif self.recording_closing:
            while self.recording_closing:
                sleep(1)
        else:
            pass


def list_window(items):
    """
    Creates a new window with a list of string items to select from.
    Double clicking returns the string on the item selected

    Input:  list of strings
    Output: selected list element
    """
    dialog_widget = QtWidgets.QDialog()
    dialog_widget.setWindowTitle('Choose option:')
    dialog_widget.resize(200, 200)
    list_widget = QtWidgets.QListWidget()
    list_widget.addItems(items)
    list_widget.itemDoubleClicked.connect(dialog_widget.accept)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidget(list_widget)
    scroll.setWidgetResizable(True)
    dialog_layout = QtWidgets.QVBoxLayout(dialog_widget)
    dialog_layout.addWidget(scroll)
    dialog_widget.show()
    if dialog_widget.exec_():
        selected_item = items[list_widget.currentRow()]
    else:
        selected_item = None

    return selected_item


def add_boolean_setting_to_widget(label, state, value_change_callback, parent, layout):
    checkbox_widget = QtWidgets.QCheckBox(label, parent)
    layout.addWidget(checkbox_widget)
    checkbox_widget.setChecked(state)
    checkbox_widget.stateChanged.connect(lambda: value_change_callback(checkbox_widget.isChecked()))


def add_x_y_setting_to_widget(label, x_val, y_val, value_change_callback, parent, layout):

    widget = QtWidgets.QWidget(parent)
    layout.addWidget(widget)

    label = QtWidgets.QLabel(label, widget)

    xy_widget = QtWidgets.QWidget(widget)

    x_label = QtWidgets.QLabel('    X', xy_widget)
    x_textbox = QtWidgets.QTextEdit(str(x_val), xy_widget)
    y_label = QtWidgets.QLabel('    Y', xy_widget)
    y_textbox = QtWidgets.QTextEdit(str(y_val), xy_widget)
    xy_layout = QtWidgets.QHBoxLayout(xy_widget)
    xy_layout.setContentsMargins(2, 2, 2, 2)
    xy_layout.setSpacing(4)
    xy_layout.addWidget(x_label)
    xy_layout.addWidget(x_textbox)
    xy_layout.addWidget(y_label)
    xy_layout.addWidget(y_textbox)
    x_textbox.setFixedHeight(27)
    y_textbox.setFixedHeight(27)

    x_textbox.textChanged.connect(lambda: value_change_callback([x_textbox.toPlainText(),
                                                                 y_textbox.toPlainText()]))

    widget_layout = QtWidgets.QHBoxLayout(widget)
    widget_layout.setContentsMargins(2, 2, 2, 2)
    widget_layout.setSpacing(2)
    widget_layout.addWidget(label)
    widget_layout.addWidget(xy_widget)

    widget_layout.setStretch(0, 1)
    widget_layout.setStretch(1, 2)


def add_text_setting_to_widget(label, text, value_change_callback, parent, layout, button_kwargs=None):

    widget = QtWidgets.QWidget(parent)
    layout.addWidget(widget)

    label = QtWidgets.QLabel(label, widget)
    textbox = QtWidgets.QTextEdit(text, widget)
    textbox.textChanged.connect(lambda: value_change_callback(textbox.toPlainText()))

    button = None
    if not (button_kwargs is None):
        button = QtWidgets.QPushButton(button_kwargs['label'], widget)
        button.setFixedSize(60, 27)
        button.clicked.connect(lambda: button_kwargs['connect'](textbox))

    textbox.setFixedHeight(27)
    widget_layout = QtWidgets.QHBoxLayout(widget)
    widget_layout.setContentsMargins(2, 2, 2, 2)
    widget_layout.setSpacing(2)
    widget_layout.addWidget(label)
    widget_layout.addWidget(textbox)
    if not (button_kwargs is None):
        widget_layout.addWidget(button)

    if not (button_kwargs is None):
        widget_layout.setStretch(0, 2)
        widget_layout.setStretch(1, 3)
        widget_layout.setStretch(2, 1)
    else:
        widget_layout.setStretch(0, 1)
        widget_layout.setStretch(1, 2)


class ItemDictionaryWidget(QtWidgets.QFrame):

    def __init__(self, label, col_1_label, col_2_label, item_dict, value_change_callback, parent):
        super().__init__(parent=parent)

        self.value_change_callback = value_change_callback

        QtWidgets.QHBoxLayout(self)

        left_panel_widget = QtWidgets.QWidget(self)
        self.layout().addWidget(left_panel_widget)
        left_panel_layout = QtWidgets.QVBoxLayout(left_panel_widget)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(2)
        left_panel_layout.setAlignment(QtCore.Qt.AlignTop)

        label = QtWidgets.QLabel(label, left_panel_widget)
        left_panel_layout.addWidget(label)
        new_item_button = QtWidgets.QPushButton('New Group', left_panel_widget)
        new_item_button.clicked.connect(lambda: self.create_new_item())
        new_item_button.setFont(QtGui.QFont(new_item_button.fontInfo().family(), 8))
        left_panel_layout.addWidget(new_item_button)

        right_panel_widget = QtWidgets.QWidget(self)
        self.layout().addWidget(right_panel_widget)
        right_panel_layout = QtWidgets.QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(2)

        right_panel_header_widget = QtWidgets.QWidget(right_panel_widget)
        right_panel_layout.addWidget(right_panel_header_widget)
        col_1_label = QtWidgets.QLabel(col_1_label, right_panel_header_widget)
        col_2_label = QtWidgets.QLabel(col_2_label, right_panel_header_widget)
        right_panel_header_layout = QtWidgets.QHBoxLayout(right_panel_header_widget)
        right_panel_header_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_header_layout.setSpacing(0)
        right_panel_header_layout.addWidget(col_1_label)
        right_panel_header_layout.addWidget(col_2_label)

        scroll = QtWidgets.QScrollArea(right_panel_header_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(70)
        right_panel_layout.addWidget(scroll)
        self.item_widget_list_widget = QtWidgets.QWidget(scroll)
        scroll.setWidget(self.item_widget_list_widget)
        self.item_widget_list_layout = QtWidgets.QVBoxLayout(self.item_widget_list_widget)
        self.item_widget_list_layout.setContentsMargins(0, 0, 0, 0)
        self.item_widget_list_layout.setSpacing(0)

        self.layout().setStretch(0, 1)
        self.layout().setStretch(1, 2)

        self.items = {}

        for key, value in item_dict.items():
            self.create_new_item(col_1_value=key, col_2_value=value)

    def get_item_dict(self):
        return {
            str(item['col_1_textbox'].toPlainText()): {
                'string': str(item['col_2_textbox'].toPlainText()),
                'item_id': item_id,
                'readout_successful': lambda item_id: self.set_item_color(item_id, (100, 255, 100)),
                'readout_failed': lambda item_id: self.set_item_color(item_id, (255, 100, 100))
            }
            for item_id, item in self.items.items()
        }

    def value_changed_event(self):
        return self.value_change_callback(self.get_item_dict())

    def create_new_item(self, col_1_value='', col_2_value=''):

        item_id = uuid4()

        item_widget = QtWidgets.QWidget(self.item_widget_list_widget)
        self.item_widget_list_layout.addWidget(item_widget)
        item_layout = QtWidgets.QHBoxLayout(item_widget)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.setSpacing(0)

        col_1_textbox = QtWidgets.QTextEdit(col_1_value, item_widget)
        col_1_textbox.textChanged.connect(self.value_changed_event)
        col_1_textbox.setFixedHeight(27)
        item_layout.addWidget(col_1_textbox)

        col_2_textbox = QtWidgets.QTextEdit(col_2_value, item_widget)
        col_2_textbox.textChanged.connect(self.value_changed_event)
        col_2_textbox.setFixedHeight(27)
        item_layout.addWidget(col_2_textbox)

        remove_button = QtWidgets.QPushButton('X', item_widget)
        remove_button.setFixedSize(15, 27)
        item_layout.addWidget(remove_button)
        remove_button.clicked.connect(lambda: self.remove_item(item_id))

        self.items[item_id] = {'col_1_textbox': col_1_textbox,
                               'col_2_textbox': col_2_textbox,
                               'item_widget': item_widget}

        self.value_change_callback(self.get_item_dict())

    def remove_item(self, item_id):
        item = self.items.pop(item_id)
        item['item_widget'].deleteLater()
        self.value_change_callback(self.get_item_dict())

    def set_item_color(self, item_id, color):
        for column_textbox_name in ('col_1_textbox', 'col_2_textbox'):
            p = self.items[item_id][column_textbox_name].viewport().palette()
            p.setColor(self.items[item_id][column_textbox_name].viewport().backgroundRole(), QtGui.QColor(*color))
            self.items[item_id][column_textbox_name].viewport().setPalette(p)


def convert_item_list_string_values_to_channel_map(item_dict):
    channel_map = {}
    for channel_group, item in item_dict.items():
        try:
            if not ('-' in item['string']):
                raise ValueError('There is no "-" in the string')
            first_channel = int(item['string'][:item['string'].find('-')]) - 1
            second_channel = int(item['string'][item['string'].find('-') + 1:])
            if first_channel >= second_channel:
                raise ValueError('Second channel should be a larger value')
            channel_map[channel_group] = {
                'string': item['string'],
                'list': np.arange(first_channel, second_channel, dtype=np.int64)
            }
            item['readout_successful'](item['item_id'])
        except:
            item['readout_failed'](item['item_id'])

    return channel_map


class QOneShotThread(QtCore.QThread):

    def __init__(self, target):
        self._target = target
        super().__init__()

    def run(self) -> None:
        self._target()


class RecordingManagerGUI(QtWidgets.QMainWindow):

    exit_code_to_reboot = -123

    geometry = None

    test_devices_thread = None
    initialize_devices_thread = None
    start_recording_thread = None
    stop_recording_thread = None

    def __init__(self, recording_manager, geometry=None):
        """
        :param RecordingManager recording_manager:
        """
        super().__init__()

        self.recording_manager = recording_manager

        self.setFixedSize(350, 600)

        if not (geometry is None):
            self.restoreGeometry(geometry)

        # Create general GUI layout

        self.central_widget = QtWidgets.QWidget(self)
        self.button_widget = QtWidgets.QWidget(self.central_widget)
        self.button_widget.setMaximumHeight(220)
        self.general_settings_widget = QtWidgets.QWidget(self.central_widget)
        self.current_recording_file_path = QtWidgets.QLabel('Current recording folder:', self.central_widget)
        self.current_recording_file_path.setFixedHeight(40)
        self.general_settings_label = QtWidgets.QLabel('General Settings', self.central_widget)
        self.general_settings_label.setFont(QtGui.QFont('SansSerif', 15))
        self.central_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.central_layout.addWidget(self.button_widget)
        self.central_layout.addWidget(self.current_recording_file_path)
        self.central_layout.addWidget(self.general_settings_label)
        self.central_layout.addWidget(self.general_settings_widget)
        self.setCentralWidget(self.central_widget)

        # Add buttons to button widget

        # Define buttons
        self.load_last_settings_button = QtWidgets.QPushButton('Load Last Settings', self.button_widget)
        self.load_last_settings_button.clicked.connect(self.load_last_settings_button_callback)
        self.load_settings_button = QtWidgets.QPushButton('Load Settings', self.button_widget)
        self.load_settings_button.clicked.connect(self.load_settings_button_callback)
        self.save_settings_button = QtWidgets.QPushButton('Save Settings', self.button_widget)
        self.save_settings_button.clicked.connect(self.save_settings_button_callback)
        self.camera_settings_button = QtWidgets.QPushButton('Camera Settings', self.button_widget)
        self.camera_settings_button.clicked.connect(self.camera_settings_button_callback)
        self.task_settings_button = QtWidgets.QPushButton('Task Settings', self.button_widget)
        self.task_settings_button.clicked.connect(self.task_settings_button_callback)
        self.test_devices_button = QtWidgets.QPushButton('Test Devices', self.button_widget)
        self.test_devices_button.clicked.connect(self.test_devices_button_callback)
        self.initialize_devices_button = QtWidgets.QPushButton('Initialize Devices', self.button_widget)
        self.initialize_devices_button.clicked.connect(self.initialize_devices_button_callback)
        self.start_recording_button = QtWidgets.QPushButton('Start Recording', self.button_widget)
        self.start_recording_button.clicked.connect(self.start_recording_button_callback)
        self.start_recording_button.setEnabled(False)
        self.stop_recording_button = QtWidgets.QPushButton('Stop Recording', self.button_widget)
        self.stop_recording_button.clicked.connect(self.stop_recording_button_callback)
        self.stop_recording_button.setEnabled(False)

        self.default_button_style = self.stop_recording_button.styleSheet()

        self.stop_recording_button.clicked.connect(lambda: print({'general': self.recording_manager.general_settings,
                                                                  'camera': self.recording_manager.camera_settings,
                                                                  'task': self.recording_manager.task_settings}))

        # Position buttons
        self.button_layout = QtWidgets.QGridLayout(self.button_widget)
        self.button_layout.addWidget(self.load_last_settings_button, 0, 0)
        self.button_layout.addWidget(self.load_settings_button, 1, 0)
        self.button_layout.addWidget(self.save_settings_button, 2, 0)
        self.button_layout.addWidget(self.camera_settings_button, 4, 0)
        self.button_layout.addWidget(self.task_settings_button, 5, 0)
        self.button_layout.addWidget(self.test_devices_button, 0, 1)
        self.button_layout.addWidget(self.initialize_devices_button, 1, 1)
        self.button_layout.addWidget(self.start_recording_button, 2, 1)
        self.button_layout.addWidget(self.stop_recording_button, 3, 1)
        self.button_layout.setHorizontalSpacing(40)
        self.button_layout.setVerticalSpacing(2)

        # Add settings to general settings widget
        self.general_settings_layout = QtWidgets.QVBoxLayout(self.general_settings_widget)
        self.general_settings_layout.setContentsMargins(2, 2, 2, 2)
        self.general_settings_layout.setSpacing(2)

        checkboxes_widget = QtWidgets.QWidget(self.general_settings_widget)
        checkboxes_layout = QtWidgets.QHBoxLayout(checkboxes_widget)
        checkboxes_layout.setContentsMargins(2, 2, 2, 2)
        checkboxes_layout.setSpacing(2)
        self.general_settings_layout.addWidget(checkboxes_widget)
        add_boolean_setting_to_widget(
            'Tracking', self.recording_manager.general_settings['Tracking'],
            lambda state: self.recording_manager.update_general_settings('Tracking', state),
            checkboxes_widget, checkboxes_layout
        )
        add_boolean_setting_to_widget(
            'Task active', self.recording_manager.general_settings['TaskActive'],
            lambda state: self.recording_manager.update_general_settings('TaskActive', state),
            checkboxes_widget, checkboxes_layout
        )

        add_x_y_setting_to_widget(
            'Arena size',
            self.recording_manager.general_settings['arena_size'][0],
            self.recording_manager.general_settings['arena_size'][1],
            lambda text: self.recording_manager.update_general_settings(
                'arena_size', np.array(list(map(lambda x: float(x) if len(x) > 0 else np.nan, text)))
            ),
            self.general_settings_widget, self.general_settings_layout
        )
        add_text_setting_to_widget(
            'Experimenter', self.recording_manager.general_settings['experimenter'],
            lambda text: self.recording_manager.update_general_settings('experimenter', text),
            self.general_settings_widget, self.general_settings_layout,
            button_kwargs={'label': 'options',
                           'connect': lambda text_box: self.pick_setting_from_options('experimenter',
                                                                                      text_box)}
        )
        add_text_setting_to_widget(
            'Experiment ID', self.recording_manager.general_settings['experiment_id'],
            lambda text: self.recording_manager.update_general_settings('experiment_id', text),
            self.general_settings_widget, self.general_settings_layout,
            button_kwargs={'label': 'options',
                           'connect': lambda text_box: self.pick_setting_from_options('experiment_id',
                                                                                      text_box)}
        )
        add_text_setting_to_widget(
            'Animal ID', self.recording_manager.general_settings['animal'],
            lambda text: self.recording_manager.update_general_settings('animal', text),
            self.general_settings_widget, self.general_settings_layout,
            button_kwargs={'label': 'options',
                           'connect': lambda text_box: self.pick_setting_from_options('animal',
                                                                                      text_box)}
        )
        add_text_setting_to_widget(
            'Bad channel list', self.recording_manager.general_settings['badChan'],
            lambda text: self.recording_manager.update_general_settings('badChan', text),
            self.general_settings_widget, self.general_settings_layout,
            button_kwargs={'label': 'options',
                           'connect': lambda text_box: self.pick_setting_from_options('badChan',
                                                                                      text_box)}
        )

        channel_map_widget = ItemDictionaryWidget(
            'Channel map', 'Group', 'Channel range',
            {key: item['string']
             for key, item in self.recording_manager.general_settings['channel_map'].items()},
            lambda item_dict: self.recording_manager.update_general_settings(
                'channel_map',
                convert_item_list_string_values_to_channel_map(item_dict)
            ),
            self.general_settings_widget
        )
        self.general_settings_layout.addWidget(channel_map_widget)

    def pick_setting_from_options(self, setting_key, setting_text_box):
        setting_options = self.recording_manager.list_general_setting_options(setting_key)
        selected_item = list_window(setting_options)
        if not (selected_item is None):
            setting_text_box.setPlainText(selected_item)

    def save_settings_button_callback(self):
        fpath = HFunc.openSingleFileDialog('save', suffix='nwb', caption='Save file name and location')
        if not (fpath is None):
            self.recording_manager.save_settings(fpath)

    def load_settings_button_callback(self):
        fpath = HFunc.openSingleFileDialog('load', suffix='nwb', caption='Select file to load')
        if not (fpath is None):
            self.recording_manager.load_settings(fpath)
            self.restart()

    def load_last_settings_button_callback(self):
        try:
            self.recording_manager.load_last_settings()
            self.restart()
        except ValueError as e:
            HFunc.show_message(str(e))

    def camera_settings_button_callback(self):
        camera_settings_app = self.recording_manager.camera_settings_app()
        CameraSettingsGUI(camera_settings_app, parent=self)

    def task_settings_button_callback(self):
        task_settings_app = self.recording_manager.task_settings_app()
        TaskSettingsGUI(task_settings_app, parent=self)

    @staticmethod
    def display_device_status_message(device_status):

        n_unavailable = sum([not device['available'] for device in device_status])

        if n_unavailable == 0:
            message = 'All devices available.'
        else:
            message = '{} of {} devices unavailable!'.format(n_unavailable, len(device_status))

        message_more = ''
        for device in device_status:
            if device['available']:
                message_more += '{} online\n'.format(device['name'])
            else:
                message_more += '{} OFFLINE\n'.format(device['name'])

        HFunc.show_message(message, message_more=message_more)

    def test_devices(self):
        try:
            device_status = self.recording_manager.test_devices()
            self.display_device_status_message(device_status)
        except RecordingManagerException as e:
            HFunc.show_message(str(e))

    def test_devices_finished(self):
        self.test_devices_button.setStyleSheet(self.default_button_style)
        while self.test_devices_thread.isRunning():
            sleep(0.01)
        self.test_devices_thread = None

    def test_devices_button_callback(self):
        self.test_devices_button.setStyleSheet('background-color: red')
        self.test_devices_thread = QOneShotThread(self.test_devices)
        self.test_devices_thread.finished.connect(self.test_devices_finished)
        self.test_devices_thread.start()

    def initialize_devices(self):
        try:
            self.recording_manager.init_devices()
        except RecordingManagerException as e:
            print(e)
            HFunc.show_message(str(e))

    def initialize_devices_finished(self):
        if self.recording_manager.recording_initialized:
            self.initialize_devices_button.setStyleSheet('background-color: green')
            self.initialize_devices_button.setEnabled(False)
            self.start_recording_button.setEnabled(True)
        else:
            self.initialize_devices_button.setStyleSheet(self.default_button_style)
        while self.initialize_devices_thread.isRunning():
            sleep(0.01)
        self.initialize_devices_thread = None

    def initialize_devices_button_callback(self):
        self.initialize_devices_button.setStyleSheet('background-color: red')
        self.initialize_devices_thread = QOneShotThread(self.initialize_devices)
        self.initialize_devices_thread.finished.connect(self.initialize_devices_finished)
        self.initialize_devices_thread.start()

    def start_recording(self):
        try:
            self.recording_manager.start_rec()
        except RecordingManagerException as e:
            print(e)
            HFunc.show_message(str(e))

    def start_recording_finished(self):
        if self.recording_manager.recording_active:
            self.start_recording_button.setStyleSheet('background-color: green')
            self.start_recording_button.setEnabled(False)
            self.stop_recording_button.setEnabled(True)
        else:
            self.start_recording_button.setStyleSheet(self.default_button_style)
        while self.start_recording_thread.isRunning():
            sleep(0.01)
        self.start_recording_thread = None

    def start_recording_button_callback(self):
        self.start_recording_button.setStyleSheet('background-color: red')
        self.start_recording_thread = QOneShotThread(self.start_recording)
        self.start_recording_thread.finished.connect(self.start_recording_finished)
        self.start_recording_thread.start()

    def stop_recording(self):
        try:
            self.recording_manager.stop_rec()
        except RecordingManagerException as e:
            print(e)
            HFunc.show_message(str(e))

    def stop_recording_finished(self):
        if not self.recording_manager.recording_active:
            self.stop_recording_button.setStyleSheet(self.default_button_style)
            self.stop_recording_button.setEnabled(False)
            self.initialize_devices_button.setEnabled(True)
            HFunc.show_message('Recording Completed Successfully.')
        else:
            self.stop_recording_button.setStyleSheet(self.default_button_style)
        while self.stop_recording_thread.isRunning():
            sleep(0.01)
        self.stop_recording_thread = None

    def stop_recording_button_callback(self):
        self.stop_recording_button.setStyleSheet('background-color: red')
        self.start_recording_button.setStyleSheet(self.default_button_style)
        self.stop_recording_thread = QOneShotThread(self.stop_recording)
        self.stop_recording_thread.finished.connect(self.stop_recording_finished)
        self.stop_recording_thread.start()

    def restart(self):
        self.geometry = self.saveGeometry()
        QtCore.QCoreApplication.exit(self.exit_code_to_reboot)


def recording_manager_gui_with_load_settings_loop(recording_manager):

    # While loop allows restarting GUI without restarting the RecordingManager application
    exit_code = RecordingManagerGUI.exit_code_to_reboot
    geometry = None
    while exit_code == RecordingManagerGUI.exit_code_to_reboot:

        # Start application
        app = QtWidgets.QApplication(sys.argv)
        window = RecordingManagerGUI(recording_manager, geometry=geometry)
        window.show()
        exit_code = app.exec_()
        geometry = deepcopy(window.geometry)
        app.exit(exit_code)
        del app


def main():
    recording_manager = RecordingManager()
    recording_manager_gui_with_load_settings_loop(recording_manager)
    recording_manager.close()


if __name__ == '__main__':
    main()
