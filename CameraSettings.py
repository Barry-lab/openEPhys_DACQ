
from PyQt4 import QtGui
import sys
import numpy as np
from copy import copy
import NWBio
from HelperFunctions import openSingleFileDialog, show_message, QThread_with_completion_callback
import pyqtgraph
from RPiInterface import CameraControl
from io import BytesIO
from socket import socket
import struct
from PIL import Image
from threading import Thread, Timer, Lock
from time import sleep
import cv2

class PerpetualTimer(object):
    '''
    Keeps calling a function at specified interval until stop() method called.
    '''
    def __init__(self, interval, callback):
        '''
        interval - float - seconds between each function call.
        callback - function that is called each time interval has elapsed.
        '''
        self.callback = callback
        self.interval = interval
        self.timer_active = True
        self.isRunning = False
        self.start_timer()

    def run(self):
        self.callback()

    def timeout(self):
        self.isRunning = False
        if self.timer_active:
            Thread(target=self.run).start()
            self.start_timer()

    def start_timer(self):
        if self.timer_active:
            self.isRunning = True
            self.timer = Timer(self.interval, self.timeout)
            self.timer.start()

    def stop(self):
        self.timer_active = False
        if self.isRunning:
            self.timer.cancel()

class CameraDisplay(object):
    def __init__(self, stream_port, image_display_method, cameraID, FPS=10):
        self.image_display_method = image_display_method
        self.cameraID = cameraID
        # Set internal variables
        self.calibration_overlay_on = False
        self.replacement_frame = None
        self.process_and_display_frame_Lock = Lock()
        # Start daemon thread for streaming
        self.stream_open = True
        self.T_stream = Thread(target=self.stream, args=(stream_port,))
        self.T_stream.start()
        # Activate frame timer
        self.pass_frame = False
        self.update_frame_timer = PerpetualTimer(float(1.0 / FPS), self.use_frame)

    def use_frame(self):
        self.pass_frame = True

    def overlay_calibration_on_frame(self, frame):
        pattern = self.calibration_pattern
        ndots_x = self.calibration_ndots_xy[0]
        ndots_y = self.calibration_ndots_xy[1]
        frame = cv2.drawChessboardCorners(frame, (ndots_x, ndots_y), pattern, True)

        return frame

    def process_and_display_frame(self, image_stream):
        with self.process_and_display_frame_Lock:
            if self.replacement_frame is None:
                # Rewind the stream, open it as an image with PIL and convert to numpy
                image_stream.seek(0)
                image = Image.open(image_stream)
                frame = np.array(image)
            else:
                frame = self.replacement_frame
            # Overlay calibration pattern if requested
            if self.calibration_overlay_on:
                frame = self.overlay_calibration_on_frame(frame)
            # Display frame
            self.image_display_method(self.cameraID, frame)

    def stream(self, port):
        # Open connection
        self.server_socket = socket()
        self.server_socket.bind(('0.0.0.0', int(port)))
        self.server_socket.listen(0)
        self.connection = self.server_socket.accept()[0].makefile('rb')
        try:
            while self.stream_open:
                # Read the length of the image as a 32-bit unsigned int. If the
                # length is zero, quit the loop
                image_len = struct.unpack('<L', self.connection.read(struct.calcsize('<L')))[0]
                if not image_len:
                    break
                # Construct a stream to hold the image data and read the image
                # data from the connection
                image_stream = BytesIO()
                image_stream.write(self.connection.read(image_len))
                if self.pass_frame:
                    Thread(target=self.process_and_display_frame, args=(image_stream,)).start()
                    self.pass_frame = False
                else:
                    sleep(0.01)
        except Exception as e:
            if not isinstance(e, struct.error):
                raise e

    def calibration_overlay(self, on_off, pattern=None, ndots_xy=None):
        with self.process_and_display_frame_Lock:
            self.calibration_pattern = pattern
            self.calibration_ndots_xy = ndots_xy
            self.calibration_overlay_on = on_off

    def use_replacement_frame(self, on_off, frame=None):
        with self.process_and_display_frame_Lock:
            if on_off:
                self.replacement_frame = frame
            else:
                self.replacement_frame = None

    def close(self):
        self.update_frame_timer.stop()
        self.stream_open = False
        self.T_stream.join()
        self.connection.close()
        self.server_socket.close()


class CameraSettingsApp(object):
    '''
    Handles camera settings, calibration and provides live stream of camera feed.

    Note! CameraSettingsApp requires a specific workflow for calibration data to match the settings. 
          All settings must be specified before any other methods are called.
    '''
    def __init__(self, CameraSettings=None, with_GUI=False, apply_settings_method=None):
        # Default variables
        self.display_stream_ports_start = 8000
        # Create internal variables
        self.CameraControllers = {}
        self.CameraDisplays = {}
        self.CalibrationData = {}
        # Initialize
        self.with_GUI = with_GUI
        self.apply_settings_method = apply_settings_method
        self.general_settings = self.default_settings('general_settings')
        self.camera_specific_settings = self.default_settings('camera_specific_settings')
        if not (CameraSettings is None):
            self.import_settings(CameraSettings['General'], CameraSettings['CameraSpecific'])
        if self.with_GUI:
            self.GUI = CameraSettingsGUI(self)

    @staticmethod
    def _default_general_settings():
        '''
        Specifies default settigs as well as data types.
        '''
        return {'tracking_mode':          {'value': 'dual_led', 
                                           'description': ('Tracking mode', ('1 LED', 'single_led'), ('2 LED', 'dual_led'))}, 
                'resolution_option':      {'value': 'low', 
                                           'description': ('Video resolution', ('Low', 'low'), ('High', 'high'))}, 
                'LED_separation':         {'value': np.float64(5.0), 
                                           'description': 'LED separation (cm)'}, 
                'LED_angle':              {'value': np.float64(0.0), 
                                           'description': 'LED angle'}, 
                'camera_transfer_radius': {'value': np.float64(40.0), 
                                           'description': 'Camera transfer radius (cm)'}, 
                'smoothing_radius':       {'value': np.int64(3), 
                                           'description': 'OnlineTracker smoothing (cm)'}, 
                'centralIP':              {'value': '192.168.0.10', 
                                           'description': 'IP address of Recording PC'}, 
                'global_clock_ip':        {'value': '192.168.0.8', 
                                           'description': 'IP address of Global Clock RPi'}, 
                'password':               {'value': 'raspberry', 
                                           'description': 'Camera RPi password'}, 
                'username':               {'value': 'pi', 
                                           'description': 'Camera RPi username'}, 
                'OnlineTracker_port':     {'value': '5600', 
                                           'description': 'OnlineTracker port'}, 
                'ZMQcomms_port':          {'value': '5601', 
                                           'description': 'ZMQcomms port'}
                }

    @staticmethod
    def _default_camera_specific_settings():
        '''
        Specifies default settigs as well as data types.
        '''
        return {'1': {'address':  {'value': '192.168.0.20', 
                                   'description': 'IP'}, 
                      'location_xy': {'value': np.array([350.0, 250.0], dtype=np.float64), 
                                   'description': 'Location X,Y (cm)'}, 
                      'offset_xy':   {'value': np.array([6.0, 6.0], dtype=np.float64), 
                                   'description': 'Offset X,Y (cm)'}, 
                      'spacing':  {'value': np.float64(20.38), 
                                   'description': 'Spacing (cm)'}, 
                      'ndots_xy':  {'value': np.array([4, 11], dtype=np.int64), 
                                   'description': 'Dots X,Y'}
                      }
                }

    def general_settings_keyorder(self, settings=None):
        if settings is None:
            settings = self.general_settings
        if not (settings is None):
            return sorted(settings.keys())

    def single_camera_specific_settings_keyorder(self, settings=None):
        if settings is None:
            cameraID = self.camera_specific_settings.keys()[0]
            settings = self.camera_specific_settings[cameraID]
        if not (settings is None):
            return sorted(settings.keys())

    def default_settings(self, settings_name):
        if settings_name == 'general_settings':
            return CameraSettingsApp._default_general_settings()
        elif settings_name == 'camera_specific_settings':
            return CameraSettingsApp._default_camera_specific_settings()

    def default_single_camera_specific_settings(self):
        '''
        Returns settings and keyorder of the first cameraID
        '''
        camera_specific_settings = self.default_settings('camera_specific_settings')
        settings = camera_specific_settings[camera_specific_settings.keys()[0]]
        keyorder = self.single_camera_specific_settings_keyorder()

        return settings

    def update_settings(self, general_settings, camera_specific_settings):
        self.general_settings = general_settings
        self.camera_specific_settings = camera_specific_settings

    def import_settings(self, general_settings, camera_specific_settings):
        # Create new settings dictionaries by copying over description from default settings.
        default_general_settings = self.default_settings('general_settings')
        default_single_camera_specific_settings = self.default_single_camera_specific_settings()
        for key in general_settings.keys():
            general_settings[key] = {'value': general_settings[key], 
                            'description': default_general_settings[key]['description']}
        for cameraID in camera_specific_settings.keys():
            if 'CalibrationData' in camera_specific_settings[cameraID].keys():
                self.CalibrationData[cameraID] = camera_specific_settings[cameraID].pop('CalibrationData')
            for key in camera_specific_settings[cameraID].keys():
                camera_specific_settings[cameraID][key] = {'value': camera_specific_settings[cameraID][key], 
                                                           'description': default_single_camera_specific_settings[key]['description']}
        self.general_settings = general_settings
        self.camera_specific_settings = camera_specific_settings

    def export_settings(self):
        general = {}
        for key in self.general_settings.keys():
            general[key] = self.general_settings[key]['value']
        camera_specific = {}
        for cameraID in self.camera_specific_settings.keys():
            camera_specific[cameraID] = {}
            for key in self.camera_specific_settings[cameraID].keys():
                camera_specific[cameraID][key] = self.camera_specific_settings[cameraID][key]['value']
            if cameraID in self.CalibrationData.keys():
                camera_specific[cameraID]['CalibrationData'] = self.CalibrationData[cameraID]

        return {'General': general, 'CameraSpecific': camera_specific}

    def load_from_file(self, filename):
        CameraSettings = NWBio.load_settings(filename, '/CameraSettings/')
        self.import_settings(CameraSettings['General'], CameraSettings['CameraSpecific'])
        if self.with_GUI:
            self.GUI.write_settings_into_GUI(self.general_settings, self.general_settings_keyorder(), 
                                             self.camera_specific_settings, self.single_camera_specific_settings_keyorder())

    def save_to_file(self, filename):
        NWBio.save_settings(filename, self.export_settings(), path='/CameraSettings/')

    def apply(self):
        if not (self.apply_settings_method is None):
            self.apply_settings_method(self.export_settings())
        else:
            raise ValueError('No method apply_to_parent() provided.')

    def _initialize_camera(self, cameraID):
        address = self.camera_specific_settings[cameraID]['address']['value']
        port = self.general_settings['ZMQcomms_port']['value']
        username = self.general_settings['username']['value']
        password = self.general_settings['password']['value']
        resolution_option = 'high'
        self.CameraControllers[cameraID] = CameraControl(address, port, username, 
                                                         password, resolution_option)

    def initialize_cameras(self):
        self.CameraControllers = {}
        T__initialize_camera = []
        for cameraID in self.camera_specific_settings.keys():
            T = Thread(target=self._initialize_camera, args=(cameraID,))
            T.start()
            T__initialize_camera.append(T)
        for T in T__initialize_camera:
            T.join()

    def _close_camera(self, cameraID):
        controller = self.CameraControllers.pop(cameraID)
        controller.close()

    def close_cameras(self):
        T__close_camera = []
        for key in self.CameraControllers.keys():
            T = Thread(target=self._close_camera, args=(key,))
            T.start()
            T__close_camera.append(T)
        for T in T__close_camera:
            T.join()

    def ensure_cameras_are_closed(self):
        if hasattr(self, 'CameraControllers'):
            self.close_cameras()

    def camera_display_start(self, image_display_method):
        '''
        image_display_method - is called with 2 inputs (cameraID, frame) to update displayed image.
        '''
        # Produce separate ports for streaming from each camera
        display_stream_ports = {}
        for n_cam, cameraID in enumerate(self.camera_specific_settings.keys()):
            display_stream_ports[cameraID] = self.display_stream_ports_start + n_cam
        # Start CameraDisplay class for each camera
        for cameraID in self.camera_specific_settings.keys():
            self.CameraDisplays[cameraID] = CameraDisplay(display_stream_ports[cameraID], 
                                                          image_display_method, cameraID)
        # Start streaming from all cameras
        for cameraID in self.camera_specific_settings.keys():
            self.CameraControllers[cameraID].start_streaming(self.general_settings['centralIP']['value'], 
                                                             display_stream_ports[cameraID])

    def camera_display_stop(self):
        for key in copy(self.CameraDisplays.keys()):
            controller = self.CameraDisplays.pop(key)
            controller.close()

    def initialize_cameras_and_streaming(self, image_display_method):
        '''
        image_display_method - is called with 2 inputs (cameraID, frame) to update displayed image.
        '''
        self.initialize_cameras()
        self.camera_display_start(image_display_method)

    def stop_cameras_and_streaming(self):
        self.close_cameras()
        self.camera_display_stop()

    def calibrate(self, cameraID):
        '''
        Erases any previous calibration data.

        Attempts to calibrate the camera with current settings.
        If successful, the data is stored in self.CalibrationData.

        Returns True if calibration successful, False otherwise.

        cameraID - specifies which camera to calibrate.
        '''
        if cameraID in self.CalibrationData.keys():
            tmp =  self.CalibrationData.pop(cameraID)
        if cameraID in self.CameraControllers.keys():
            calibration_parameters = {'ndots_xy': self.camera_specific_settings[cameraID]['ndots_xy']['value'], 
                                      'spacing': self.camera_specific_settings[cameraID]['spacing']['value'], 
                                      'offset_xy': self.camera_specific_settings[cameraID]['offset_xy']['value']}
            calibration = self.CameraControllers[cameraID].calibrate(calibration_parameters)
            if not (calibration is None):
                self.CalibrationData[cameraID] = calibration
                return True
            else:
                return False

    def overlay_calibration(self, cameraID, on_off):
        if (cameraID in self.CalibrationData.keys()) and (cameraID in self.CameraDisplays.keys()):
            if on_off:
                pattern = self.CalibrationData[cameraID]['low']['pattern']
                ndots_xy = self.camera_specific_settings[cameraID]['ndots_xy']['value']
                self.CameraDisplays[cameraID].calibration_overlay(True, pattern, ndots_xy)
            else:
                self.CameraDisplays[cameraID].calibration_overlay(False)

    def show_calibration_frame(self, cameraID, on_off):
        if (cameraID in self.CalibrationData.keys()) and (cameraID in self.CameraDisplays.keys()):
            if on_off:
                frame = self.CalibrationData[cameraID]['low']['frame']
                self.CameraDisplays[cameraID].use_replacement_frame(True, frame)
            else:
                self.CameraDisplays[cameraID].use_replacement_frame(False)

    def close(self):
        self.ensure_cameras_are_closed()
        if hasattr(self, 'CameraDisplays'):
            for cameraID in self.CameraDisplays.keys():
                self.CameraDisplays[cameraID].close()
        if hasattr(self, 'GUI'):
            self.GUI.close()


class radio_button_menu(QtGui.QWidget):

    def __init__(self, name_value_pairs):
        super(radio_button_menu, self).__init__()
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.buttons = {}
        for name, value in name_value_pairs:
            self.buttons[value] = QtGui.QRadioButton(name)
            layout.addWidget(self.buttons[value])
        self.setLayout(layout)

    def set_checked(self, value):
        self.buttons[value].setChecked(True)

    def get_checked(self):
        for value in self.buttons.keys():
            if self.buttons[value].isChecked():
                return value


def setDoubleBoxStretch(box, first, second):
    box.setStretch(0, first)
    box.setStretch(1, second)
    return box


class settings_menu(QtGui.QWidget):
    def __init__(self, settings, keyorder, primary_orientation, secondary_orientation):
        '''
        settings - dict - from CameraSettings.default_settings()
        keyorder - list - settings.keys() in preferred order
        primary_orientation - str - 'horizontal' or 'vertical' or 'h' or 'v'
                              orientation of the main layout
        secondary_orientation - str - 'horizontal' or 'vertical' or 'h' or 'v'
                                orientation of each individual settings layout
        '''
        super(settings_menu, self).__init__()
        self.settings = settings
        # Create top layout
        if primary_orientation == 'horizontal' or primary_orientation == 'h':
            layout = QtGui.QHBoxLayout()
        elif primary_orientation == 'vertical' or primary_orientation == 'v':
            layout = QtGui.QVBoxLayout()
        else:
            raise ValueError('primary_orientation argument must be horizontal or vertical')
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        # Create settings option
        self.handles = {}
        for key in keyorder:
            if secondary_orientation == 'horizontal' or secondary_orientation == 'h':
                box = QtGui.QHBoxLayout()
            elif secondary_orientation == 'vertical' or secondary_orientation == 'v':
                box = QtGui.QVBoxLayout()
            else:
                raise ValueError('secondary_orientation argument must be horizontal or vertical')
            box.setContentsMargins(0,0,0,0)
            # Create settings option type based on settings[key]['description value']
            if isinstance(self.settings[key]['description'], str):
                box.addWidget(QtGui.QLabel(self.settings[key]['description']))
                self.handles[key] = QtGui.QLineEdit()
            elif isinstance(self.settings[key]['description'], tuple):
                box.addWidget(QtGui.QLabel(self.settings[key]['description'][0]))
                self.handles[key] = radio_button_menu(self.settings[key]['description'][1:])
            box.addWidget(self.handles[key])
            # Add individual setting to settings_menu layout in a frame
            frame = QtGui.QFrame()
            frame.setLayout(setDoubleBoxStretch(box, 2, 1))
            frame.setFrameStyle(0)
            layout.addWidget(frame)
        # Populate settings_menu values with data from settings
        self.put(self.settings)

    def put(self, settings):
        '''
        Populates settings_menu in handles with data from settings.

        settings - dict - with structure corresponding to create_settings_menu() input.
        '''
        for key in settings.keys():
            value = copy(settings[key]['value'])
            if isinstance(self.handles[key], QtGui.QLineEdit):
                if isinstance(value, np.ndarray):
                    value = ','.join(map(str, value))
                elif isinstance(value, np.int64) or isinstance(value, np.float64):
                    value = str(value)
                elif not isinstance(value, str):
                    raise ValueError('Expected numpy.ndarray, numpy.int64, numpy.float64 or str,\n' + \
                                     'but got ' + str(type(value)))
                self.handles[key].setText(value)
            elif isinstance(self.handles[key], radio_button_menu):
                self.handles[key].set_checked(value)
            else:
                raise ValueError('Expected ' + str(QtGui.QLineEdit) + ' or ' + str(radio_button_menu) +',\n' + \
                                 'but got ' + str(type(self.handles[key])))

    def get(self):
        '''
        Returns settings values with values extracted from settings_menu in handles.
        '''
        for key in self.settings.keys():
            if isinstance(self.handles[key], QtGui.QLineEdit):
                value = str(self.handles[key].text())
                if isinstance(self.settings[key]['value'], np.ndarray):
                    self.settings[key]['value'] = np.array(value.split(','), 
                                                           dtype=self.settings[key]['value'].dtype)
                elif isinstance(self.settings[key]['value'], np.int64):
                    self.settings[key]['value'] = np.int64(value)
                elif isinstance(self.settings[key]['value'], np.float64):
                    self.settings[key]['value'] = np.float64(value)
                elif isinstance(self.settings[key]['value'], str):
                    self.settings[key]['value'] = value
                else:
                    raise ValueError('Expected numpy.ndarray, numpy.int64, numpy.float64 or str,\n' + \
                                     'but got ' + str(type(self.settings[key]['value'])))
            elif isinstance(self.handles[key], radio_button_menu):
                self.settings[key]['value'] = self.handles[key].get_checked()
            else:
                raise ValueError('Expected ' + str(QtGui.QLineEdit) + ' or ' + str(radio_button_menu) +',\n' + \
                                 'but got ' + str(type(self.handles[key])))

        return copy(self.settings)


class CameraDisplayWidget(QtGui.QWidget):
    def __init__(self, cameraIDs, frame_shape=(300, 300, 3)):
        '''
        cameraIDs - list - cameraID values to use for accessing individual displays
        '''
        super(CameraDisplayWidget, self).__init__()
        self.cameraIDs = cameraIDs
        self.blank_frame = np.uint8(np.random.random(frame_shape) * 255)
        view_pos = CameraDisplayWidget.assign_viewbox_locations(len(self.cameraIDs))
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        self.viewboxes = {}
        self.imageItems = {}
        for npos, cameraID in enumerate(self.cameraIDs):
            graphics_view = pyqtgraph.GraphicsView()
            graphics_layout = pyqtgraph.GraphicsLayout()
            graphics_layout.addLabel('Camera ' + str(cameraID), col=0, row=0)
            graphics_layout.setContentsMargins(0, 0, 0, 0)
            self.viewboxes[cameraID] = graphics_layout.addViewBox(col=0, row=1)
            self.viewboxes[cameraID].setAspectLocked(True)
            self.viewboxes[cameraID].invertY()
            self.imageItems[cameraID] = pyqtgraph.ImageItem()
            self.viewboxes[cameraID].addItem(self.imageItems[cameraID])
            self.update_frame(cameraID, self.blank_frame)
            graphics_view.setCentralItem(graphics_layout)
            layout.addWidget(graphics_view, view_pos['row'][npos], view_pos['col'][npos])

    @staticmethod
    def assign_viewbox_locations(n_cameras):
        # Compute the number of columns and rows necessary for number of cameras
        # with columns having the priority
        n_cols = 1
        n_rows = 1
        increase_next = 'cols'
        while n_rows * n_cols < n_cameras:
            if increase_next == 'cols':
                n_cols += 1
                increase_next = 'rows'
            elif increase_next == 'rows':
                n_rows += 1
                increase_next = 'cols'
        # Assign grid positions for each camera
        view_pos = {'col': [], 'row': []}
        for npos in range(n_cameras):
            view_pos['col'].append(int(np.mod(npos, n_cols)))
            view_pos['row'].append(int(np.floor(npos / float(n_cols))))

        return view_pos

    def update_frame(self, cameraID, frame):
        '''
        frame - numpy.ndarray - np.uint8 - (height, width, channels)
        '''
        frame = np.swapaxes(frame,0,1)
        self.imageItems[cameraID].setImage(frame)


class CameraSettingsGUI(QtGui.QWidget):
    def __init__(self, CameraSettingsApp):
        '''
        CameraSettingsApp - CameraSettingsApp instance
        '''
        super(CameraSettingsGUI, self).__init__()
        self.CameraSettingsApp = CameraSettingsApp
        # Initialize GUI window
        self.resize(1100, 750)
        self.setWindowTitle('Camera Settings')
        # Create top menu items
        self.loadButton = QtGui.QPushButton('Load')
        self.loadButton.clicked.connect(self.loadButton_click)
        self.saveButton = QtGui.QPushButton('Save')
        self.saveButton.clicked.connect(self.saveButton_click)
        self.applyButton = QtGui.QPushButton('Apply')
        self.applyButton.clicked.connect(self.applyButton_click)
        self.closeButton = QtGui.QPushButton('Close')
        self.closeButton.clicked.connect(self.closeButton_click)
        self.addCameraButton = QtGui.QPushButton('Add Camera')
        self.addCameraButton.clicked.connect(self.addCameraButton_click)
        self.initializeCamerasButton = QtGui.QPushButton('Init. Cameras')
        self.initializeCamerasButton.setCheckable(True)
        self.initializeCamerasButton.clicked.connect(self.initializeCamerasButton_click)
        self.testTrackingButton = QtGui.QPushButton('Test Tracking')
        self.testTrackingButton.clicked.connect(self.testTrackingButton_click)
        top_menu_widget = QtGui.QWidget()
        top_menu_layout = QtGui.QVBoxLayout(top_menu_widget)
        top_menu_layout.addWidget(self.loadButton)
        top_menu_layout.addWidget(self.saveButton)
        top_menu_layout.addWidget(self.applyButton)
        top_menu_layout.addWidget(self.closeButton)
        top_menu_layout.addWidget(self.addCameraButton)
        top_menu_layout.addWidget(self.initializeCamerasButton)
        top_menu_layout.addWidget(self.testTrackingButton)
        top_menu_frame = QtGui.QFrame()
        top_menu_frame.setLayout(top_menu_layout)
        top_menu_frame.setFrameStyle(3)
        # Create Camera Specific Settings menu items
        self.camera_specific_items = {}
        self.camera_specific_items_key_counter = 0
        camera_specific_settings_master_widget = QtGui.QWidget()
        self.camera_specific_settings_layout = QtGui.QVBoxLayout(camera_specific_settings_master_widget)
        # Create General Settings menu layout in a frame
        self.general_settings_menu_layout = QtGui.QVBoxLayout()
        general_settings_menu_frame = QtGui.QFrame()
        general_settings_menu_frame.setLayout(self.general_settings_menu_layout)
        general_settings_menu_frame.setFrameStyle(0)
        general_settings_scroll_area = QtGui.QScrollArea()
        general_settings_scroll_area.setWidget(general_settings_menu_frame)
        general_settings_scroll_area.setWidgetResizable(True)
        # Create General Settings menu scroll area
        camera_specific_settings_scroll_area = QtGui.QScrollArea()
        camera_specific_settings_scroll_area.setWidget(camera_specific_settings_master_widget)
        camera_specific_settings_scroll_area.setWidgetResizable(True)
        # Create Camera View Box
        self.camera_display_layout = QtGui.QVBoxLayout()
        camera_display_frame = QtGui.QFrame()
        camera_display_frame.setLayout(self.camera_display_layout)
        camera_display_frame.setFrameStyle(3)
        camera_display_frame.setMinimumSize(700,500)
        # Put all boxes into main window
        top_hbox = QtGui.QHBoxLayout()
        top_hbox.addWidget(top_menu_frame)
        top_hbox.addWidget(camera_specific_settings_scroll_area)
        bottom_hbox = QtGui.QHBoxLayout()
        bottom_hbox.addWidget(general_settings_scroll_area)
        bottom_hbox.addWidget(camera_display_frame)
        main_vbox = QtGui.QVBoxLayout()
        main_vbox.addItem(top_hbox)
        main_vbox.addItem(setDoubleBoxStretch(bottom_hbox, 1, 2))
        self.setLayout(setDoubleBoxStretch(main_vbox, 1, 2))
        # Load settings into settings_menus
        self.write_settings_into_GUI(self.CameraSettingsApp.general_settings, 
                                     self.CameraSettingsApp.general_settings_keyorder(),
                                     self.CameraSettingsApp.camera_specific_settings, 
                                     self.CameraSettingsApp.single_camera_specific_settings_keyorder())
        self.show()

    def add_camera_specific_settings(self, settings, keyorder):
        key = copy(self.camera_specific_items_key_counter)
        self.camera_specific_items_key_counter += 1
        self.camera_specific_items[key] = {}
        single_camera_settings_layout = QtGui.QHBoxLayout()
        single_camera_settings_layout.setContentsMargins(2,2,2,2)
        # Add all settings
        self.camera_specific_items[key]['settings_menu'] = settings_menu(settings, keyorder, 'h', 'v')
        single_camera_settings_layout.addWidget(self.camera_specific_items[key]['settings_menu'])
        # Add Buttons
        buttons_layout = QtGui.QGridLayout()
        buttons_layout.setContentsMargins(0,0,0,0)
        single_camera_settings_layout.addItem(buttons_layout)
        # Add button to remove this camera
        self.camera_specific_items[key]['removeButton'] = QtGui.QPushButton('Remove')
        self.camera_specific_items[key]['removeButton'].clicked.connect(lambda: self.remove_single_camera_settings(key))
        buttons_layout.addWidget(self.camera_specific_items[key]['removeButton'], 0, 0)
        # Add button to calibrate this camera
        self.camera_specific_items[key]['calibrationButton'] = QtGui.QPushButton('Calibrate')
        self.camera_specific_items[key]['calibrationButton'].clicked.connect(lambda: self.calibrationButton_click(key))
        self.camera_specific_items[key]['calibrationButton'].setEnabled(False)
        buttons_layout.addWidget(self.camera_specific_items[key]['calibrationButton'], 0, 1)
        # Add button to show calibration image for this camera
        self.camera_specific_items[key]['showCalibrationButton'] = QtGui.QPushButton('Show')
        self.camera_specific_items[key]['showCalibrationButton'].clicked.connect(lambda: self.showCalibrationButton_click(key))
        self.camera_specific_items[key]['showCalibrationButton'].setEnabled(False)
        self.camera_specific_items[key]['showCalibrationButton'].setCheckable(True)
        buttons_layout.addWidget(self.camera_specific_items[key]['showCalibrationButton'], 1, 0)
        # Add button to overlay calibration of this camera
        self.camera_specific_items[key]['calibrationOverlayButton'] = QtGui.QPushButton('Overlay')
        self.camera_specific_items[key]['calibrationOverlayButton'].clicked.connect(lambda: self.calibrationOverlayButton_click(key))
        self.camera_specific_items[key]['calibrationOverlayButton'].setEnabled(False)
        self.camera_specific_items[key]['calibrationOverlayButton'].setCheckable(True)
        buttons_layout.addWidget(self.camera_specific_items[key]['calibrationOverlayButton'], 1, 1)
        # Create frame for single camera_specific_settings
        single_camera_settings_frame = QtGui.QFrame()
        single_camera_settings_frame.setLayout(single_camera_settings_layout)
        single_camera_settings_frame.setFrameStyle(2)
        single_camera_settings_frame.setMaximumHeight(60)
        self.camera_specific_settings_layout.addWidget(single_camera_settings_frame)
        self.camera_specific_items[key]['frame_remover'] = single_camera_settings_frame.deleteLater

    def remove_single_camera_settings(self, key):
        # Remove camera specific settings from GUI
        self.camera_specific_items[key]['settings_menu'].close()
        self.camera_specific_items[key]['removeButton'].close()
        self.camera_specific_items[key]['calibrationButton'].close()
        self.camera_specific_items[key]['showCalibrationButton'].close()
        self.camera_specific_items[key]['calibrationOverlayButton'].close()
        self.camera_specific_items[key]['frame_remover']()
        del self.camera_specific_items[key]

    def clear_camera_specific_settings_in_GUI(self):
        for key in self.camera_specific_items.keys():
            self.remove_single_camera_settings(key)

    def get_cameraID_from_camera_specific_items(self, key):
        settings = self.camera_specific_items[key]['settings_menu'].get()
        return settings['ID']['value']

    def read_settings_from_GUI(self):
        general_settings = self.general_settings_menu.get()
        camera_specific_settings = {}
        cameraIDs = []
        for key in self.camera_specific_items.keys():
            settings = self.camera_specific_items[key]['settings_menu'].get()
            cameraID = settings.pop('ID')['value']
            cameraIDs.append(cameraID)
            camera_specific_settings[cameraID] = settings
        if len(cameraIDs) != len(set(cameraIDs)):
            show_message('Duplicates not allowed in camera IDs.', 
                         'Currently set IDs: ' + str(sorted(cameraIDs)))
            raise ValueError('Duplicates not allowed in camera IDs.\n' + \
                             'Currently set IDs: ' + str(sorted(cameraIDs)))

        return general_settings, camera_specific_settings

    def write_settings_into_GUI(self, general_settings, general_settings_keyorder, 
                                camera_specific_settings, single_camera_specific_settings_keyorder):
        if hasattr(self, 'general_settings_menu'):
            self.general_settings_menu.deleteLater()
            del self.general_settings_menu
        self.general_settings_menu = settings_menu(general_settings, general_settings_keyorder, 'v', 'h')
        self.general_settings_menu_layout.addWidget(self.general_settings_menu)
        self.clear_camera_specific_settings_in_GUI()
        keyorder = self.CameraSettingsApp.single_camera_specific_settings_keyorder()
        keyorder = ['ID'] + keyorder
        for cameraID in sorted(camera_specific_settings.keys()):
            settings = copy(camera_specific_settings[cameraID])
            settings['ID'] = {'value': cameraID, 'description': 'ID'}
            self.add_camera_specific_settings(settings, keyorder)

    def update_settings_on_CameraSettingsApp(self):
        general_settings, camera_specific_settings = self.read_settings_from_GUI()
        self.CameraSettingsApp.update_settings(general_settings, camera_specific_settings)

    def loadButton_click(self):
        filename = openSingleFileDialog('load', suffix='nwb', caption='Select file')
        if not (filename is None):
            self.CameraSettingsApp.load_from_file(filename)
            print('Settings loaded.')

    def saveButton_click(self):
        self.update_settings_on_CameraSettingsApp()
        filename = openSingleFileDialog('save', suffix='nwb', caption='Save file name and location')
        if not (filename is None):
            self.CameraSettingsApp.save_to_file(filename)
            print('Settings saved.')

    def applyButton_click(self):
        self.update_settings_on_CameraSettingsApp()
        self.CameraSettingsApp.apply()

    def closeButton_click(self):
        self.CameraSettingsApp.close()

    def addCameraButton_click(self):
        default_cameraID = '0'
        settings = self.CameraSettingsApp.default_single_camera_specific_settings()
        keyorder = self.CameraSettingsApp.single_camera_specific_settings_keyorder()
        settings['ID'] = {'value': default_cameraID, 'description': 'ID'}
        keyorder = ['ID'] + keyorder
        self.add_camera_specific_settings(settings, keyorder)

    def initializeCamerasButton_click(self):
        # Disable or enable GUI interactions
        if not hasattr(self, 'initializeCamerasButtonDefaultStyleSheet'):
            self.initializeCamerasButtonDefaultStyleSheet = self.initializeCamerasButton.styleSheet()
        initializeCamerasButton_state = self.initializeCamerasButton.isChecked()
        self.initializeCamerasButton.setEnabled(False)
        self.addCameraButton.setEnabled(not initializeCamerasButton_state)
        self.general_settings_menu.setEnabled(not initializeCamerasButton_state)
        for key in self.camera_specific_items.keys():
            self.camera_specific_items[key]['settings_menu'].setEnabled(not initializeCamerasButton_state)
            self.camera_specific_items[key]['removeButton'].setEnabled(not initializeCamerasButton_state)
            self.camera_specific_items[key]['calibrationButton'].setEnabled(initializeCamerasButton_state)
            self.camera_specific_items[key]['showCalibrationButton'].setEnabled(initializeCamerasButton_state)
            self.camera_specific_items[key]['calibrationOverlayButton'].setEnabled(initializeCamerasButton_state)
        if initializeCamerasButton_state:
            self.initializeCamerasButton.setStyleSheet('background-color: red')
            # Initialize CameraDisplayWidget class based on camera_specific_settings
            general_settings, camera_specific_settings = self.read_settings_from_GUI()
            cameraIDs = camera_specific_settings.keys()
            self.CameraDisplayWidget = CameraDisplayWidget(sorted(cameraIDs))
            self.camera_display_layout.addWidget(self.CameraDisplayWidget)
            # Call camera_display_start() method on CameraSettingsApp after updating CameraSettings
            self.update_settings_on_CameraSettingsApp()
            self.main_worker = QThread_with_completion_callback(self.initializeCameras_finished_callback, 
                                                                self.CameraSettingsApp.initialize_cameras_and_streaming, 
                                                                function_args=(self.CameraDisplayWidget.update_frame,))
        else:
            # Close CameraDisplayWidget class
            self.CameraSettingsApp.stop_cameras_and_streaming()
            self.CameraDisplayWidget.close()
            self.initializeCamerasButton.setStyleSheet(self.initializeCamerasButtonDefaultStyleSheet)
            self.initializeCamerasButton.setEnabled(True)

    def initializeCameras_finished_callback(self):
        self.initializeCamerasButton.setEnabled(True)
        self.initializeCamerasButton.setStyleSheet('background-color: green')

    def calibrationButton_click(self, key):
        self.camera_specific_items[key]['calibrationButton'].setEnabled(False)
        if not hasattr(self, 'calibrationButtonDefaultStyleSheet'):
            self.calibrationButtonDefaultStyleSheet = self.camera_specific_items[key]['calibrationButton'].styleSheet()
        self.camera_specific_items[key]['calibrationButton'].setStyleSheet('background-color: red')
        cameraID = self.get_cameraID_from_camera_specific_items(key)
        self.main_worker = QThread_with_completion_callback(self.calibration_finished_callback, 
                                                            self.CameraSettingsApp.calibrate, 
                                                            return_output=True, 
                                                            callback_args=(key,), 
                                                            function_args=(cameraID,))


    def calibration_finished_callback(self, outcome, key):
        self.camera_specific_items[key]['calibrationButton'].setEnabled(True)
        if outcome == True:
            self.camera_specific_items[key]['calibrationButton'].setStyleSheet('background-color: green')
        else:
            self.camera_specific_items[key]['calibrationButton'].setStyleSheet(self.calibrationButtonDefaultStyleSheet)

    def showCalibrationButton_click(self, key):
        isChecked = self.camera_specific_items[key]['showCalibrationButton'].isChecked()
        cameraID = self.get_cameraID_from_camera_specific_items(key)
        Thread(target=self.CameraSettingsApp.show_calibration_frame, 
                   args=(cameraID, isChecked)).start()

    def calibrationOverlayButton_click(self, key):
        isChecked = self.camera_specific_items[key]['calibrationOverlayButton'].isChecked()
        cameraID = self.get_cameraID_from_camera_specific_items(key)
        Thread(target=self.CameraSettingsApp.overlay_calibration, 
                   args=(cameraID, isChecked)).start()

    def testTrackingButton_click(self):
        raise NotImplementedError


def main():
    app = QtGui.QApplication(sys.argv)
    camera_settings_application = CameraSettingsApp(with_GUI=True)
    app.exec_()
    
if __name__ == '__main__':
    main()
