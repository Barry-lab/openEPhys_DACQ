
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from functools import partial
import numpy as np
from copy import copy
from openEPhys_DACQ import NWBio
from openEPhys_DACQ.HelperFunctions import openSingleFileDialog, show_message, QThread_with_completion_callback
import pyqtgraph
from openEPhys_DACQ.RPiInterface import CameraControl
from io import BytesIO
from socket import socket
import struct
from PIL import Image
from threading import Thread, Lock
import cv2


class CameraStreamCapture(object):
    """
    Listens to MPJEG stream from camera and prepares images for display.
    """
    def __init__(self, stream_port, update_frame, cameraID):
        """
        stream_port - int - port at 0.0.0.0 at which to listen to the MJPEG stream
        update_frame - is called with after receiving each frame with cameraID and prepared frame
        cameraID - key for images_for_display and image_for_display_Locks correct value
        """
        self.update_frame = update_frame
        self.cameraID = cameraID
        # Set internal variables
        self.calibration_overlay_on = False
        self.replacement_frame = None
        self.prepare_frame_Lock = Lock()
        # Start thread for streaming
        self.stream_open = True
        self.T_stream = Thread(target=self._stream, args=(stream_port,))
        self.T_stream.start()

    def _overlay_calibration_on_frame(self, frame):
        """
        Adds overlay onto the frame.
        """
        pattern = self.calibration_pattern
        ndots_x = self.calibration_ndots_xy[0]
        ndots_y = self.calibration_ndots_xy[1]
        frame = cv2.drawChessboardCorners(frame, (ndots_x, ndots_y), pattern, True)

        return frame

    def _prepare_frame(self, image_stream):
        """
        Prepares frame from MJPEG stream as a numpy array or uses the replacement frame
        and adds overlay of calibration if requested.
        Calls update_frame provided at initialization with self.cameraID and prepared frame.
        """
        with self.prepare_frame_Lock:
            if self.replacement_frame is None:
                # Rewind the stream, open it as an image with PIL and convert to numpy
                image_stream.seek(0)
                image = Image.open(image_stream)
                frame = np.array(image)
            else:
                frame = copy(self.replacement_frame)
            # Overlay calibration pattern if requested
            if self.calibration_overlay_on:
                frame = self._overlay_calibration_on_frame(frame)
        # Update images_for_display
        self.update_frame(self.cameraID, frame)

    def _stream(self, port):
        """
        Listens to MJPEG stream at 0.0.0.0 specified port.
        Calls _prepare_frame() method with fileld BytesIO class for each received frame.
        Finishes if self.stream_open is False or stream is closed.
        """
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
                Thread(target=self._prepare_frame, args=(image_stream,)).start()
        except Exception as e:
            if not isinstance(e, struct.error):
                raise e

    def calibration_overlay(self, on_off, pattern=None, ndots_xy=None):
        """
        Sets calibration overlay.

        on_off - bool
        pattern - output from CameraRPiController.Calibrator.get_calibration_data
        ndots_xy - [int - dots in X axis, int - dots in Y axis]
        """
        with self.prepare_frame_Lock:
            self.calibration_pattern = pattern
            self.calibration_ndots_xy = ndots_xy
            self.calibration_overlay_on = on_off

    def use_replacement_frame(self, on_off, frame=None):
        """
        Sets replacement frame.
        on_off - bool
        frame - replacement frame as numpy array - required if on_off=True
        """
        with self.prepare_frame_Lock:
            if on_off:
                self.replacement_frame = copy(frame)
            else:
                self.replacement_frame = None

    def close(self):
        self.stream_open = False
        self.T_stream.join()
        self.connection.close()
        self.server_socket.close()


class CameraSettingsApp(object):
    """
    Handles camera settings, calibration and provides live stream of camera feed.

    Note! CameraSettingsApp requires a specific workflow for calibration data to match the settings. 
          All settings must be specified before any other methods are called.
    """
    def __init__(self, apply_settings_method, camera_settings=None):
        """
        :param apply_settings_method: this called by :py:func:`CameraSettingsApp.apply` with camera_settings as arg
        :param dict camera_settings: camera_settings to initialise the applicaiton with. Otherwise defaults are used.
        """
        # Default variables
        self.display_stream_ports_start = 8000
        # Create internal variables
        self.CameraControllers = {}
        self.CameraStreamCaptures = {}
        self.CalibrationData = {}
        # Initialize
        self.apply_settings_method = apply_settings_method
        self.general_settings = self.default_settings('general_settings')
        self.camera_specific_settings = self.default_settings('camera_specific_settings')
        if not (camera_settings is None):
            self.import_settings(camera_settings['General'], camera_settings['CameraSpecific'])

    @staticmethod
    def _default_general_settings():
        """
        Specifies default settigs as well as data types.
        """
        return {'tracking_mode':          {'value': 'dual_led', 
                                           'description': ('Tracking mode', ('1 LED', 'single_led'), 
                                                                            ('2 LED', 'dual_led'), 
                                                                            ('Motion', 'motion'))}, 
                'resolution_option':      {'value': 'low', 
                                           'description': ('Video resolution', ('Low', 'low'), ('High', 'high'))}, 
                'LED_separation':         {'value': np.float64(5.0), 
                                           'description': 'LED separation (cm)'}, 
                'LED_angle':              {'value': np.float64(0.0), 
                                           'description': 'LED angle'}, 
                'camera_transfer_radius': {'value': np.float64(40.0), 
                                           'description': 'Camera transfer radius (cm)'}, 
                'smoothing_box':          {'value': np.int64(5), 
                                           'description': 'OnlineTracker smoothing (pix)'}, 
                'motion_threshold':       {'value': np.int64(10), 
                                           'description': '[Motion] Detection threshold (0-255)'}, 
                'motion_size':            {'value': np.int64(500), 
                                           'description': '[Motion] Pixel count threshold'}, 
                'framerate':              {'value': np.int64(30), 
                                           'description': 'Frame rate (Hz)'}, 
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
        """
        Specifies default settigs as well as data types.
        """
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
            cameraID = list(self.camera_specific_settings.keys())[0]
            settings = self.camera_specific_settings[cameraID]
        if not (settings is None):
            return sorted(settings.keys())

    def default_settings(self, settings_name):
        if settings_name == 'general_settings':
            return CameraSettingsApp._default_general_settings()
        elif settings_name == 'camera_specific_settings':
            return CameraSettingsApp._default_camera_specific_settings()

    def default_single_camera_specific_settings(self):
        """
        Returns settings and keyorder of the first cameraID
        """
        camera_specific_settings = self.default_settings('camera_specific_settings')
        settings = camera_specific_settings[list(camera_specific_settings.keys())[0]]
        keyorder = self.single_camera_specific_settings_keyorder()

        return settings

    def update_settings(self, general_settings, camera_specific_settings):
        self.general_settings = general_settings
        self.camera_specific_settings = camera_specific_settings

    def import_settings(self, general_settings, camera_specific_settings):
        """
        Creates new settings dictionaries by copying values into default settings.
        """
        new_general_settings = self.default_settings('general_settings')
        for key in new_general_settings.keys():
            if key in general_settings.keys():
                new_general_settings[key]['value'] = general_settings[key]
        new_camera_specific_settings = {}
        for cameraID in list(camera_specific_settings.keys()):
            if 'CalibrationData' in camera_specific_settings[cameraID].keys():
                self.CalibrationData[cameraID] = camera_specific_settings[cameraID].pop('CalibrationData')
            new_camera_specific_settings[cameraID] = self.default_single_camera_specific_settings()
            for key in new_camera_specific_settings[cameraID].keys():
                if key in camera_specific_settings[cameraID].keys():
                    new_camera_specific_settings[cameraID][key]['value'] = camera_specific_settings[cameraID][key]
        self.general_settings = new_general_settings
        self.camera_specific_settings = new_camera_specific_settings

    def export_settings(self):
        """
        Creates a simplified dictionary of settings that is compatible with import_settings() method.
        """
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

    def save_to_file(self, filename):
        NWBio.save_settings(filename, self.export_settings(), path='/CameraSettings/')

    def apply(self):
        """
        Calls the apply_settings_method provided at initialization with export_settings() method output.
        """
        self.apply_settings_method(self.export_settings())

    def _initialize_camera(self, cameraID):
        """
        Initializes control of a specific camera.
        """
        address = self.camera_specific_settings[cameraID]['address']['value']
        port = self.general_settings['ZMQcomms_port']['value']
        username = self.general_settings['username']['value']
        password = self.general_settings['password']['value']
        resolution_option = 'low'
        framerate = 10
        self.CameraControllers[cameraID] = CameraControl(address, port, username, 
                                                         password, resolution_option, 
                                                         framerate=framerate)

    def initialize_cameras(self):
        """
        Initializes control of all cameras according to current camera_specific_settings.
        """
        self.CameraControllers = {}
        t__initialize_camera = []
        for cameraID in self.camera_specific_settings.keys():
            t = Thread(target=self._initialize_camera, args=(cameraID,))
            t.start()
            t__initialize_camera.append(t)
        for t in t__initialize_camera:
            t.join()

    def _close_camera(self, cameraID):
        """
        Closes specific camera controller.
        """
        controller = self.CameraControllers.pop(cameraID)
        controller.close()

    def close_cameras(self):
        """
        Closes all camera controllers.
        """
        t_close_camera = []
        for key in list(self.CameraControllers.keys()):
            t = Thread(target=self._close_camera, args=(key,))
            t.start()
            t_close_camera.append(t)
        for t in t_close_camera:
            t.join()
        del self.CameraControllers

    def ensure_cameras_are_closed(self):
        if hasattr(self, 'CameraControllers'):
            self.close_cameras()

    def camera_display_start(self, update_frame):
        """
        Starts CameraStreamCapture for all cameras.

        update_frame - is called with after receiving each frame with cameraID and prepared frame
        """
        # Produce separate ports for streaming from each camera
        display_stream_ports = {}
        for n_cam, cameraID in enumerate(self.camera_specific_settings.keys()):
            display_stream_ports[cameraID] = self.display_stream_ports_start + n_cam
        # Start CameraStreamCapture class for each camera
        for cameraID in self.camera_specific_settings.keys():
            self.CameraStreamCaptures[cameraID] = CameraStreamCapture(display_stream_ports[cameraID], 
                                                                      update_frame, cameraID)
        # Start streaming from all cameras
        for cameraID in self.camera_specific_settings.keys():
            self.CameraControllers[cameraID].start_streaming(self.general_settings['centralIP']['value'], 
                                                             display_stream_ports[cameraID])

    def camera_display_stop(self):
        """
        Stops CameraStreamCapture of all cameras.
        """
        for key in list(self.CameraStreamCaptures.keys()):
            controller = self.CameraStreamCaptures.pop(key)
            controller.close()

    def initialize_cameras_and_streaming(self, update_frame):
        """
        First initializes cameras and then starts CameraStreamCapture for all cameras.

        update_frame - is called with after receiving each frame with cameraID and prepared frame
        """
        self.initialize_cameras()
        self.camera_display_start(update_frame)

    def stop_cameras_and_streaming(self):
        """
        Stops camera controllers and then CameraStreamCapture of all cameras.
        """
        self.close_cameras()
        self.camera_display_stop()

    def calibrate(self, cameraID):
        """
        Erases any previous calibration data.

        Attempts to calibrate the camera with current settings.
        If successful, the data is stored in self.CalibrationData.

        Returns True if calibration successful, False otherwise.

        cameraID - specifies which camera to calibrate.
        """
        if cameraID in list(self.CalibrationData.keys()):
            _ = self.CalibrationData.pop(cameraID)
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
        """
        Sets calibration overlay if CalibrationData available for cameraID.
        """
        if (cameraID in self.CalibrationData.keys()) and (cameraID in self.CameraStreamCaptures.keys()):
            if on_off:
                pattern = self.CalibrationData[cameraID]['low']['pattern']
                ndots_xy = self.camera_specific_settings[cameraID]['ndots_xy']['value']
                self.CameraStreamCaptures[cameraID].calibration_overlay(True, pattern, ndots_xy)
            else:
                self.CameraStreamCaptures[cameraID].calibration_overlay(False)

    def show_calibration_frame(self, cameraID, on_off):
        """
        Sets live view replacement with calibration image, 
        if CalibrationData available for cameraID.
        """
        if (cameraID in self.CalibrationData.keys()) and (cameraID in self.CameraStreamCaptures.keys()):
            if on_off:
                frame = self.CalibrationData[cameraID]['low']['frame']
                self.CameraStreamCaptures[cameraID].use_replacement_frame(True, frame)
            else:
                self.CameraStreamCaptures[cameraID].use_replacement_frame(False)

    def close(self):
        self.ensure_cameras_are_closed()
        if hasattr(self, 'CameraStreamCaptures'):
            for cameraID in self.CameraStreamCaptures.keys():
                self.CameraStreamCaptures[cameraID].close()
        if hasattr(self, 'GUI'):
            self.GUI.close()


class radio_button_menu(QtWidgets.QWidget):

    def __init__(self, name_value_pairs):
        super(radio_button_menu, self).__init__()
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.buttons = {}
        for name, value in name_value_pairs:
            self.buttons[value] = QtWidgets.QRadioButton(name)
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


class SettingsMenu(QtWidgets.QWidget):
    """
    Widget with multiple editable settings.
    """
    def __init__(self, settings, keyorder, primary_orientation, secondary_orientation):
        """
        settings - dict - from CameraSettings.default_settings()
        keyorder - list - settings.keys() in preferred order
        primary_orientation - str - 'horizontal' or 'vertical' or 'h' or 'v'
                              orientation of the main layout
        secondary_orientation - str - 'horizontal' or 'vertical' or 'h' or 'v'
                                orientation of each individual settings layout
        """
        super(SettingsMenu, self).__init__()
        self.settings = settings
        # Create top layout
        if primary_orientation == 'horizontal' or primary_orientation == 'h':
            layout = QtWidgets.QHBoxLayout()
        elif primary_orientation == 'vertical' or primary_orientation == 'v':
            layout = QtWidgets.QVBoxLayout()
        else:
            raise ValueError('primary_orientation argument must be horizontal or vertical')
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)
        # Create settings option
        self.handles = {}
        for key in keyorder:
            if secondary_orientation == 'horizontal' or secondary_orientation == 'h':
                box = QtWidgets.QHBoxLayout()
            elif secondary_orientation == 'vertical' or secondary_orientation == 'v':
                box = QtWidgets.QVBoxLayout()
            else:
                raise ValueError('secondary_orientation argument must be horizontal or vertical')
            box.setContentsMargins(0,0,0,0)
            # Create settings option type based on settings[key]['description value']
            if isinstance(self.settings[key]['description'], str):
                box.addWidget(QtWidgets.QLabel(self.settings[key]['description']))
                self.handles[key] = QtWidgets.QLineEdit()
            elif isinstance(self.settings[key]['description'], tuple):
                box.addWidget(QtWidgets.QLabel(self.settings[key]['description'][0]))
                self.handles[key] = radio_button_menu(self.settings[key]['description'][1:])
            box.addWidget(self.handles[key])
            # Add individual setting to SettingsMenu layout in a frame
            frame = QtWidgets.QFrame()
            frame.setLayout(setDoubleBoxStretch(box, 2, 1))
            frame.setFrameStyle(0)
            layout.addWidget(frame)
        # Populate SettingsMenu values with data from settings
        self.put(self.settings)

    def put(self, settings):
        """
        Populates SettingsMenu in handles with data from settings.

        settings - dict - with structure corresponding to SettingsMenu() input.
        """
        for key in settings.keys():
            value = copy(settings[key]['value'])
            if isinstance(self.handles[key], QtWidgets.QLineEdit):
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
                raise ValueError('Expected ' + str(QtWidgets.QLineEdit) + ' or ' + str(radio_button_menu) +',\n' + \
                                 'but got ' + str(type(self.handles[key])))

    def get(self):
        """
        Returns settings values with values extracted from SettingsMenu in handles.
        """
        for key in self.settings.keys():
            if isinstance(self.handles[key], QtWidgets.QLineEdit):
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
                raise ValueError('Expected ' + str(QtWidgets.QLineEdit) + ' or ' + str(radio_button_menu) +',\n' + \
                                 'but got ' + str(type(self.handles[key])))

        return copy(self.settings)


class CameraDisplayWidget(QtWidgets.QWidget):
    """
    Widget that displays images from cameras in a grid.

    Use update_frame() method to update to make new frames available to be displayed.
    """
    def __init__(self, cameraIDs, fps=10):
        """
        cameraIDs - list - cameraID values to use for accessing individual displays
        fps - int - image update frequency
        """
        super(CameraDisplayWidget, self).__init__()
        self.cameraIDs = cameraIDs
        # Compute viewbox locations and set layout
        view_pos = CameraDisplayWidget.assign_viewbox_locations(len(self.cameraIDs))
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        # Prepare variables
        self.images = {}
        self.image_Locks = {}
        self.viewboxes = {}
        self.imageItems = {}
        self.timers = {}
        # Populate variables for each camera
        for npos, cameraID in enumerate(self.cameraIDs):
            # Create image and lock
            self.images[cameraID] = np.uint8(np.random.random((300, 300, 3)) * 255)
            self.image_Locks[cameraID] = Lock()
            # Create viewbox
            graphics_view = pyqtgraph.GraphicsView()
            graphics_layout = pyqtgraph.GraphicsLayout()
            graphics_layout.addLabel('Camera ' + str(cameraID), col=0, row=0)
            graphics_layout.setContentsMargins(0, 0, 0, 0)
            self.viewboxes[cameraID] = graphics_layout.addViewBox(col=0, row=1)
            self.viewboxes[cameraID].setAspectLocked(True)
            self.viewboxes[cameraID].invertY()
            self.imageItems[cameraID] = pyqtgraph.ImageItem()
            self.viewboxes[cameraID].addItem(self.imageItems[cameraID])
            graphics_view.setCentralItem(graphics_layout)
            layout.addWidget(graphics_view, view_pos['row'][npos], view_pos['col'][npos])
            # Start updating image in viewbox
            self.timers[cameraID] = QTimer()
            self.timers[cameraID].timeout.connect(partial(self._display_frame, cameraID))
            self.timers[cameraID].start(int(1000 / fps))

    @staticmethod
    def assign_viewbox_locations(n_cameras):
        """
        Returns the number of columns and rows necessary for number of cameras
        with columns having the priority.
        """
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
        """
        Updates the current latest frame for camera cameraID.

        frame - numpy array of shape (height, width, 3)
        """
        with self.image_Locks[cameraID]:
            self.images[cameraID] = frame

    def _display_frame(self, cameraID):
        """
        Uses self.images and self.image_Locks to acquire current image.
        Then displays it in the correct viewbox.
        """
        with self.image_Locks[cameraID]:
            frame = self.images[cameraID]
        frame = np.swapaxes(frame, 0, 1)
        self.imageItems[cameraID].setImage(frame)

    def close(self, *args, **kwargs):
        for cameraID in self.timers.keys():
            self.timers[cameraID].stop()
        super(CameraDisplayWidget, self).close(*args, **kwargs)



class CameraSettingsGUI(QtWidgets.QDialog):
    """
    GUI for CameraSettingsApp.
    """
    def __init__(self, CameraSettingsApp, parent=None):
        """
        :param CameraSettingsApp CameraSettingsApp:
        """
        super(CameraSettingsGUI, self).__init__(parent=parent)
        self.CameraSettingsApp = CameraSettingsApp
        # Initialize GUI window
        self.resize(1100, 750)
        self.setWindowTitle('Camera Settings')
        # Create top menu items
        self.loadButton = QtWidgets.QPushButton('Load')
        self.loadButton.clicked.connect(self.loadButton_click)
        self.saveButton = QtWidgets.QPushButton('Save')
        self.saveButton.clicked.connect(self.saveButton_click)
        self.apply_button = QtWidgets.QPushButton('Apply')
        self.apply_button.clicked.connect(self.apply_button_callback)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.cancel_button_callback)
        self.addCameraButton = QtWidgets.QPushButton('Add Camera')
        self.addCameraButton.clicked.connect(self.addCameraButton_click)
        self.initializeCamerasButton = QtWidgets.QPushButton('Init. Cameras')
        self.initializeCamerasButton.setCheckable(True)
        self.initializeCamerasButton.clicked.connect(self.initializeCamerasButton_click)
        self.testTrackingButton = QtWidgets.QPushButton('Test Tracking')
        self.testTrackingButton.clicked.connect(self.testTrackingButton_click)
        top_menu_widget = QtWidgets.QWidget()
        top_menu_layout = QtWidgets.QVBoxLayout(top_menu_widget)
        top_menu_layout.addWidget(self.loadButton)
        top_menu_layout.addWidget(self.saveButton)
        top_menu_layout.addWidget(self.apply_button)
        top_menu_layout.addWidget(self.cancel_button)
        top_menu_layout.addWidget(self.addCameraButton)
        top_menu_layout.addWidget(self.initializeCamerasButton)
        top_menu_layout.addWidget(self.testTrackingButton)
        top_menu_frame = QtWidgets.QFrame()
        top_menu_frame.setLayout(top_menu_layout)
        top_menu_frame.setFrameStyle(3)
        # Create Camera Specific Settings menu items
        self.camera_specific_items = {}
        self.camera_specific_items_key_counter = 0
        camera_specific_settings_master_widget = QtWidgets.QWidget()
        self.camera_specific_settings_layout = QtWidgets.QVBoxLayout(camera_specific_settings_master_widget)
        # Create General Settings menu layout in a frame
        self.general_SettingsMenu_layout = QtWidgets.QVBoxLayout()
        general_SettingsMenu_frame = QtWidgets.QFrame()
        general_SettingsMenu_frame.setLayout(self.general_SettingsMenu_layout)
        general_SettingsMenu_frame.setFrameStyle(0)
        general_settings_scroll_area = QtWidgets.QScrollArea()
        general_settings_scroll_area.setWidget(general_SettingsMenu_frame)
        general_settings_scroll_area.setWidgetResizable(True)
        # Create General Settings menu scroll area
        camera_specific_settings_scroll_area = QtWidgets.QScrollArea()
        camera_specific_settings_scroll_area.setWidget(camera_specific_settings_master_widget)
        camera_specific_settings_scroll_area.setWidgetResizable(True)
        # Create Camera View Box
        self.camera_display_layout = QtWidgets.QVBoxLayout()
        camera_display_frame = QtWidgets.QFrame()
        camera_display_frame.setLayout(self.camera_display_layout)
        camera_display_frame.setFrameStyle(3)
        camera_display_frame.setMinimumSize(700,500)
        # Put all boxes into main window
        top_hbox = QtWidgets.QHBoxLayout()
        top_hbox.addWidget(top_menu_frame)
        top_hbox.addWidget(camera_specific_settings_scroll_area)
        bottom_hbox = QtWidgets.QHBoxLayout()
        bottom_hbox.addWidget(general_settings_scroll_area)
        bottom_hbox.addWidget(camera_display_frame)
        main_vbox = QtWidgets.QVBoxLayout()
        main_vbox.addItem(top_hbox)
        main_vbox.addItem(setDoubleBoxStretch(bottom_hbox, 1, 2))
        self.setLayout(setDoubleBoxStretch(main_vbox, 1, 2))
        # Load settings into SettingsMenus
        self.write_settings_into_GUI(self.CameraSettingsApp.general_settings, 
                                     self.CameraSettingsApp.general_settings_keyorder(),
                                     self.CameraSettingsApp.camera_specific_settings, 
                                     self.CameraSettingsApp.single_camera_specific_settings_keyorder())
        self.show()

    def _add_camera_specific_settings(self, settings, keyorder):
        """
        Creates camera_specific_settings and adds it to the camera_specific_settings_layout.

        settings - as required by SettingsMenu class
        keyorder - as required by SettingsMenu class
        """
        key = copy(self.camera_specific_items_key_counter)
        self.camera_specific_items_key_counter += 1
        self.camera_specific_items[key] = {}
        single_camera_settings_layout = QtWidgets.QHBoxLayout()
        single_camera_settings_layout.setContentsMargins(2,2,2,2)
        # Add all settings
        self.camera_specific_items[key]['SettingsMenu'] = SettingsMenu(settings, keyorder, 'h', 'v')
        single_camera_settings_layout.addWidget(self.camera_specific_items[key]['SettingsMenu'])
        # Add Buttons
        buttons_layout = QtWidgets.QGridLayout()
        buttons_layout.setContentsMargins(0,0,0,0)
        single_camera_settings_layout.addItem(buttons_layout)
        # Add button to remove this camera
        self.camera_specific_items[key]['removeButton'] = QtWidgets.QPushButton('Remove')
        self.camera_specific_items[key]['removeButton'].clicked.connect(lambda: self._remove_single_camera_settings(key))
        buttons_layout.addWidget(self.camera_specific_items[key]['removeButton'], 0, 0)
        # Add button to calibrate this camera
        self.camera_specific_items[key]['calibrationButton'] = QtWidgets.QPushButton('Calibrate')
        self.camera_specific_items[key]['calibrationButton'].clicked.connect(lambda: self.calibrationButton_click(key))
        self.camera_specific_items[key]['calibrationButton'].setEnabled(False)
        buttons_layout.addWidget(self.camera_specific_items[key]['calibrationButton'], 0, 1)
        # Add button to show calibration image for this camera
        self.camera_specific_items[key]['showCalibrationButton'] = QtWidgets.QPushButton('Show')
        self.camera_specific_items[key]['showCalibrationButton'].clicked.connect(lambda: self.showCalibrationButton_click(key))
        self.camera_specific_items[key]['showCalibrationButton'].setEnabled(False)
        self.camera_specific_items[key]['showCalibrationButton'].setCheckable(True)
        buttons_layout.addWidget(self.camera_specific_items[key]['showCalibrationButton'], 1, 0)
        # Add button to overlay calibration of this camera
        self.camera_specific_items[key]['calibrationOverlayButton'] = QtWidgets.QPushButton('Overlay')
        self.camera_specific_items[key]['calibrationOverlayButton'].clicked.connect(lambda: self.calibrationOverlayButton_click(key))
        self.camera_specific_items[key]['calibrationOverlayButton'].setEnabled(False)
        self.camera_specific_items[key]['calibrationOverlayButton'].setCheckable(True)
        buttons_layout.addWidget(self.camera_specific_items[key]['calibrationOverlayButton'], 1, 1)
        # Create frame for single camera_specific_settings
        single_camera_settings_frame = QtWidgets.QFrame()
        single_camera_settings_frame.setLayout(single_camera_settings_layout)
        single_camera_settings_frame.setFrameStyle(2)
        single_camera_settings_frame.setMaximumHeight(60)
        self.camera_specific_settings_layout.addWidget(single_camera_settings_frame)
        self.camera_specific_items[key]['frame_remover'] = single_camera_settings_frame.deleteLater

    def _remove_single_camera_settings(self, key):
        """
        Removes single camera specific settings from GUI
        """
        self.camera_specific_items[key]['SettingsMenu'].close()
        self.camera_specific_items[key]['removeButton'].close()
        self.camera_specific_items[key]['calibrationButton'].close()
        self.camera_specific_items[key]['showCalibrationButton'].close()
        self.camera_specific_items[key]['calibrationOverlayButton'].close()
        self.camera_specific_items[key]['frame_remover']()
        del self.camera_specific_items[key]

    def _clear_camera_specific_settings_in_GUI(self):
        """
        Removes all camera specific settings from GUI
        """
        for key in list(self.camera_specific_items.keys()):
            self._remove_single_camera_settings(key)

    def _get_cameraID_from_camera_specific_items(self, key):
        """
        Returns edited cameraID from camera specific SettingsMenu
        """
        settings = self.camera_specific_items[key]['SettingsMenu'].get()
        return settings['ID']['value']

    def _read_settings_from_GUI(self):
        """
        Returns general and camera specific settings from SettingsMenus in format
        compatible with CameraSettingsApp.update_settings.
        """
        general_settings = self.general_SettingsMenu.get()
        camera_specific_settings = {}
        cameraIDs = []
        for key in self.camera_specific_items.keys():
            settings = self.camera_specific_items[key]['SettingsMenu'].get()
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
        """
        Updates general and camera specific settings in GUI

        general_settings - CameraSettingsApp.general_settings value
        general_settings_keyorder - display keyorder for general_settings
        camera_specific_settings - CameraSettingsApp.camera_specific_settings value
        single_camera_specific_settings_keyorder - display keyorder for camera_specific_settings
        """
        if hasattr(self, 'general_SettingsMenu'):
            self.general_SettingsMenu.deleteLater()
            del self.general_SettingsMenu
        self.general_SettingsMenu = SettingsMenu(general_settings, general_settings_keyorder, 'v', 'h')
        self.general_SettingsMenu_layout.addWidget(self.general_SettingsMenu)
        self._clear_camera_specific_settings_in_GUI()
        keyorder = self.CameraSettingsApp.single_camera_specific_settings_keyorder()
        keyorder = ['ID'] + keyorder
        for cameraID in sorted(camera_specific_settings.keys()):
            settings = copy(camera_specific_settings[cameraID])
            settings['ID'] = {'value': cameraID, 'description': 'ID'}
            self._add_camera_specific_settings(settings, keyorder)

    def _update_settings_on_CameraSettingsApp(self):
        """
        Calls CameraSettingsApp.update_settings with output from _read_settings_from_GUI() method.
        """
        general_settings, camera_specific_settings = self._read_settings_from_GUI()
        self.CameraSettingsApp.update_settings(general_settings, camera_specific_settings)

    def loadButton_click(self):
        """
        Attempts to load settings into CameraSettingsApp from selected file.
        """
        filename = openSingleFileDialog('load', suffix='nwb', caption='Select file')
        if not (filename is None):
            self.CameraSettingsApp.load_from_file(filename)
            self.write_settings_into_GUI(self.CameraSettingsApp.general_settings,
                                         self.CameraSettingsApp.general_settings_keyorder(),
                                         self.CameraSettingsApp.camera_specific_settings,
                                         self.CameraSettingsApp.single_camera_specific_settings_keyorder())

    def saveButton_click(self):
        """
        Attempts to save settings from CameraSettingsApp to selected file.
        """
        self._update_settings_on_CameraSettingsApp()
        filename = openSingleFileDialog('save', suffix='nwb', caption='Save file name and location')
        if not (filename is None):
            self.CameraSettingsApp.save_to_file(filename)

    def apply_button_callback(self):
        """
        Calls CameraSettingsApp.apply method.
        """
        self._update_settings_on_CameraSettingsApp()
        self.CameraSettingsApp.apply()
        self.CameraSettingsApp.close()
        self.close()

    def cancel_button_callback(self):
        """
        Calls CameraSettingsApp.close, which in turn calls the close() method in CameraSettingsGUI.
        """
        self.CameraSettingsApp.close()
        self.close()

    def addCameraButton_click(self):
        """
        Calls _add_camera_specific_settings() method with 
        default_single_camera_specific_settings from CameraSettingsApp.
        """
        default_cameraID = '0'
        settings = self.CameraSettingsApp.default_single_camera_specific_settings()
        keyorder = self.CameraSettingsApp.single_camera_specific_settings_keyorder()
        settings['ID'] = {'value': default_cameraID, 'description': 'ID'}
        keyorder = ['ID'] + keyorder
        self._add_camera_specific_settings(settings, keyorder)

    def initializeCamerasButton_click(self):
        """
        Enables and disables correct functions on GUI.
        If toggled on:
            Initializes CameraDisplayWidget
            calls _update_settings_on_CameraSettingsApp() method
            calls CameraSettingsApp.initialize_cameras_and_streaming
        If toggled off:
            calls CameraSettingsApp.stop_cameras_and_streaming
            calls CameraDisplayWidget.close
        """
        # Disable or enable GUI interactions
        if not hasattr(self, 'initializeCamerasButtonDefaultStyleSheet'):
            self.initializeCamerasButtonDefaultStyleSheet = self.initializeCamerasButton.styleSheet()
        initializeCamerasButton_state = self.initializeCamerasButton.isChecked()
        self.initializeCamerasButton.setEnabled(False)
        self.addCameraButton.setEnabled(not initializeCamerasButton_state)
        self.general_SettingsMenu.setEnabled(not initializeCamerasButton_state)
        for key in self.camera_specific_items.keys():
            self.camera_specific_items[key]['SettingsMenu'].setEnabled(not initializeCamerasButton_state)
            self.camera_specific_items[key]['removeButton'].setEnabled(not initializeCamerasButton_state)
            self.camera_specific_items[key]['calibrationButton'].setEnabled(initializeCamerasButton_state)
            self.camera_specific_items[key]['showCalibrationButton'].setEnabled(initializeCamerasButton_state)
            self.camera_specific_items[key]['calibrationOverlayButton'].setEnabled(initializeCamerasButton_state)
        if initializeCamerasButton_state:
            self.initializeCamerasButton.setStyleSheet('background-color: red')
            # Initialize CameraDisplayWidget class based on camera_specific_settings
            general_settings, camera_specific_settings = self._read_settings_from_GUI()
            cameraIDs = camera_specific_settings.keys()
            self.CameraDisplayWidget = CameraDisplayWidget(sorted(cameraIDs))
            self.camera_display_layout.addWidget(self.CameraDisplayWidget)
            # Call camera_display_start() method on CameraSettingsApp after updating CameraSettings
            self._update_settings_on_CameraSettingsApp()
            self.main_worker = QThread_with_completion_callback(self._initializeCameras_finished_callback, 
                                                                self.CameraSettingsApp.initialize_cameras_and_streaming, 
                                                                function_args=(self.CameraDisplayWidget.update_frame,))
        else:
            # Close CameraDisplayWidget class
            self.CameraSettingsApp.stop_cameras_and_streaming()
            self.CameraDisplayWidget.close()
            self.initializeCamerasButton.setStyleSheet(self.initializeCamerasButtonDefaultStyleSheet)
            self.initializeCamerasButton.setEnabled(True)

    def _initializeCameras_finished_callback(self):
        """
        Is called once CameraSettingsApp.initialize_cameras_and_streaming call is finished.
        Enables initializeCamerasButton and sets it green.
        """
        self.initializeCamerasButton.setEnabled(True)
        self.initializeCamerasButton.setStyleSheet('background-color: green')

    def calibrationButton_click(self, key):
        """
        Calls CameraSettingsApp.calibrate for the correct cameraID.
        """
        self.camera_specific_items[key]['calibrationButton'].setEnabled(False)
        if not hasattr(self, 'calibrationButtonDefaultStyleSheet'):
            self.calibrationButtonDefaultStyleSheet = self.camera_specific_items[key]['calibrationButton'].styleSheet()
        self.camera_specific_items[key]['calibrationButton'].setStyleSheet('background-color: red')
        cameraID = self._get_cameraID_from_camera_specific_items(key)
        self.main_worker = QThread_with_completion_callback(self.calibration_finished_callback, 
                                                            self.CameraSettingsApp.calibrate, 
                                                            return_output=True, 
                                                            callback_args=(key,), 
                                                            function_args=(cameraID,))


    def calibration_finished_callback(self, outcome, key):
        """
        Is called once CameraSettingsApp.calibrate call is finished.
        Enables calibrationButton and sets it green or default, depending on outcome.
        """
        self.camera_specific_items[key]['calibrationButton'].setEnabled(True)
        if outcome == True:
            self.camera_specific_items[key]['calibrationButton'].setStyleSheet('background-color: green')
        else:
            self.camera_specific_items[key]['calibrationButton'].setStyleSheet(self.calibrationButtonDefaultStyleSheet)

    def showCalibrationButton_click(self, key):
        """
        Calls CameraSettingsApp.show_calibration_frame.
        """
        isChecked = self.camera_specific_items[key]['showCalibrationButton'].isChecked()
        cameraID = self._get_cameraID_from_camera_specific_items(key)
        Thread(target=self.CameraSettingsApp.show_calibration_frame, 
                   args=(cameraID, isChecked)).start()

    def calibrationOverlayButton_click(self, key):
        """
        Calls CameraSettingsApp.overlay_calibration.
        """
        isChecked = self.camera_specific_items[key]['calibrationOverlayButton'].isChecked()
        cameraID = self._get_cameraID_from_camera_specific_items(key)
        Thread(target=self.CameraSettingsApp.overlay_calibration, 
                   args=(cameraID, isChecked)).start()

    def testTrackingButton_click(self):
        raise NotImplementedError
