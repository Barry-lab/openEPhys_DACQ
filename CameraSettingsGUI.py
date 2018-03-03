### This program handles the calibration and settings for Raspberry Pi Cameras
### that are used for tracking LED(s) during electrophysiological recordings.

### By Sander Tanni, May 2017, UCL

# How to add more cameras (RPis):
# Add more Cameras to the Individual Cameras list with QDesigner
# Add plots for new cameras to the f_plots with QDesigner
# Follow the objectName convention of previous items
# Add new objectNames to the lists in CameraSettings.__init__ GUI variables lines

# How to add more general camera settings:
# Add new edited copies of the frames to the General Settings list with QDesigner
# Include the new input fields in self.get_TrackingSettings(self) and self.load(self)

from PyQt4 import QtGui
import CameraSettingsGUIDesign
import sys
import os
from sshScripts import ssh
import cPickle as pickle
import pyqtgraph as pg
import numpy as np
from shutil import copyfile, rmtree
from PIL import Image
from HelperFunctions import openSingleFileDialog
from tempfile import mkdtemp
from RecordingManager import update_tracking_camera_files
import threading
from copy import deepcopy
import NWBio

def plotImage(im_view, image):
    # This function is used to display an image in any of the plots in the bottom
    # Add white padding to frame the image
    image = np.pad(image, [(1, 1), (1, 1), (0, 0)], mode='constant', constant_values=255)
    im_view.clear()
    view = im_view.addViewBox()
    view.setAspectLocked(True)
    im_item = pg.ImageItem()
    view.addItem(im_item)
    im_item.setImage(np.swapaxes(np.flipud(image),0,1))

def get_current_image(TrackingSettings, n_rpi, RPiImageTempFolder):
    # Use SSH connection to send commands
    connection = ssh(TrackingSettings['RPiInfo'][str(n_rpi)]['IP'], TrackingSettings['username'], TrackingSettings['password'])
    # Run getImage.py on RPi to capture a frame
    com_str = 'cd ' + TrackingSettings['tracking_folder'] + ' && python getImage.py'
    connection.sendCommand(com_str)
    connection.disconnect()
    # Copy over output files to local TEMP folder
    src_file = TrackingSettings['username'] + '@' + TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
               TrackingSettings['tracking_folder'] + '/frame.jpg'
    dst_file = os.path.join(RPiImageTempFolder, 'frame' + str(n_rpi) + '.jpg')
    callstr = 'scp -q ' + src_file + ' ' + dst_file
    _ = os.system(callstr)

def calibrate_camera(TrackingSettings, n_rpi, RPiCalibrationTempFolder):
    # Use SSH connection to send commands
    connection = ssh(TrackingSettings['RPiInfo'][str(n_rpi)]['IP'], TrackingSettings['username'], TrackingSettings['password'])
    # Run calibrate.py on RPi
    com_str = 'cd ' + TrackingSettings['tracking_folder'] + ' && python calibrate.py'
    connection.sendCommand(com_str)
    connection.disconnect()
    # Copy over output files to local TEMP folder
    src_file = TrackingSettings['username'] + '@' + TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
               TrackingSettings['tracking_folder'] + '/calibrationData.p'
    dst_file = os.path.join(RPiCalibrationTempFolder, 'calibrationData' + str(n_rpi) + '.p')
    callstr = 'scp -q ' + src_file + ' ' + dst_file
    _ = os.system(callstr)

def get_overlay_on_current_image(TrackingSettings, n_rpi, RPiImageTempFolder):
    # Use SSH connection to send commands
    connection = ssh(TrackingSettings['RPiInfo'][str(n_rpi)]['IP'], TrackingSettings['username'], TrackingSettings['password'])
    # Run getImage.py on RPi to capture a frame
    com_str = 'cd ' + TrackingSettings['tracking_folder'] + ' && python calibrate.py overlay'
    connection.sendCommand(com_str)
    connection.disconnect()
    # Copy over output files to local TEMP folder
    src_file = TrackingSettings['username'] + '@' + TrackingSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
               TrackingSettings['tracking_folder'] + '/overlay.jpg'
    dst_file = os.path.join(RPiImageTempFolder, 'overlay' + str(n_rpi) + '.jpg')
    callstr = 'scp -q ' + src_file + ' ' + dst_file
    _ = os.system(callstr)


class CameraSettings(QtGui.QMainWindow, CameraSettingsGUIDesign.Ui_MainWindow):
    def __init__(self, parent=None):
        super(CameraSettings, self).__init__(parent=parent)
        self.setupUi(self)
        self.parent = parent
        # Set up GUI variables
        self.trackingFolder = '/home/pi/Tracking'
        self.pt_rpi_ips = [self.pt_rpi_ip_1, self.pt_rpi_ip_2, self.pt_rpi_ip_3, self.pt_rpi_ip_4]
        self.cb_rpis = [self.cb_rpi_1, self.cb_rpi_2, self.cb_rpi_3, self.cb_rpi_4]
        self.pt_rpi_loc = [self.pt_rpi_loc_1, self.pt_rpi_loc_2, self.pt_rpi_loc_3, self.pt_rpi_loc_4]
        self.im_views = [self.im_view_1, self.im_view_2, self.im_view_3, self.im_view_4]
        self.calibrationData = {}
        # Set GUI interaction connections
        self.pb_show_image.clicked.connect(lambda:self.show_image())
        self.pb_calibrate.clicked.connect(lambda:self.calibrate())
        self.pb_show_calibration.clicked.connect(lambda:self.show_calibration())
        self.pb_overlay_calibration.clicked.connect(lambda:self.overlay())
        self.pb_test_tracking.clicked.connect(lambda:self.test_tracking())
        self.pb_load.clicked.connect(lambda:self.load())
        self.pb_save.clicked.connect(lambda:self.save())
        self.pb_apply.clicked.connect(lambda:self.apply())
        self.pb_cancel.clicked.connect(lambda:self.cancel())
        # Initialize Exposure Setting list
        itemstrings = ['off', 'auto', 'night', 'nightpreview', 'backlight', 'spotlight', 'sports', \
                       'snow', 'beach', 'verylong', 'fixedfps', 'antishake', 'fireworks']
        self.lw_exposure_settings.addItems(itemstrings)
        self.lw_exposure_settings.setCurrentRow(1)

    def get_TrackingSettings(self):
        # Check which LED option is checked
        if self.rb_led_single.isChecked():
            LEDmode = 'single'
        elif self.rb_led_double.isChecked():
            LEDmode = 'double'
        # Check if saving images is requested
        if self.rb_save_im_yes.isChecked():
            save_frames = True
        elif self.rb_save_im_no.isChecked():
            save_frames = False
        # Put camera settings from GUI to a dictionary
        use_RPi_Bool = np.array([0] * len(self.cb_rpis), dtype=bool)
        RPiInfo = {}
        for n_rpi in range(len(self.cb_rpis)):
            use_RPi_Bool[n_rpi] = self.cb_rpis[n_rpi].isChecked()
            RPiInfo[str(n_rpi)] = {'IP': str(self.pt_rpi_ips[n_rpi].toPlainText()), 
                                   'active': np.array(self.cb_rpis[n_rpi].isChecked()), 
                                   'location': np.array(map(float, str(self.pt_rpi_loc[n_rpi].toPlainText()).split(',')), dtype=np.float64)}
        use_RPi_nrs = np.arange(len(self.cb_rpis))[use_RPi_Bool]
        TrackingSettings = {'LEDmode': LEDmode, 
                            'save_frames': np.array(save_frames), 
                            'arena_size': np.array([str(self.parent.pt_arena_size_x.toPlainText()), str(self.parent.pt_arena_size_y.toPlainText())], dtype=np.float64), 
                            'calibration_n_dots': np.array([str(self.pt_ndots_x.toPlainText()), str(self.pt_ndots_y.toPlainText())], dtype=np.int64), 
                            'corner_offset': np.array([str(self.pt_offset_x.toPlainText()), str(self.pt_offset_y.toPlainText())], dtype=np.float64), 
                            'calibration_spacing': np.float64(str(self.pt_calibration_spacing.toPlainText())), 
                            'camera_iso': np.int64(str(self.pt_camera_iso.toPlainText())), 
                            'LED_separation': np.float64(str(self.pt_LED_separation.toPlainText())), 
                            'LED_angle': np.float64(str(self.pt_LED_angle.toPlainText())), 
                            'camera_transfer_radius': np.float64(str(self.pt_camera_transfer_radius.toPlainText())), 
                            'shutter_speed': np.int64(str(self.pt_shutter_speed.toPlainText())), 
                            'exposure_setting': str(self.lw_exposure_settings.currentItem().text()), 
                            'exposure_settings_selection': np.int64(self.lw_exposure_settings.currentRow()), 
                            'smoothing_radius': np.int64(str(self.pt_smooth_r.toPlainText())), 
                            'resolution': np.array(map(int, str(self.pt_resolution.toPlainText()).split(',')), dtype=np.int64), 
                            'centralIP': str(self.pt_local_ip.toPlainText()), 
                            'password': str(self.pt_rpi_password.toPlainText()), 
                            'username': str(self.pt_rpi_username.toPlainText()), 
                            'pos_port': str(self.pt_posport.toPlainText()), 
                            'stop_port': str(self.pt_stopport.toPlainText()), 
                            'RPiInfo': RPiInfo, 
                            'use_RPi_nrs': use_RPi_nrs, 
                            'tracking_folder': self.trackingFolder, 
                            'calibrationData': self.calibrationData}

        return TrackingSettings

    def load(self,TrackingSettings=None):
        if TrackingSettings is None:
            # Load TrackingSettings
            filename = openSingleFileDialog('load', suffix='nwb', caption='Select file')
            TrackingSettings = NWBio.load_settings(filename, '/TrackingSettings/')
        # Set current calibration data
        self.calibrationData = TrackingSettings['calibrationData']
        # Put TrackingSettings to GUI
        if TrackingSettings['LEDmode'] == 'single':
            self.rb_led_single.setChecked(True)
        elif TrackingSettings['LEDmode'] == 'double':
            self.rb_led_double.setChecked(True)
        if TrackingSettings['save_frames']:
            self.rb_save_im_yes.setChecked(True)
        elif not TrackingSettings['save_frames']:
            self.rb_save_im_no.setChecked(True)
        self.parent.pt_arena_size_x.setPlainText(str(TrackingSettings['arena_size'][0]))
        self.parent.pt_arena_size_y.setPlainText(str(TrackingSettings['arena_size'][1]))
        self.pt_ndots_x.setPlainText(str(TrackingSettings['calibration_n_dots'][0]))
        self.pt_ndots_y.setPlainText(str(TrackingSettings['calibration_n_dots'][1]))
        self.pt_offset_x.setPlainText(str(TrackingSettings['corner_offset'][0]))
        self.pt_offset_y.setPlainText(str(TrackingSettings['corner_offset'][1]))
        self.pt_calibration_spacing.setPlainText(str(TrackingSettings['calibration_spacing']))
        self.pt_smooth_r.setPlainText(str(TrackingSettings['smoothing_radius']))
        self.pt_LED_separation.setPlainText(str(TrackingSettings['LED_separation']))
        self.pt_LED_angle.setPlainText(str(TrackingSettings['LED_angle']))
        self.pt_camera_transfer_radius.setPlainText(str(TrackingSettings['camera_transfer_radius']))
        self.pt_camera_iso.setPlainText(str(TrackingSettings['camera_iso']))
        self.pt_shutter_speed.setPlainText(str(TrackingSettings['shutter_speed']))
        self.lw_exposure_settings.setCurrentRow(TrackingSettings['exposure_settings_selection'])
        CamRes = TrackingSettings['resolution']
        CamResStr = str(CamRes[0]) + ', ' + str(CamRes[1])
        self.pt_resolution.setPlainText(CamResStr)
        self.pt_local_ip.setPlainText(TrackingSettings['centralIP'])
        self.pt_rpi_password.setPlainText(TrackingSettings['password'])
        self.pt_rpi_username.setPlainText(TrackingSettings['username'])
        self.pt_posport.setPlainText(TrackingSettings['pos_port'])
        self.pt_stopport.setPlainText(TrackingSettings['stop_port'])
        for n_rpi in range(len(self.cb_rpis)):
            self.pt_rpi_ips[n_rpi].setPlainText(TrackingSettings['RPiInfo'][str(n_rpi)]['IP'])
            self.cb_rpis[n_rpi].setChecked(TrackingSettings['RPiInfo'][str(n_rpi)]['active'])
            self.pt_rpi_loc[n_rpi].setPlainText(','.join(map(str,TrackingSettings['RPiInfo'][str(n_rpi)]['location'])))
        self.trackingFolder = TrackingSettings['tracking_folder']

    def save(self):
        TrackingSettings = self.get_TrackingSettings()
        filename = openSingleFileDialog('save', suffix='nwb', caption='Save file name and location')
        NWBio.save_settings(filename, TrackingSettings, path='/TrackingSettings/')
        print('Settings saved.')

    def apply(self):
        TrackingSettings = self.get_TrackingSettings()
        update_tracking_camera_files(TrackingSettings)
        self.parent.Settings['TrackingSettings'] = deepcopy(TrackingSettings)
        self.close()

    def cancel(self):
        self.close()

    def show_image(self):
        TrackingSettings = self.get_TrackingSettings()
        update_tracking_camera_files(TrackingSettings)
        if len(TrackingSettings['use_RPi_nrs']) == 0:
            print('No cameras selected')
        else:
            print('Getting images ...')
            # Acquire current image from all tracking RPis
            RPiImageTempFolder = mkdtemp('RPiImageTempFolder')
            T_getRPiImage = []
            for n_rpi in TrackingSettings['use_RPi_nrs']:
                T = threading.Thread(target=get_current_image, args=[TrackingSettings, n_rpi, RPiImageTempFolder])
                T.start()
                T_getRPiImage.append(T)
            for T in T_getRPiImage:
                T.join()
            # Plot current frame for each RPi
            for n_rpi in TrackingSettings['use_RPi_nrs']:
                image = Image.open(os.path.join(RPiImageTempFolder, 'frame' + str(n_rpi) + '.jpg'))
                plotImage(self.im_views[n_rpi], image)
            rmtree(RPiImageTempFolder)
            print('Images displayed.')

    def calibrate(self):
        TrackingSettings = self.get_TrackingSettings()
        update_tracking_camera_files(TrackingSettings)
        if len(TrackingSettings['use_RPi_nrs']) == 0:
            print('No cameras selected')
        else:
            print('Calibrating cameras ...')
            # Get calibration data from all cameras
            RPiCalibrationTempFolder = mkdtemp('RPiCalibrationTempFolder')
            T_calibrateRPi = []
            for n_rpi in TrackingSettings['use_RPi_nrs']:
                T = threading.Thread(target=calibrate_camera, args=[TrackingSettings, n_rpi, RPiCalibrationTempFolder])
                T.start()
                T_calibrateRPi.append(T)
            for T in T_calibrateRPi:
                T.join()
            # Load calibration data
            for n_rpi in TrackingSettings['use_RPi_nrs']:
                with open(os.path.join(RPiCalibrationTempFolder, 'calibrationData' + str(n_rpi) + '.p'), 'rb') as file:
                    self.calibrationData[str(n_rpi)] = pickle.load(file)
            # Delete temporary folder
            rmtree(RPiCalibrationTempFolder)
            # Show calibration data
            self.show_calibration()

    def show_calibration(self):
        # Loads current calibrationData and shows it in the plots
        TrackingSettings = self.get_TrackingSettings()
        if len(TrackingSettings['use_RPi_nrs']) == 0:
            print('No cameras selected')
        else:
            for n_rpi in TrackingSettings['use_RPi_nrs']:
                if str(n_rpi) in TrackingSettings['calibrationData'].keys():
                    image = TrackingSettings['calibrationData'][str(n_rpi)]['image']
                else:
                    image = np.zeros((608,800,3), dtype=np.uint8)
                    image[:,:,0] = 255
                plotImage(self.im_views[n_rpi], image)

    def overlay(self):
        # Captures current image and overlays on it the currently active calibration chessboard corner pattern.
        TrackingSettings = self.get_TrackingSettings()
        update_tracking_camera_files(TrackingSettings)
        if len(TrackingSettings['use_RPi_nrs']) == 0:
            print('No cameras selected')
        else:
            print('Getting calibration overlay images ...')
            # Acquire current image from all tracking RPis
            RPiImageTempFolder = mkdtemp('RPiImageTempFolder')
            T_getRPiImage = []
            for n_rpi in TrackingSettings['use_RPi_nrs']:
                T = threading.Thread(target=get_overlay_on_current_image, args=[TrackingSettings, n_rpi, RPiImageTempFolder])
                T.start()
                T_getRPiImage.append(T)
            for T in T_getRPiImage:
                T.join()
            # Plot current image with overlay for each RPi
            for n_rpi in TrackingSettings['use_RPi_nrs']:
                image = Image.open(os.path.join(RPiImageTempFolder, 'overlay' + str(n_rpi) + '.jpg'))
                plotImage(self.im_views[n_rpi], image)
            print('Calibration overlay displayed.')

    def test_tracking(self):
        # Opens up a new window for displaying the primary LED positions as detected
        # and error of each RPi from the mean of RPis
        from PyQt4.QtCore import QTimer
        import RPiInterface as rpiI
        from scipy.spatial.distance import euclidean

        def stop_test_tracking(self):
            # Stops the tracking process and closes the test_tracking window
            self.tracking_timer.stop()
            self.RPIpos.close()
            self.test_tracking_win.close()

        def update_position_text(self):
            # Updates the data in the textboxes
            with self.RPIpos.posDatasLock:
                posDatas = self.RPIpos.posDatas # Retrieve latest position data
            # Get Position values of all RPis and update text boxes
            positions = np.zeros((len(posDatas), 2), dtype=np.float32)
            for nRPi in range(len(posDatas)):
                if posDatas[nRPi]:
                    positions[nRPi, 0] = posDatas[nRPi][3]
                    positions[nRPi, 1] = posDatas[nRPi][4]
                    # Update the text boxes for this RPi
                    self.pt_RPinr[nRPi].setText('%d' % posDatas[nRPi][0])
                    self.pt_posX[nRPi].setText('%.1f' % positions[nRPi, 0])
                    self.pt_posY[nRPi].setText('%.1f' % positions[nRPi, 1])
                else:
                    positions[nRPi, 0] = None
                    positions[nRPi, 1] = None
            # Compute error from mean of all RPis and insert in text box
            if not np.any(np.isnan(positions)):
                for nRPi in range(len(posDatas)):
                    distance = euclidean(np.mean(positions, axis=0), positions[nRPi, :])
                    self.pt_poserror[nRPi].setText('%.1f' % distance)

        # Get RPi Settings
        TrackingSettings = self.get_TrackingSettings()
        # Set up dialog box
        self.test_tracking_win = QtGui.QDialog()
        self.test_tracking_win.setWindowTitle('Test Tracking')
        vbox = QtGui.QVBoxLayout()
        # Add box titles for columns
        hbox_titles = QtGui.QHBoxLayout()
        pt_test_tracking_1 = QtGui.QLineEdit('RPi nr')
        pt_test_tracking_1.setReadOnly(True)
        hbox_titles.addWidget(pt_test_tracking_1)
        pt_test_tracking_2 = QtGui.QLineEdit('pos X')
        pt_test_tracking_2.setReadOnly(True)
        hbox_titles.addWidget(pt_test_tracking_2)
        pt_test_tracking_3 = QtGui.QLineEdit('pox Y')
        pt_test_tracking_3.setReadOnly(True)
        hbox_titles.addWidget(pt_test_tracking_3)
        pt_test_tracking_4 = QtGui.QLineEdit('error')
        pt_test_tracking_4.setReadOnly(True)
        hbox_titles.addWidget(pt_test_tracking_4)
        vbox.addLayout(hbox_titles)
        # Add rows for all RPis
        self.pt_RPinr = []
        self.pt_posX = []
        self.pt_posY = []
        self.pt_poserror = []
        hbox_pos = []
        for nRPi in range(len(TrackingSettings['use_RPi_nrs'])):
            hbox_pos.append(QtGui.QHBoxLayout())
            # Add RPi nr box
            self.pt_RPinr.append(QtGui.QLineEdit())
            self.pt_RPinr[nRPi].setReadOnly(True)
            hbox_pos[nRPi].addWidget(self.pt_RPinr[nRPi])
            # Add pos X box
            self.pt_posX.append(QtGui.QLineEdit())
            self.pt_posX[nRPi].setReadOnly(True)
            hbox_pos[nRPi].addWidget(self.pt_posX[nRPi])
            # Add pos Y box
            self.pt_posY.append(QtGui.QLineEdit())
            self.pt_posY[nRPi].setReadOnly(True)
            hbox_pos[nRPi].addWidget(self.pt_posY[nRPi])
            # Add error box
            self.pt_poserror.append(QtGui.QLineEdit())
            self.pt_poserror[nRPi].setReadOnly(True)
            hbox_pos[nRPi].addWidget(self.pt_poserror[nRPi])
            vbox.addLayout(hbox_pos[nRPi])
        # Add stop button
        self.pb_stop_tracking = QtGui.QPushButton('Stop Test')
        self.pb_stop_tracking.clicked.connect(lambda:stop_test_tracking(self))
        vbox.addWidget(self.pb_stop_tracking)
        # Finalise dialog box parameters
        self.test_tracking_win.setLayout(vbox)
        self.test_tracking_win.setGeometry(300, 200, 250, 20 * (len(TrackingSettings['use_RPi_nrs']) + 2))
        # Start the RPis
        update_tracking_camera_files(TrackingSettings)
        trackingControl = rpiI.TrackingControl(TrackingSettings)
        trackingControl.start()
        # Set up RPi latest position updater
        self.RPIpos = rpiI.onlineTrackingData(TrackingSettings)
        # Set up constant update of position fields with QTimer
        self.tracking_timer = QTimer()
        self.tracking_timer.timeout.connect(lambda:update_position_text(self))
        self.tracking_timer.start(33)
        # Open up the dialog window
        self.test_tracking_win.exec_()
        # When dialog window closes, stop the RPis
        trackingControl.stop()

# The following is the default ending for a QtGui application script
def main():
    app = QtGui.QApplication(sys.argv)
    form = CameraSettings()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main()