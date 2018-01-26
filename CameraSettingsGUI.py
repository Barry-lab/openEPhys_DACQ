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
# Include the new input fields in self.get_camera_settings_dict(self) and self.load(self)

from PyQt4 import QtGui
import CameraSettingsGUIDesign
import sys
import os
from sshScripts import ssh
import cPickle as pickle
import pyqtgraph as pg
import numpy as np
from shutil import copyfile
from PIL import Image

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

def plotImage(im_view, image):
    # This function is used to display an image in any of the plots in the bottom
    im_view.clear()
    view = im_view.addViewBox()
    view.setAspectLocked(True)
    im_item = pg.ImageItem()
    view.addItem(im_item)
    im_item.setImage(np.swapaxes(np.flipud(image),0,1))

def Camera_Files_Update_Function(SettingsFolder, RPiSettings, useCalibration=True):
    # This function updates all RPis with information saved on disk and RPiSettings file
    for n_rpi in RPiSettings['use_RPi_nrs']:
        # Adjust RPiNumber
        with open(SettingsFolder + '/RPi/RPiNumber', 'wb') as file:
            file.write(str(n_rpi))
        # Include calibration files
        if useCalibration:
            calibrationFile = SettingsFolder + '/calibrationData' + str(n_rpi) + '.p'
            calibrationTmatrixFile = SettingsFolder + '/calibrationTmatrix' + str(n_rpi) + '.p'
            dst_calibrationFile = SettingsFolder + '/RPi/calibrationData.p'
            dst_calibrationTmatrixFile = SettingsFolder + '/RPi/calibrationTmatrix.p'
            if os.path.isfile(calibrationFile):
                copyfile(calibrationFile, dst_calibrationFile)
                copyfile(calibrationTmatrixFile, dst_calibrationTmatrixFile)
            else:
                show_message('Calibration file not found for RPi ' + str(n_rpi + 1) + '.')
        # Push files to RPi
        callstr = 'rsync -avrP -e "ssh -l server" ' + SettingsFolder + '/RPi/ ' + RPiSettings['username'] + '@' + RPiSettings['RPiIP'][n_rpi] + ':' + RPiSettings['tracking_folder'] + '/ --delete'
        os.system(callstr)

class CameraSettings(QtGui.QMainWindow, CameraSettingsGUIDesign.Ui_MainWindow):
    def __init__(self, parent=None):
        super(CameraSettings, self).__init__(parent=parent)
        self.setupUi(self)
        # Set up GUI variables
        self.trackingFolder = '/home/pi/Tracking'
        self.pt_rpi_ips = [self.pt_rpi_ip_1, self.pt_rpi_ip_2, self.pt_rpi_ip_3, self.pt_rpi_ip_4]
        self.cb_rpis = [self.cb_rpi_1, self.cb_rpi_2, self.cb_rpi_3, self.cb_rpi_4]
        self.pt_rpi_loc = [self.pt_rpi_loc_1, self.pt_rpi_loc_2, self.pt_rpi_loc_3, self.pt_rpi_loc_4]
        self.im_views = [self.im_view_1, self.im_view_2, self.im_view_3, self.im_view_4]
        # Set GUI interaction connections
        self.pb_show_image.clicked.connect(lambda:self.show_image())
        self.pb_calibrate.clicked.connect(lambda:self.calibrate())
        self.pb_show_calibration.clicked.connect(lambda:self.show_calibration())
        self.pb_overlay_calibration.clicked.connect(lambda:self.overlay())
        self.pb_test_tracking.clicked.connect(lambda:self.test_tracking())
        self.pb_load_last.clicked.connect(lambda:self.load_last())
        self.pb_load.clicked.connect(lambda:self.load())
        self.pb_save.clicked.connect(lambda:self.save())
        self.pb_apply.clicked.connect(lambda:self.apply())
        # Initialize Exposure Setting list
        itemstrings = ['off', 'auto', 'night', 'nightpreview', 'backlight', 'spotlight', 'sports', \
                       'snow', 'beach', 'verylong', 'fixedfps', 'antishake', 'fireworks']
        self.lw_exposure_settings.addItems(itemstrings)
        self.lw_exposure_settings.setCurrentRow(1)

    def openFolderDialog(self, caption, directory):
        # Pops up a GUI to select a folder
        dialog = QtGui.QFileDialog(self, caption, directory)
        dialog.setFileMode(QtGui.QFileDialog.Directory)
        dialog.setOption(QtGui.QFileDialog.ShowDirsOnly, True)
        if dialog.exec_():
            # Get paths of selected folder
            tmp = dialog.selectedFiles()
            selected_folder = str(tmp[0])
        else:
            selected_folder = 'No folder selected'

        return selected_folder

    def openSingleFileDialog(self, caption='Select a file', directory=None):
        # Pops up a GUI to select a single file. All others with same prefix will be loaded
        dialog = QtGui.QFileDialog(self, caption, directory)
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        dialog.setViewMode(QtGui.QFileDialog.List) # or Detail
        if dialog.exec_():
            # Get path and file name of selection
            tmp = dialog.selectedFiles()
            selected_file = str(tmp[0])

        return selected_file

    def get_camera_settings_dict(self):
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
        # Put resolution into integer format
        tmp = str(self.pt_resolution.toPlainText())
        CamResolution = [int(tmp[:tmp.find(',')]), int(tmp[tmp.find(',') + 1:])]
        # Put camera settings from GUI to a dictionary
        use_RPi_Bool = np.array([0] * len(self.cb_rpis), dtype=bool)
        RPiIP = []
        RPi_Usage = []
        RPi_location = []
        for n_rpi in range(len(self.cb_rpis)):
            RPiIP.append(str(self.pt_rpi_ips[n_rpi].toPlainText()))
            RPi_Usage.append(self.cb_rpis[n_rpi].isChecked())
            use_RPi_Bool[n_rpi] = self.cb_rpis[n_rpi].isChecked()
            RPi_location.append(str(self.pt_rpi_loc[n_rpi].toPlainText()))
        use_RPi_nrs = list(np.arange(len(self.cb_rpis))[use_RPi_Bool])
        RPiSettings = {'LEDmode': LEDmode, 
                       'save_frames': save_frames, 
                       'arena_size': [float(str(self.pt_arena_size_x.toPlainText())), float(str(self.pt_arena_size_y.toPlainText()))], 
                       'calibration_n_dots': [int(str(self.pt_ndots_x.toPlainText())), int(str(self.pt_ndots_y.toPlainText()))], 
                       'corner_offset': [float(str(self.pt_offset_x.toPlainText())), float(str(self.pt_offset_y.toPlainText()))], 
                       'calibration_spacing': float(str(self.pt_calibration_spacing.toPlainText())), 
                       'camera_iso': int(str(self.pt_camera_iso.toPlainText())), 
                       'LED_separation': float(str(self.pt_LED_separation.toPlainText())), 
                       'LED_angle': float(str(self.pt_LED_angle.toPlainText())), 
                       'camera_transfer_radius': float(str(self.pt_camera_transfer_radius.toPlainText())), 
                       'shutter_speed': int(str(self.pt_shutter_speed.toPlainText())), 
                       'exposure_setting': str(self.lw_exposure_settings.currentItem().text()), 
                       'exposure_settings_selection': self.lw_exposure_settings.currentRow(), 
                       'smoothing_radius': int(str(self.pt_smooth_r.toPlainText())), 
                       'resolution': CamResolution, 
                       'centralIP': str(self.pt_local_ip.toPlainText()), 
                       'password': str(self.pt_rpi_password.toPlainText()), 
                       'username': str(self.pt_rpi_username.toPlainText()), 
                       'pos_port': str(self.pt_posport.toPlainText()), 
                       'stop_port': str(self.pt_stopport.toPlainText()), 
                       'RPiIP': RPiIP, 
                       'RPi_Usage': RPi_Usage, 
                       'use_RPi_nrs': use_RPi_nrs, 
                       'RPi_location': RPi_location, 
                       'tracking_folder': self.trackingFolder}

        return RPiSettings

    def update_camera_files(self, useCalibration=True):
        # Recreate original RPi folder in the TEMP folder
        callstr = 'rsync -avzh ' + self.scripts_root + '/RaspberryPi/ ' + self.TEMPfolder + '/RPi/'
        os.system(callstr)
        RPiSettings = self.get_camera_settings_dict()
        # Save dictionary to RPi folder
        with open(self.TEMPfolder + '/RPi/RPiSettings.p','wb') as file:
            pickle.dump(RPiSettings, file)
        # Sync each RPi with the TEMP/RPi folder, including the correct RPiNumber file
        # and correct calibrationData if available
        Camera_Files_Update_Function(SettingsFolder=self.TEMPfolder, RPiSettings=RPiSettings, useCalibration=useCalibration)

    def show_image(self):
        self.update_camera_files(useCalibration = False)
        RPiSettings = self.get_camera_settings_dict()
        for n_rpi in RPiSettings['use_RPi_nrs']:
            # Use SSH connection to send commands
            connection = ssh(RPiSettings['RPiIP'][n_rpi], RPiSettings['username'], RPiSettings['password'])
            # Run getImage.py on RPi to capture a frame
            com_str = 'cd ' + self.trackingFolder + ' && python getImage.py'
            connection.sendCommand(com_str)
            # Copy over output files to local TEMP folder
            callstr = 'scp ' + RPiSettings['username'] + '@' + RPiSettings['RPiIP'][n_rpi] + ':' + self.trackingFolder + '/frame.jpg ' + \
                 str(self.TEMPfolder) + '/frame' + str(n_rpi) + '.jpg'
            os.system(callstr)
        # Plot current frame for each RPi
        for n_rpi in RPiSettings['use_RPi_nrs']:
            image = Image.open(str(self.TEMPfolder) + '/frame' + str(n_rpi) + '.jpg')
            plotImage(self.im_views[n_rpi], image)

    def calibrate(self):
        self.update_camera_files(useCalibration = False)
        RPiSettings = self.get_camera_settings_dict()
        for n_rpi in RPiSettings['use_RPi_nrs']:
            print('Calibrating camera ' + str(n_rpi) + ' of ' + str(RPiSettings['use_RPi_nrs']))
            # Use SSH connection to send commands
            connection = ssh(RPiSettings['RPiIP'][n_rpi], RPiSettings['username'], RPiSettings['password'])
            # connection.sendCommand('cd ' + self.trackingFolder + ' && nohup python calibrate.py >/dev/null 2>&1 &')
            # Run calibrate.py on RPi
            com_str = 'cd ' + self.trackingFolder + ' && python calibrate.py'
            connection.sendCommand(com_str)
            # Copy over output files to local TEMP folder
            callstr = 'scp ' + RPiSettings['username'] + '@' + RPiSettings['RPiIP'][n_rpi] + ':' + self.trackingFolder + '/calibrationData.p ' + \
                 str(self.TEMPfolder) + '/calibrationData' + str(n_rpi) + '.p'
            os.system(callstr)
            callstr = 'scp ' + RPiSettings['username'] + '@' + RPiSettings['RPiIP'][n_rpi] + ':' + self.trackingFolder + '/calibrationTmatrix.p ' + \
                 self.TEMPfolder + '/calibrationTmatrix' + str(n_rpi) + '.p'
            os.system(callstr)
        # Save current RPiSettings also to TEMP folder with the calibration data
        RPiSettingsFile = self.TEMPfolder + '/RPiSettings.p'
        with open(RPiSettingsFile, 'wb') as file:
            pickle.dump(RPiSettings, file)
        # Show calibration data
        self.show_calibration()

    def save(self):
        self.load(loadFile=self.TEMPfolder + '/RPiSettings.p')
        RPiSettings = self.get_camera_settings_dict()
        # Get folder to which data will be saved
        saveFolder = self.openFolderDialog(caption='Select folder', directory=self.data_root)
        RPiSettingsFile = saveFolder + '/RPiSettings.p'
        with open(RPiSettingsFile, 'wb') as file:
            pickle.dump(RPiSettings, file)
        # Save calibration data for each RPi if present
        for n_rpi in RPiSettings['use_RPi_nrs']:
            calibrationFile = self.TEMPfolder + '/calibrationData' + str(n_rpi) + '.p'
            if os.path.isfile(calibrationFile):
                calibrationTmatrixFile = self.TEMPfolder + '/calibrationTmatrix' + str(n_rpi) + '.p'
                dst_calibrationFile = saveFolder + '/calibrationData' + str(n_rpi) + '.p'
                dst_calibrationTmatrixFile = saveFolder + '/calibrationTmatrix' + str(n_rpi) + '.p'
                copyfile(calibrationFile, dst_calibrationFile)
                copyfile(calibrationTmatrixFile, dst_calibrationTmatrixFile)
            else:
                show_message('Calibration file not found for RPi ' + str(n_rpi + 1) + '.')

    def load(self,loadFile=None):
        # Get RPiSettings file path
        if not loadFile:
            loadFile = self.openSingleFileDialog(caption='Select RPiSettings.p', directory=self.data_root)
        # Load RPiSettings
        with open(loadFile,'rb') as file:
            RPiSettings = pickle.load(file)
        # Put RPiSettings to GUI
        if RPiSettings['LEDmode'] == 'single':
            self.rb_led_single.setChecked(True)
        elif RPiSettings['LEDmode'] == 'double':
            self.rb_led_double.setChecked(True)
        if RPiSettings['save_frames']:
            self.rb_save_im_yes.setChecked(True)
        elif not RPiSettings['save_frames']:
            self.rb_save_im_no.setChecked(True)
        self.pt_arena_size_x.setPlainText(str(RPiSettings['arena_size'][0]))
        self.pt_arena_size_y.setPlainText(str(RPiSettings['arena_size'][1]))
        self.pt_ndots_x.setPlainText(str(RPiSettings['calibration_n_dots'][0]))
        self.pt_ndots_y.setPlainText(str(RPiSettings['calibration_n_dots'][1]))
        self.pt_offset_x.setPlainText(str(RPiSettings['corner_offset'][0]))
        self.pt_offset_y.setPlainText(str(RPiSettings['corner_offset'][1]))
        self.pt_calibration_spacing.setPlainText(str(RPiSettings['calibration_spacing']))
        self.pt_smooth_r.setPlainText(str(RPiSettings['smoothing_radius']))
        self.pt_LED_separation.setPlainText(str(RPiSettings['LED_separation']))
        self.pt_LED_angle.setPlainText(str(RPiSettings['LED_angle']))
        self.pt_camera_transfer_radius.setPlainText(str(RPiSettings['camera_transfer_radius']))
        self.pt_camera_iso.setPlainText(str(RPiSettings['camera_iso']))
        self.pt_shutter_speed.setPlainText(str(RPiSettings['shutter_speed']))
        self.lw_exposure_settings.setCurrentRow(RPiSettings['exposure_settings_selection'])
        CamRes = RPiSettings['resolution']
        CamResStr = str(CamRes[0]) + ', ' + str(CamRes[1])
        self.pt_resolution.setPlainText(CamResStr)
        self.pt_local_ip.setPlainText(RPiSettings['centralIP'])
        self.pt_rpi_password.setPlainText(RPiSettings['password'])
        self.pt_rpi_username.setPlainText(RPiSettings['username'])
        self.pt_posport.setPlainText(RPiSettings['pos_port'])
        self.pt_stopport.setPlainText(RPiSettings['stop_port'])
        for n_rpi in range(len(RPiSettings['RPiIP'])):
            self.pt_rpi_ips[n_rpi].setPlainText(RPiSettings['RPiIP'][n_rpi])
            self.cb_rpis[n_rpi].setChecked(RPiSettings['RPi_Usage'][n_rpi])
            self.pt_rpi_loc[n_rpi].setPlainText(RPiSettings['RPi_location'][n_rpi])
        self.trackingFolder = RPiSettings['tracking_folder']
        # Copy calibration data to local TEMP folder for use in GUI
        loadFolder = loadFile[:loadFile.rfind('/')]
        if loadFolder != self.TEMPfolder:
            for n_rpi in RPiSettings['use_RPi_nrs']:
                calibrationFile = loadFolder + '/calibrationData' + str(n_rpi) + '.p'
                calibrationTmatrixFile = loadFolder + '/calibrationTmatrix' + str(n_rpi) + '.p'
                dst_calibrationFile = self.TEMPfolder + '/calibrationData' + str(n_rpi) + '.p'
                dst_calibrationTmatrixFile = self.TEMPfolder + '/calibrationTmatrix' + str(n_rpi) + '.p'
                if os.path.isfile(calibrationFile):
                    copyfile(calibrationFile, dst_calibrationFile)
                    copyfile(calibrationTmatrixFile, dst_calibrationTmatrixFile)
        # Save current RPiSettings also to TEMP folder with the calibration data
        RPiSettingsFile = self.TEMPfolder + '/RPiSettings.p'
        with open(RPiSettingsFile, 'wb') as file:
            pickle.dump(RPiSettings, file)
        # Show calibration data
        self.show_calibration()

    def load_last(self):
        # Find the latest saved Camera Settings
        from RecordingManager import findLatestTimeFolder
        latest_folder = findLatestTimeFolder(self.RecGUI_dataFolder)
        # Compile full path to the RecGUI_Settings.p file in this folder and load the settings
        latest_CameraSettings_FullPath = self.RecGUI_dataFolder + '/' + latest_folder + '/RPiSettings.p'
        self.load(loadFile=latest_CameraSettings_FullPath)

    def show_calibration(self):
        # Loads current calibrationData and shows it in the plots
        RPiSettings = self.get_camera_settings_dict()
        for n_rpi in RPiSettings['use_RPi_nrs']:
            calibrationFile = self.TEMPfolder + '/calibrationData' + str(n_rpi) + '.p'
            if os.path.isfile(calibrationFile):
                with open(calibrationFile,'rb') as file:
                    calibrationData = pickle.load(file)
                    image = calibrationData['image']
                    plotImage(self.im_views[n_rpi], image)
            else:
                show_message('Calibration file not found for RPi ' + str(n_rpi + 1) + '.')

    def overlay(self):
        # Captures current image and overlays on it the currently active calibration chessboard corner pattern.
        self.update_camera_files()
        RPiSettings = self.get_camera_settings_dict()
        for n_rpi in RPiSettings['use_RPi_nrs']:
            # Use SSH connection to send commands
            connection = ssh(RPiSettings['RPiIP'][n_rpi], RPiSettings['username'], RPiSettings['password'])
            # Run calibrate.py on RPi with the overlay argument
            com_str = 'cd ' + self.trackingFolder + ' && python calibrate.py overlay'
            connection.sendCommand(com_str)
            # Copy over overlay.jpg image to local TEMP folder
            callstr = 'scp ' + RPiSettings['username'] + '@' + RPiSettings['RPiIP'][n_rpi] + ':' + self.trackingFolder + '/overlay.jpg ' + \
                      str(self.TEMPfolder) + '/overlay' + str(n_rpi) + '.jpg'
            os.system(callstr)
        # Plot current image with overlay for each RPi
        for n_rpi in RPiSettings['use_RPi_nrs']:
            image = Image.open(str(self.TEMPfolder) + '/overlay' + str(n_rpi) + '.jpg')
            plotImage(self.im_views[n_rpi], image)

    def apply(self):
        # Save current RPiSettings also to TEMP folder with the calibration data
        RPiSettings = self.get_camera_settings_dict()
        RPiSettingsFile = self.TEMPfolder + '/RPiSettings.p'
        with open(RPiSettingsFile, 'wb') as file:
            pickle.dump(RPiSettings, file)
        # Update camera files and close the window
        self.update_camera_files()
        self.close()

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
        RPiSettings = self.get_camera_settings_dict()
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
        for nRPi in range(len(RPiSettings['use_RPi_nrs'])):
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
        self.test_tracking_win.setGeometry(300, 200, 250, 20 * (len(RPiSettings['use_RPi_nrs']) + 2))
        # Start the RPis
        self.update_camera_files()
        trackingControl = rpiI.TrackingControl(RPiSettings)
        trackingControl.start()
        # Set up RPi latest position updater
        self.RPIpos = rpiI.onlineTrackingData(RPiSettings)
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