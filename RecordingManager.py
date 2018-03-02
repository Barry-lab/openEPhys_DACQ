### This program has a GUI and handles the data flows synchrony of
### tracking with the electrophysiological recording of OpenEphysGUI.

### By Sander Tanni, May 2017, UCL

from PyQt4 import QtGui
import RecordingManagerDesign
import sys
from datetime import datetime
import os
import RPiInterface as rpiI
import cPickle as pickle
import subprocess
from shutil import copyfile, copytree, rmtree
import csv
import time
from CumulativePosPlot import PosPlot
from OpenEphysInterface import SendOpenEphysSingleMessage, SubscribeToOpenEphys
import HelperFunctions as hfunct
import threading
from copy import deepcopy
from scipy.io import savemat
from tempfile import mkdtemp
import numpy as np

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

def findLatestDateFolder(path):
    # Get subdirectory names and find folder with most recent date
    dir_items = os.listdir(path)
    dir_times = []
    item_nrs = []
    for n_item in range(len(dir_items)):
        try: # Use only items for which the name converts to correct date time format
            item_time = datetime.strptime(dir_items[n_item], '%y-%m-%d')
            item_nrs.append(n_item)
            dir_times.append(item_time)
        except:
            tmp = []
    latest_time = max(dir_times)
    latest_date_folder = dir_items[item_nrs[dir_times.index(latest_time)]]

    return latest_date_folder

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

def save_badChan_to_file(badChanString, targetFolder):
    if len(badChanString) > 0:
        # Separate input string into a list using ',' as deliminaters
        if badChanString.find(',') > -1: # If more than one channel specified
            # Find all values tetrode and channel values listed
            badChanStringList = badChanString.split(',')
        else:
            badChanStringList = [badChanString]
        # Identify any ranges specified with '-' and append these channels to the list
        for chanString in badChanStringList:
            if chanString.find('-') > -1:
                chan_from = chanString[:chanString.find('-')]
                chan_to = chanString[chanString.find('-') + 1:]
                for nchan in range(int(chan_to) - int(chan_from) + 1):
                    badChanStringList.append(str(nchan + int(chan_from)))
                badChanStringList.remove(chanString) # Remove the '-' containing list element
        # Reorder list of bad channels
        badChanStringList.sort(key=int)
        # Save as a text file to target location
        with open(targetFolder + '/BadChan', 'wb') as file:
            for chan_nr_str in badChanStringList:
                file.write(chan_nr_str + os.linesep)
    else:
        with open(targetFolder + '/BadChan', 'wb') as file:
            file.write('')

def get_recording_folder_path(recording_folder_root):
    if os.path.isdir(recording_folder_root):
        # Get the name of the folder with latest date time as name
        latest_folder = findLatestTimeFolder(recording_folder_root)
        foldertime = datetime.strptime(latest_folder, '%Y-%m-%d_%H-%M-%S')
        # If the folder time is less than 60 seconds behind rec_time_dt
        if (datetime.now() - foldertime).total_seconds() < 5:
            recording_folder = os.path.join(recording_folder_root, latest_folder)
        else: # Display error in text box
            recording_folder = False
    else: # Display error in text box
        recording_folder = False

    return recording_folder

def check_if_nwb_recording(path):
    # Find NWB file in the path
    dirfiles = os.listdir(path)
    dataFilePath = False
    for fname in dirfiles:
        if fname.endswith('.nwb'):
            dataFilePath = os.path.join(path, fname)
    if dataFilePath:
        # Check for file size at intervals to see if it is changing in size. Stop checking if constant.
        lastsize = os.stat(dataFilePath).st_size
        time.sleep(0.1)
        currsize = os.stat(dataFilePath).st_size
        file_recording = lastsize != currsize
    else:
        file_recording = False

    return file_recording

def update_specific_camera(RPiSettings, n_rpi, tmpFolder):
    # Use up to date scripts
    copytree('RaspberryPi', tmpFolder)
    # Store RPiSettings
    with open(os.path.join(tmpFolder, 'RPiSettings.p'),'wb') as file:
        pickle.dump(RPiSettings, file)
    # Specify RPi number
    with open(os.path.join(tmpFolder, 'RPiNumber'), 'wb') as file:
        file.write(str(n_rpi))
    # Store correct calibration data
    if str(n_rpi) in RPiSettings['calibrationData'].keys():
        calibrationFile = os.path.join(tmpFolder, 'calibrationData.p')
        with open(calibrationFile, 'wb') as file:
            pickle.dump(RPiSettings['calibrationData'][str(n_rpi)], file)
    # Recreate this folder on the tracking RPi
    tmp_path = os.path.join(tmpFolder, '')
    callstr = 'rsync -qavrP -e "ssh -l server" ' + tmp_path + ' ' + \
              RPiSettings['username'] + '@' + RPiSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
              RPiSettings['tracking_folder'] + ' --delete'
    _ = os.system(callstr)

def update_tracking_camera_files(RPiSettings):
    # Updates settings and scripts on all tracking cameras in use (based on settings)
    # Create temporary working directory
    RPiTempFolder = mkdtemp('RPiTempFolder')
    T_updateRPi = []
    for n_rpi in RPiSettings['use_RPi_nrs']:
        RPiTempSubFolder = os.path.join(RPiTempFolder, str(n_rpi))
        T = threading.Thread(target=update_specific_camera, args=[RPiSettings, n_rpi, RPiTempSubFolder])
        T.start()
        T_updateRPi.append(T)
    for T in T_updateRPi:
        T.join()
    rmtree(RPiTempFolder)

def retrieve_specific_camera_tracking_data(n_rpi, RPiSettings, folder_path):
    src_file = RPiSettings['username'] + '@' + RPiSettings['RPiInfo'][str(n_rpi)]['IP'] + ':' + \
               RPiSettings['tracking_folder'] + '/logfile.csv'
    dst_file = os.path.join(folder_path, 'PosLog' + str(n_rpi) + '.csv')
    callstr = 'scp -q ' + src_file + ' ' + dst_file
    _ = os.system(callstr)

def retrieve_tracking_data(RPiSettings, folder_path):
    # Copies all tracking data from active RPis to designated folder
    T_retrievePosLogsRPi = []
    for n_rpi in RPiSettings['use_RPi_nrs']:
        T = threading.Thread(target=retrieve_specific_camera_tracking_data, args=[n_rpi, RPiSettings, folder_path])
        T.start()
        T_retrievePosLogsRPi.append(T)
    for T in T_retrievePosLogsRPi:
        T.join()


class RecordingManager(QtGui.QMainWindow, RecordingManagerDesign.Ui_MainWindow):

    def __init__(self, parent=None):
        super(RecordingManager, self).__init__(parent=parent)
        self.setupUi(self)
        # Set GUI environment
        self.pt_root_folder.setPlainText(os.path.expanduser('~') + '/RecordingData')
        self.RecordingManagerSettingsFolder = 'RecordingManagerSettings'
        self.file_server_path = '/media/QNAP/sanderT/room418'
        # Create empty settings dictonary
        self.Settings = {}
        # Set GUI interaction connections
        self.pb_load_last.clicked.connect(lambda:self.load_last_settings())
        self.pb_load.clicked.connect(lambda:self.load_settings())
        self.pb_save.clicked.connect(lambda:self.save_settings())
        self.pb_root_folder.clicked.connect(lambda:self.root_folder_browse())
        self.pb_cam_set.clicked.connect(lambda:self.camera_settings())
        self.pb_task_set.clicked.connect(lambda:self.task_settings())
        self.pb_start_rec.clicked.connect(lambda:self.start_rec())
        self.pb_stop_rec.clicked.connect(lambda:self.stop_rec())
        self.pb_process_data.clicked.connect(lambda:self.process_data())
        self.pb_sync_server.clicked.connect(lambda:self.sync_server())
        self.pb_open_rec_folder.clicked.connect(lambda:self.open_rec_folder())
        # Store original stylesheets for later use
        self.original_stylesheets = {}
        self.original_stylesheets['pb_start_rec'] = self.pb_start_rec.styleSheet()# Keep copy of default button color
        self.original_stylesheets['pb_stop_rec'] = self.pb_stop_rec.styleSheet() # Save Stop button default

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
        RecGUI_Settings = {'root_folder': str(self.pt_root_folder.toPlainText()), 
                           'animal': str(self.pt_animal.toPlainText()), 
                           'experiment_id': str(self.pt_experiment_id.toPlainText()), 
                           'experimenter': str(self.pt_experimenter.toPlainText()), 
                           'badChan': str(self.pt_badChan.toPlainText()), 
                           'rec_folder': str(self.pt_rec_folder.toPlainText()), 
                           'PosPlot': np.array(self.rb_posPlot_yes.isChecked()), 
                           'channel_map': channel_map, 
                           'TaskActive': np.array(self.rb_task_yes.isChecked())}

        return RecGUI_Settings

    def load_settings(self, path=None):
        if path is None: # Get user to select settings to load if not given
            path = hfunct.openSingleFileDialog('load', suffix='p', caption='Select file to load')
        # Load Settings file
        with open(path,'rb') as file:
            Settings = pickle.load(file)
        # # Load calibration images if available
        # calibrationImagesFile = path[:-path[::-1].index('.')] + 'calibrationImages.npy'
        # if 'calibrationData' in Settings['RPiSettings'].keys() and os.path.isfile(calibrationImagesFile):
        #     calibrationImages = np.load(calibrationImagesFile)
        #     for n_rpi in range(len(Settings['RPiSettings']['calibrationData'])):
        #         image_pos = 0
        #         if n_rpi in Settings['RPiSettings']['use_RPi_nrs']:
        #             Settings['RPiSettings']['calibrationData'][str(n_rpi)]['image'] = calibrationImages[image_pos, :, :, :]
        #             image_pos += 1
        self.Settings = Settings
        # Put Recording Manager General settings into GUI
        RecGUI_Settings = self.Settings['General']
        self.pt_root_folder.setPlainText(RecGUI_Settings['root_folder'])
        self.pt_animal.setPlainText(RecGUI_Settings['animal'])
        self.pt_experiment_id.setPlainText(RecGUI_Settings['experiment_id'])
        self.pt_experimenter.setPlainText(RecGUI_Settings['experimenter'])
        self.pt_badChan.setPlainText(RecGUI_Settings['badChan'])
        self.pt_rec_folder.setPlainText(RecGUI_Settings['rec_folder'])
        if RecGUI_Settings['PosPlot']:
            self.rb_posPlot_yes.setChecked(True)
        else:
            self.rb_posPlot_no.setChecked(True)
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

    def save_settings(self, path=None, matlab=False):
        # Saves current settings into a file
        if path is None:
            path = hfunct.openSingleFileDialog('save', suffix='p', caption='Save file name and location')
        self.Settings['General'] = self.get_RecordingGUI_settings()
        self.Settings['Time'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        Settings = deepcopy(self.Settings)
        # if 'calibrationData' in Settings['RPiSettings'].keys():
        #     # Separate calibration images and save separately
        #     calibrationImages = []
        #     for n_rpi in range(len(Settings['RPiSettings']['calibrationData'])):
        #         if n_rpi in Settings['RPiSettings']['use_RPi_nrs']:
        #             calibrationImages.append(Settings['RPiSettings']['calibrationData'][n_rpi].pop('image'))
        #     calibrationImages = np.array(calibrationImages)
        #     calibrationImagesFile = path[:-path[::-1].index('.')] + 'calibrationImages.npy'
        #     np.save(calibrationImagesFile, calibrationImages, allow_pickle=False)
        # Save settings into a pickle file
        with open(path, 'wb') as file:
            pickle.dump(Settings, file)
        # if matlab:
        #     # Save MATLAB compatible file
        #     matFilePath = path[:-path[::-1].index('.')] + 'mat'
        #     settings = deepcopy(self.Settings)
        #     # Replace None with 0 in calibrationData so it could be saved as MATLAB file
        #     settings['RPiSettings']['calibrationData'] = map(lambda x: 0 if x is None else x, settings['RPiSettings']['calibrationData'])
        #     savemat(matFilePath, settings, long_field_names=True)
        print('Settings saved.')

    def load_last_settings(self):
        # Check if specific Animal ID has been entered
        animal_id = str(self.pt_animal.toPlainText())
        if len(animal_id) > 0:
            # Find the latest saved Recording Manager Settings
            dir_animal = str(self.pt_root_folder.toPlainText()) + '/' + animal_id
            latest_date_folder = findLatestDateFolder(dir_animal)
            latest_folder = findLatestTimeFolder(dir_animal + '/' + latest_date_folder)
        else:
            # Find the latest saved Recording Manager Settings
            latest_folder = findLatestTimeFolder(self.RecGUI_dataFolder)
        # Load the settings in the latest_folder
        self.load_settings(folder_name=latest_folder)

    def root_folder_browse(self):
        # Pops up a new window to select a folder and then inserts the path to the text box
        self.pt_root_folder.setPlainText(self.openFolderDialog())

    def get_date_folder_path(self):
        # Get directory names
        root_folder = str(self.pt_root_folder.toPlainText())
        animal = str(self.pt_animal.toPlainText())
        date_folder_path = os.path.join(root_folder,animal,datetime.now().strftime('%y-%m-%d'))

        return date_folder_path

    def camera_settings(self):
        # Opens up a new window for Camera Settings
        from CameraSettingsGUI import CameraSettings
        self.CamSet = CameraSettings(parent=self)
        self.CamSet.show()
        # Check if RPiSettings available, if so Load them.
        if 'RPiSettings' in self.Settings.keys():
            self.CamSet.load(deepcopy(self.Settings['RPiSettings']))

    def task_settings(self):
        from TaskSettingsGUI import TaskSettingsGUI
        self.TaskSet = TaskSettingsGUI(parent=self)

    def start_rec(self):
        self.Settings['General'] = self.get_RecordingGUI_settings()
        # Connect to tracking RPis
        print('Connecting to tracking RPis...')
        update_tracking_camera_files(self.Settings['RPiSettings'])
        self.trackingControl = rpiI.TrackingControl(self.Settings['RPiSettings'])
        print('Connecting to tracking RPis Successful')
        # Initialize onlineTrackingData class
        print('Initializing Online Tracking Data...')
        histogramParameters = {'margins': 10, # histogram data margins in centimeters
                               'binSize': 2, # histogram binSize in centimeters
                               'speedLimit': 10}# centimeters of distance in last second to be included
        self.RPIpos = rpiI.onlineTrackingData(self.Settings['RPiSettings'], HistogramParameters=histogramParameters, SynthData=False)
        print('Initializing Online Tracking Data Successful')
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
        # Start Open Ephys GUI recording
        print('Starting Open Ephys GUI Recording...')
        recording_folder_root = os.path.join(self.Settings['General']['root_folder'], str(self.pt_animal.toPlainText()))
        command = 'StartRecord RecDir=' + recording_folder_root + ' CreateNewDir=1'
        SendOpenEphysSingleMessage(command)
        # Make sure OpenEphys is recording
        recording_folder = get_recording_folder_path(recording_folder_root)
        while not recording_folder:
            time.sleep(0.1)
            recording_folder = get_recording_folder_path(recording_folder_root)
        while not check_if_nwb_recording(recording_folder):
            time.sleep(0.1)
        self.pt_rec_folder.setPlainText(recording_folder)
        self.Settings['General']['rec_folder'] = recording_folder
        print('Starting Open Ephys GUI Recording Successful')
        # Start the tracking scripts on all RPis
        print('Starting tracking RPis...')
        self.trackingControl.start()
        print('Starting tracking RPis Successful')
        # Change Start button style
        self.pb_start_rec.setStyleSheet('background-color: red') # Change button to red
        # Disable and Enable Start and Stop buttons, respectively
        self.pb_start_rec.setEnabled(False)
        self.pb_stop_rec.setEnabled(True)
        self.pb_process_data.setEnabled(False)
        self.pb_sync_server.setEnabled(False)
        self.pb_open_rec_folder.setEnabled(False)
        # Start task
        if self.Settings['General']['TaskActive']:
            print('Starting Task...')
            self.current_task.run()
            print('Starting Task Successful')
        # Start cumulative plot
        if self.Settings['General']['PosPlot']:
            print('Starting Position Plot...')
            self.PosPlot = PosPlot(self.Settings['RPiSettings'], self.RPIpos, histogramParameters)
            print('Starting Position Plot Successful')
        # Store recording settings to recording folder (is overwritten at stop_rec)
        self.save_settings(path=os.path.join(self.Settings['General']['rec_folder'], 'RecordingSettings.p'))

    def stop_rec(self):# Stop Task
        if self.Settings['General']['TaskActive']:
            print('Stopping Task...')
            self.current_task.stop()
            print('Stopping Task Successful')
        # Stop reading Open Ephys messages
        print('Closing Open Ephys GUI ZMQ connection...')
        self.OEmessages.disconnect()
        print('Closing Open Ephys GUI ZMQ connection Successful')
        # Stop cumulative plot
        if self.Settings['General']['PosPlot'] and hasattr(self, 'PosPlot'):
            print('Stopping Position Plot...')
            self.PosPlot.close()
            print('Stopping Position Plot Successful')
        # Stop updating online tracking data
        print('Closing Online Tracking Data...')
        self.RPIpos.close()
        print('Closing Online Tracking Data Successful')
        # Stop RPis
        print('Stopping tracking RPis...')
        self.trackingControl.stop()
        print('Stopping tracking RPis Successful')
        # Stop Open Ephys Recording
        while check_if_nwb_recording(self.Settings['General']['rec_folder']):
            print('Stopping Open Ephys GUI Recording...')
            SendOpenEphysSingleMessage('StopRecord')
            time.sleep(0.1)
        print('Stopping Open Ephys GUI Recording Successful')
        # Copy over tracking data
        print('Copying over tracking data to Recording PC...')
        retrieve_tracking_data(self.Settings['RPiSettings'], self.Settings['General']['rec_folder'])
        print('Copying over tracking data to Recording PC Successful')
        # Save badChan list to a file in the recording folder
        save_badChan_to_file(str(self.pt_badChan.toPlainText()), self.Settings['General']['rec_folder'])
        print('Saved the list of bad channels: ' + str(self.pt_badChan.toPlainText()))
        # Store recording settings to recording folder (overwrite in case changes made during recording)
        self.save_settings(path=os.path.join(self.Settings['General']['rec_folder'], 'RecordingSettings.p'), matlab=True)
        # Store settings for RecordingManager history reference
        RecordingManagerSettingsPath = os.path.join(self.Settings['General']['root_folder'], 
                                                    self.RecordingManagerSettingsFolder)
        if not os.path.isdir(RecordingManagerSettingsPath):
            os.mkdir(RecordingManagerSettingsPath)
        RecordingManagerSettingsFilePath = os.path.join(RecordingManagerSettingsPath, 
                                                        os.path.basename(self.Settings['General']['rec_folder']) + '.p')
        self.save_settings(path=RecordingManagerSettingsFilePath)
        # Change start button colors
        self.pb_start_rec.setStyleSheet(self.original_stylesheets['pb_start_rec']) # Start button to default
        # Disable Stop button and Enable other buttons
        self.pb_start_rec.setEnabled(True)
        self.pb_stop_rec.setEnabled(False)
        self.pb_process_data.setEnabled(True)
        self.pb_sync_server.setEnabled(True)
        self.pb_open_rec_folder.setEnabled(True)

    def process_data(self):
        # Applies KlustaKwik tetrode-wise to recorded data
        self.pb_start_rec.setEnabled(False)
        self.pb_process_data.setEnabled(False)
        self.pb_sync_server.setEnabled(False)
        self.pb_open_rec_folder.setEnabled(False)
        time.sleep(0.1)
        import Processing_KlustaKwik
        Processing_KlustaKwik.main(self.Settings['General']['rec_folder'])
        self.pb_start_rec.setEnabled(True)
        self.pb_process_data.setEnabled(True)
        self.pb_sync_server.setEnabled(True)
        self.pb_open_rec_folder.setEnabled(True)

    def sync_server(self):
        # Change button colors
        self.original_stylesheets['pb_sync_server'] = self.pb_sync_server.styleSheet() # Save button color
        self.pb_sync_server.setStyleSheet('background-color: red') # Change button to red
        # Extract directory tree from root folder to data to append to server folder
        rec_folder = str(self.pt_rec_folder.toPlainText())
        root_folder = str(self.pt_root_folder.toPlainText())
        directory_tree = rec_folder[len(rec_folder)-rec_folder[::-1].rfind(root_folder[::-1]):]
        server_folder = self.file_server_path + directory_tree
        # Create target folder tree on server
        os.system('mkdir -p ' + server_folder)
        # Copy files over to server
        print('Copying files to server ...')
        callstr = 'rsync -avzh ' + rec_folder + '/ ' + server_folder + '/'
        os.system(callstr)
        # Change button color back to default
        self.pb_sync_server.setStyleSheet(self.original_stylesheets['pb_sync_server'])

    def open_rec_folder(self):
        # Opens recording folder with Ubuntu file browser
        subprocess.Popen(['xdg-open', str(self.pt_rec_folder.toPlainText())])


# The following is the default ending for a QtGui application script
def main():
    app = QtGui.QApplication(sys.argv)
    form = RecordingManager()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main()
