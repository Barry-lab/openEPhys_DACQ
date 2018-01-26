### This program has a GUI and handles the data flows synchrony of
### tracking with the electrophysiological recording of OpenEphysGUI.

### By Sander Tanni, May 2017, UCL

from PyQt4 import QtGui
import RecordingManagerDesign
import sys
from datetime import datetime
import os
import shutil
import RPiInterface as rpiI
import cPickle as pickle
import subprocess
from shutil import copyfile
import csv
import time
from CumulativePosPlot import PosPlot
from OpenEphysInterface import SendOpenEphysSingleMessage, SubscribeToOpenEphys
import HelperFunctions as hfunct
import threading

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
            item_time = datetime.strptime(dir_items[n_item], '%d-%m-%y')
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


class RecordingManager(QtGui.QMainWindow, RecordingManagerDesign.Ui_MainWindow):

    def __init__(self, parent=None):
        super(RecordingManager, self).__init__(parent=parent)
        self.setupUi(self)
        # Set GUI environment
        self.scripts_root = os.path.expanduser('~') + '/openEPhys_DACQ'
        self.pt_root_folder.setPlainText(os.path.expanduser('~') + '/RecordingData')
        self.file_server_path = '/media/QNAP/sanderT/room418'
        self.RecGUI_dataFolder = str(self.pt_root_folder.toPlainText()) + '/RecordingManagerData'
        # Set GUI interaction connections
        self.pb_load_last.clicked.connect(lambda:self.load_last_settings())
        self.pb_load.clicked.connect(lambda:self.load_settings())
        self.pb_root_folder.clicked.connect(lambda:self.root_folder_browse())
        self.pb_cam_set.clicked.connect(lambda:self.camera_settings())
        self.pb_start_rec.clicked.connect(lambda:self.start_rec())
        self.pb_stop_rec.clicked.connect(lambda:self.stop_rec())
        self.pb_process_data.clicked.connect(lambda:self.process_data())
        self.pb_sync_server.clicked.connect(lambda:self.sync_server())
        self.pb_open_rec_folder.clicked.connect(lambda:self.open_rec_folder())
        # Store original stylesheets for later use
        self.original_stylesheets = {}
        self.original_stylesheets['pb_start_rec'] = self.pb_start_rec.styleSheet()# Keep copy of default button color
        self.original_stylesheets['pb_stop_rec'] = self.pb_stop_rec.styleSheet() # Save Stop button default
        # Create TEMP folder. If exists, delete and re-create
        # TEMP folder is used sort of as a working memory by the Recording Manager
        self.TEMPfolder = self.RecGUI_dataFolder + '/TEMP'
        if os.path.isdir(self.TEMPfolder):
            shutil.rmtree(self.TEMPfolder)
        os.mkdir(self.TEMPfolder)
        self.pb_start_rec.setEnabled(True)

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

    def load_settings(self, folder_name=None):
        if not folder_name: # Get user to select settings to load if not given
            recording_folder_full_path = self.openFolderDialog(caption='Select Recording Folder')
            folder_name = os.path.basename(recording_folder_full_path)
        # Copy over settings to TEMP folder
        callstr = 'rsync -avzh ' + self.RecGUI_dataFolder + '/' + folder_name + '/ ' + self.TEMPfolder + '/'
        os.system(callstr)
        # Load RecGUI_Settings and update settings in GUI
        with open(self.TEMPfolder + '/RecGUI_Settings.p','rb') as file:
            RecGUI_Settings = pickle.load(file)
        self.pt_root_folder.setPlainText(RecGUI_Settings['root_folder'])
        self.pt_animal.setPlainText(RecGUI_Settings['animal'])
        self.pt_experiment_id.setPlainText(RecGUI_Settings['experiment_id'])
        self.pt_experimenter.setPlainText(RecGUI_Settings['experimenter'])
        self.pt_badChan.setPlainText(RecGUI_Settings['badChan'])
        self.pt_rec_folder.setPlainText(RecGUI_Settings['rec_folder'])
        self.rb_posPlot_yes.setChecked(RecGUI_Settings['PosPlot'])
        if len(RecGUI_Settings['channel_map']) > 0:
            self.pt_chan_map_1.setPlainText(RecGUI_Settings['channel_map'][0][0])
            self.pt_chan_map_1_chans.setPlainText(RecGUI_Settings['channel_map'][0][1])
            if len(RecGUI_Settings['channel_map']) > 1:
                self.pt_chan_map_2.setPlainText(RecGUI_Settings['channel_map'][1][0])
                self.pt_chan_map_2_chans.setPlainText(RecGUI_Settings['channel_map'][1][1])
            else:
                self.pt_chan_map_2.setPlainText('')
                self.pt_chan_map_2_chans.setPlainText('')
        else:
            self.pt_chan_map_1.setPlainText('')
            self.pt_chan_map_1_chans.setPlainText('')
            self.pt_chan_map_2.setPlainText('')
            self.pt_chan_map_2_chans.setPlainText('')
        # Update RPi Camera Settings
        with open(self.TEMPfolder + '/RPiSettings.p','rb') as file:
            RPiSettings = pickle.load(file)
        from CameraSettingsGUI import Camera_Files_Update_Function
        Camera_Files_Update_Function(SettingsFolder=self.TEMPfolder, RPiSettings=RPiSettings, useCalibration=True)

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
        date_folder_path = os.path.join(root_folder,animal,datetime.now().strftime('%d-%m-%y'))

        return date_folder_path

    def get_recording_folder_path(self, date_folder_path):
        if os.path.isdir(date_folder_path):
            # Get the name of the folder with latest date time as name
            latest_folder = findLatestTimeFolder(date_folder_path)
            foldertime = datetime.strptime(latest_folder, '%Y-%m-%d_%H-%M-%S')
            # If the folder time is less than 60 seconds behind rec_time_dt
            if (datetime.now() - foldertime).total_seconds() < 5:
                recording_folder = os.path.join(date_folder_path, latest_folder)
            else: # Display error in text box
                recording_folder = False
        else: # Display error in text box
            recording_folder = False

        return recording_folder

    def camera_settings(self):
        # Opens up a new window for Camera Settings
        from CameraSettingsGUI import CameraSettings
        self.CamSet = CameraSettings(self)
        self.CamSet.show()
        # Pass on paths to the Camera Settings GUI
        self.CamSet.TEMPfolder = self.TEMPfolder
        self.CamSet.RecGUI_dataFolder = self.RecGUI_dataFolder
        self.CamSet.pb_load_last.setEnabled(True) # Make this button available, as RecGUI_dataFolder info is passed on
        self.CamSet.scripts_root = self.scripts_root
        self.CamSet.data_root = str(self.pt_root_folder.toPlainText())
        # Check if Settings available in TEMP folder, if so Load them.
        RPiSettingsFile = self.TEMPfolder + '/RPiSettings.p'
        if os.path.isfile(RPiSettingsFile):
            self.CamSet.load(loadFile=self.TEMPfolder + '/RPiSettings.p')

    def get_RecordingGUI_settings(self):
        # Get channel mapping info
        channel_map = []
        if len(str(self.pt_chan_map_1.toPlainText())) > 0:
            channel_map_1 = [str(self.pt_chan_map_1.toPlainText())]
            chanString = str(self.pt_chan_map_1_chans.toPlainText())
            channel_map_1.append(chanString)
            chan_from = int(chanString[:chanString.find('-')])
            chan_to = int(chanString[chanString.find('-') + 1:])
            channel_map_1.append(range(chan_from - 1, chan_to))
            channel_map.append(channel_map_1)
            if len(str(self.pt_chan_map_2.toPlainText())) > 0:
                channel_map_2 = [str(self.pt_chan_map_2.toPlainText())]
                chanString = str(self.pt_chan_map_2_chans.toPlainText())
                channel_map_2.append(chanString)
                chan_from = int(chanString[:chanString.find('-')])
                chan_to = int(chanString[chanString.find('-') + 1:])
                channel_map_2.append(range(chan_from - 1, chan_to))
                channel_map.append(channel_map_2)

        # Grabs all options in the Recording Manager and puts them into a dictionary
        RecGUI_Settings = {'root_folder': str(self.pt_root_folder.toPlainText()), 
                           'animal': str(self.pt_animal.toPlainText()), 
                           'experiment_id': str(self.pt_experiment_id.toPlainText()), 
                           'experimenter': str(self.pt_experimenter.toPlainText()), 
                           'badChan': str(self.pt_badChan.toPlainText()), 
                           'rec_folder': str(self.pt_rec_folder.toPlainText()), 
                           'PosPlot': self.rb_posPlot_yes.isChecked(), 
                           'channel_map': channel_map}

        return RecGUI_Settings

    def start_task(self, TaskSettings, TaskIO):
        print('Starting Task')
        TaskModule = hfunct.import_subdirectory_module('Tasks', TaskSettings['name'])
        self.current_task = TaskModule.Core(TaskSettings, TaskIO)
        self.current_task.run()

    def start_rec(self):
        # Load tracking RPi Settings
        with open(self.TEMPfolder + '/RPi/RPiSettings.p','rb') as file:
            self.RPiSettings = pickle.load(file)
        # Connect to tracking RPis
        print('Connecting to tracking RPis...')
        self.trackingControl = rpiI.TrackingControl(self.RPiSettings)
        print('Connecting to tracking RPis Successful')
        # Initialize onlineTrackingData class
        print('Initializing Online Tracking Data...')
        histogramParameters = {'margins': 10, # histogram data margins in centimeters
                               'binSize': 2, # histogram binSize in centimeters
                               'speedLimit': 10}# centimeters of distance in last second to be included
        self.RPIpos = rpiI.onlineTrackingData(self.RPiSettings, HistogramParameters=histogramParameters, SynthData=False)
        print('Initializing Online Tracking Data Successful')
        # Initialize listening to Open Ephys GUI messages
        print('Connecting to Open Ephys GUI via ZMQ...')
        self.OEmessages = SubscribeToOpenEphys(verbose=False)
        self.OEmessages.connect()
        print('Connecting to Open Ephys GUI via ZMQ Successful')
        # Initialize Task
        if self.rb_task_yes.isChecked():
            print('Initializing Task...')
            TaskSettings = {'name': 'Foraging_Pellets'}
            # Put input streams in a default dictionary that can be used by task as needed
            TaskIO = {'RPIPos': self.RPIpos, 
                      'OEmessages': self.OEmessages, 
                      'MessageToOE': SendOpenEphysSingleMessage}
            TaskModule = hfunct.import_subdirectory_module('Tasks', TaskSettings['name'])
            self.current_task = TaskModule.Core(TaskSettings, TaskIO)
            print('Initializing Task Successful')
        # Start Open Ephys GUI recording
        print('Starting Open Ephys GUI Recording...')
        date_folder_path = self.get_date_folder_path()
        command = 'StartRecord RecDir=' + date_folder_path + ' CreateNewDir=1'
        SendOpenEphysSingleMessage(command)
        # Make sure OpenEphys is recording
        recording_folder = self.get_recording_folder_path(date_folder_path)
        while not recording_folder:
            time.sleep(0.1)
            recording_folder = self.get_recording_folder_path(date_folder_path)
        while not check_if_nwb_recording(recording_folder):
            time.sleep(0.1)
        self.pt_rec_folder.setPlainText(recording_folder)
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
        if self.rb_task_yes.isChecked():
            print('Starting Task...')
            self.current_task.run()
            print('Starting Task Successful')
        # Start cumulative plot
        if self.rb_posPlot_yes.isChecked():
            print('Starting Position Plot...')
            self.PosPlot = PosPlot(self.RPiSettings, self.RPIpos, histogramParameters)
            print('Starting Position Plot Successful')

    def stop_rec(self):# Stop Task
        if self.rb_task_yes.isChecked():
            print('Stopping Task...')
            self.current_task.stop()
            print('Stopping Task Successful')
        # Stop reading Open Ephys messages
        print('Closing Open Ephys GUI ZMQ connection...')
        self.OEmessages.disconnect()
        print('Closing Open Ephys GUI ZMQ connection Successful')
        # Stop cumulative plot
        if self.rb_posPlot_yes.isChecked() and hasattr(self, 'PosPlot'):
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
        while check_if_nwb_recording(str(self.pt_rec_folder.toPlainText())):
            print('Stopping Open Ephys GUI Recording...')
            SendOpenEphysSingleMessage('StopRecord')
            time.sleep(0.1)
        print('Stopping Open Ephys GUI Recording Successful')
        # Copy over RPi tracking folder to Recording Folder on PC using rsync
        print('Copying over tracking data to Recording PC...')
        user = self.RPiSettings['username']
        trackingFolder = self.RPiSettings['tracking_folder']
        for n_rpi in self.RPiSettings['use_RPi_nrs']:
            RPiIP = self.RPiSettings['RPiIP'][n_rpi]
            callstr = 'rsync -avzh ' + user + '@' + RPiIP + ':' + \
                      trackingFolder + '/ ' + str(self.pt_rec_folder.toPlainText()) + \
                      '/CameraData' + str(n_rpi) + '/'
            os.system(callstr)
            # Create a copy of log file as PosLog.csv file in recording folder
            src_logfile = str(self.pt_rec_folder.toPlainText()) + '/CameraData' + str(n_rpi) + '/logfile.csv'
            dst_logfile = str(self.pt_rec_folder.toPlainText()) + '/PosLog' + str(n_rpi) + '.csv'
            copyfile(src_logfile, dst_logfile)
        print('Copying over tracking data to Recording PC Successful')
        # Keep note of RPiSettings and Calibration data on PC for this GUI.
        # These are kept in the RecGUI_dataFolder as specified in the __init__ function.
        rec_folder_name = str(self.pt_rec_folder.toPlainText())
        rec_folder_name = rec_folder_name[rec_folder_name.rfind('/') + 1:]
        RecordingManagerSaveFolder = self.RecGUI_dataFolder + '/' + rec_folder_name
        callstr = 'rsync -avzh ' + self.TEMPfolder + '/ ' + RecordingManagerSaveFolder + '/'
        os.system(callstr)
        RecGUI_Settings = self.get_RecordingGUI_settings()
        with open(RecordingManagerSaveFolder + '/RecGUI_Settings.p', 'wb') as file:
            pickle.dump(RecGUI_Settings, file)
        with open(str(self.pt_rec_folder.toPlainText()) + '/RecGUI_Settings.p', 'wb') as file:
            pickle.dump(RecGUI_Settings, file)
        print('Saved Recording Manager Settings.')
        # Save badChan list from text box to a file in the recording folder
        if len(str(self.pt_badChan.toPlainText())) > 0:
            save_badChan_to_file(str(self.pt_badChan.toPlainText()), str(self.pt_rec_folder.toPlainText()))
            print('Saved the list of bad channels: ' + str(self.pt_badChan.toPlainText()))
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
        Processing_KlustaKwik.main(str(self.pt_rec_folder.toPlainText()))
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