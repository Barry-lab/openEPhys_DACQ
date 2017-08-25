### This program allows loading detecting threshold crossing on tetrode recordings.
### The GUI provides convenient method for setting different thresholds for all channels.

### Note, the script has high demand for working memory, as all data is loaded into memory.
### To process a recording of 20 min, at least 16 GB of RAM is required to be available.

### In the output the spiketimes are aligned to the beginning of the LFP signal.

### Note, the script is poorly written when it comes to how LFPs from different tetrodes
### and channels is structured. This is because initially an attempt was made to be able to
### work with recording that have missing channels on tetrodes. This functionality is not
### reliable. Script is only fully reliable as long as channel count starts from 1.

### By Sander Tanni, May 2017, UCL

from PyQt4 import QtGui
import sys
import DetectWaveformsDesign
import ntpath
import OpenEphys
import numpy as np
import pyqtgraph as pg
import os
import cPickle as pickle
from scipy import signal
from multiprocessing import Process


def Filter(signal_in, sampling_rate=30000, highpass_frequency=600, filt_order=2):
    # Applies a high pass filter on the signal
    Wn = float(highpass_frequency) / float(sampling_rate)
    b, a = signal.butter(filt_order, Wn, 'highpass')
    signal_out = signal.filtfilt(b, a, signal_in, padlen=0)

    return signal_out


def show_message(message, message_more=None):
    # Opens up a dialog box and presents a message
    msg = QtGui.QMessageBox()
    msg.setIcon(QtGui.QMessageBox.Information)
    msg.setText(message)
    if message_more:
        msg.setInformativeText(message_more)
    msg.setWindowTitle('Message')
    msg.setStandardButtons(QtGui.QMessageBox.Ok)
    msg.exec_()


def chunks(input_list, chunk_size):
    for nchunk in range(0, len(input_list), chunk_size):
        yield input_list[nchunk:nchunk + chunk_size]


class DetectWaveforms(QtGui.QMainWindow, DetectWaveformsDesign.Ui_MainWindow):
    def __init__(self, parent=None):
        super(DetectWaveforms, self).__init__(parent=parent)
        self.setupUi(self)
        # Set necessary variables
        self.highpass_frequency = 600
        self.waveform_width = [0.0002, 0.0008] # [before, after] waveform width in seconds
        self.waveform_exemption = 0.001 # duration of exemption of other threshold crossings on tetrode in seconds
        self.pt_default_threshold.setPlainText(str(-75))
        self.default_waveform_range = 300 # Set default waveform range in microvolts
        self.tetrode_groups = [np.arange(16), np.arange(16) + 16]
        self.channel_groups = [np.arange(64), np.arange(64) + 64]
        # Set links to GUI objects
        self.axesIDs = [[self.ax_Ch1_1,self.ax_Ch1_2,self.ax_Ch1_3,self.ax_Ch1_4],\
                       [self.ax_Ch2_1,self.ax_Ch2_2,self.ax_Ch2_3,self.ax_Ch2_4],\
                       [self.ax_Ch3_1,self.ax_Ch3_2,self.ax_Ch3_3,self.ax_Ch3_4],\
                       [self.ax_Ch4_1,self.ax_Ch4_2,self.ax_Ch4_3,self.ax_Ch4_4]]
        self.thresholdBoxes = [self.sb_Ch1_threshold,self.sb_Ch2_threshold,self.sb_Ch3_threshold,self.sb_Ch4_threshold]
        self.ymin_IDs = [self.sb_min_Ch1,self.sb_min_Ch2,self.sb_min_Ch3,self.sb_min_Ch4]
        self.ymax_IDs = [self.sb_max_Ch1,self.sb_max_Ch2,self.sb_max_Ch3,self.sb_max_Ch4]
        self.gbChIDs = [self.gb_Ch1,self.gb_Ch2,self.gb_Ch3,self.gb_Ch4]
        self.gb_IDs = [self.gb_Ch1,self.gb_Ch2,self.gb_Ch3,self.gb_Ch4]
        # Initialize plot axes ranges
        for ntchan in range(4):
            self.ymax_IDs[ntchan].setValue(np.int16(self.default_waveform_range))
            self.ymin_IDs[ntchan].setValue(np.int16(-self.default_waveform_range))
        # Set UI control connections
        self.pb_load_individual_channels.clicked.connect(lambda:self.openFilesDialog())
        self.pb_load_all_channels.clicked.connect(lambda:self.openSingleFileDialog())
        self.pb_plot_trace.clicked.connect(lambda:self.plotTrace())
        self.pb_save_waveforms.clicked.connect(lambda:self.save_waveforms())
        self.pb_next.clicked.connect(lambda:self.next_channel())
        self.pb_prev.clicked.connect(lambda:self.previous_channel())
        self.pb_klustakwik.clicked.connect(lambda:self.start_klusta())
        self.cb_toggle_axes.stateChanged.connect(lambda:self.toggle_axes())
        self.pb_update_ch1.clicked.connect(lambda:self.update_plots([0]))
        self.pb_update_ch2.clicked.connect(lambda:self.update_plots([1]))
        self.pb_update_ch3.clicked.connect(lambda:self.update_plots([2]))
        self.pb_update_ch4.clicked.connect(lambda:self.update_plots([3]))
        # If below lines give errors, it is likely that necessary lines have not been added to
        # DetectWaveformsDesign.py. Search for file 'DetectWaveformsDesignAddOn.py'
        self.ax_Ch1_1.mouseClickObject.clicked.connect(lambda:self.mouseclick_on_plot(0))
        self.ax_Ch2_2.mouseClickObject.clicked.connect(lambda:self.mouseclick_on_plot(1))
        self.ax_Ch3_3.mouseClickObject.clicked.connect(lambda:self.mouseclick_on_plot(2))
        self.ax_Ch4_4.mouseClickObject.clicked.connect(lambda:self.mouseclick_on_plot(3))        
        
        
    def openFilesDialog(self):
        # Pops up a GUI to select files independently
        dialog = QtGui.QFileDialog(self, 'Select channels from a single tetrode')
        dialog.setFileMode(QtGui.QFileDialog.ExistingFiles)
        dialog.setNameFilters(['(*.continuous)'])
        dialog.setViewMode(QtGui.QFileDialog.List) # or Detail
        if dialog.exec_():
            # Get paths and file names of selection
            tmp = dialog.selectedFiles()
                    
                    
    def openSingleFileDialog(self):
        # Pops up a GUI to select a single file.
        dialog = QtGui.QFileDialog(self, caption='Select one from the set of LFP files')
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        dialog.setNameFilters(['(*.continuous)'])
        dialog.setViewMode(QtGui.QFileDialog.List) # or Detail
        if dialog.exec_():
            # Get path and file name of selection
            tmp = dialog.selectedFiles()
            selected_file = str(tmp[0])
            self.findFilesAndCreateStructure(selected_file)


    def findFilesAndCreateStructure(self, selected_file):
        # Finds all files with same prefix, prepare data structure and start loading files
        self.fpath = ntpath.dirname(selected_file)
        f_s_name = str(ntpath.basename(selected_file))
        # Check if file name is correct format
        numstart = f_s_name.find('_CH') + 3
        numend = f_s_name.find('.continuous')
        if numstart == -1 or numend == -1:
            print('Error: Unexpected file name')
        else:
            # Get prefix for this set of LFPs
            fprefix = f_s_name[:f_s_name.index('_')]
            # Get list of all other files and channel numbers from this recording
            self.fileNames = []
            channel_nrs = []
            for fname in os.listdir(self.fpath):
                if fname.startswith(fprefix) and fname.endswith('.continuous') and fname.find('_CH') > -1:
                    # Get the channel number
                    numstart = fname.find('_CH') + 3
                    numend = fname.find('.continuous')
                    if numstart == -1 or numend == -1:
                        print('Error: Unexpected file name')
                    else:
                        self.fileNames.append(fname)
                        channel_nrs.append(int(fname[numstart:numend]) - 1)
            # Order the files by channel number
            file_order = np.argsort(np.array(channel_nrs))
            channel_nrs = [channel_nrs[x] for x in file_order]
            channel_nrs = np.array(channel_nrs)
            self.fileNames = [self.fileNames[x] for x in file_order]
            # Get list of bad channels
            self.listBadChannels()
            # Automatically assign channels to tetrodes
            max_tetrode_ch_nr = int(np.ceil(float(channel_nrs.max() / float(4)))) * 4
            num_tetrodes = max_tetrode_ch_nr / 4
            tetrode_channels = np.reshape(np.arange(max_tetrode_ch_nr), (num_tetrodes, 4))
            tetrode_nrs = np.arange(num_tetrodes)
            # Remove tetrodes that don't have any channels represented
            chan_exists = np.reshape(np.in1d(tetrode_channels,channel_nrs),(num_tetrodes,4))
            del_tetrode = np.where(np.invert(np.any(chan_exists, axis=1)))[0]
            chan_exists = np.delete(chan_exists, del_tetrode, axis=0)
            tetrode_channels = np.delete(tetrode_channels, del_tetrode, axis=0)
            tetrode_nrs = np.delete(tetrode_nrs, del_tetrode)
            # Create list where elements are lists of channel numbers for that tetrode
            tetrode_channels_int = [[]] * len(tetrode_nrs)
            tetrode_channels_str = [[]] * len(tetrode_nrs)
            for ntet in range(len(tetrode_channels_int)):
                tetrode_channels_int[ntet] = tetrode_channels[ntet,np.where(chan_exists[ntet,:])[0]]
                tetrode_channels_str[ntet] = map(str, list(tetrode_channels_int[ntet] + 1))
                tetrode_channels_int[ntet] = list(tetrode_channels_int[ntet])
            # Assign completed channel numbering variables to self
            self.tetrode_numbers_int = list(tetrode_nrs)
            self.tetrode_numbers_str = map(str, list(np.array(self.tetrode_numbers_int) + 1))
            self.tetrode_channels_int = tetrode_channels_int
            self.tetrode_channels_str = tetrode_channels_str
            self.channel_numbers_int = list(channel_nrs)
            self.channel_numbers_str = map(str, channel_nrs + 1)
            # Populate Tetrode List Box with options
            self.initialize_lw_tetrodes()
            # Create empty list structure for data
            self.LFPs = []
            for channels in self.tetrode_channels_int:
                self.LFPs.append([[]] * len(channels))
            # Create empty list structure for waveform windows
            self.waveform_windows = []
            for channels in self.tetrode_channels_int:
                self.waveform_windows.append([[]] * len(channels))
            # Create empty list structure for std values
            self.stdevs = []
            for channels in self.tetrode_channels_int:
                self.stdevs.append([[]] * len(channels))
            # Create empty list structure for threshold values
            self.thresholds = []
            for channels in self.tetrode_channels_int:
                self.thresholds.append([[]] * len(channels))
            # Create empty list structure for plot minval values
            self.minvals = []
            for channels in self.tetrode_channels_int:
                self.minvals.append([[]] * len(channels))
            # Create empty list structure for plot maxval values
            self.maxvals = []
            for channels in self.tetrode_channels_int:
                self.maxvals.append([[]] * len(channels))
            # Create empty list structure for spiketime values
            self.spiketimes = []
            for channels in self.tetrode_channels_int:
                self.spiketimes.append([[]] * len(channels))
            self.load_data()
            # Spikes detected counter
            self.spikes_detected = []
            for channels in self.tetrode_channels_int:
                self.spikes_detected.append([False] * len(channels))


    def Referencing(self):
        # Takes in a list of tetrodes, where each element is a list of channel LFP datas for that tetrode.
        # References each channel to all others, excluding the ones on same tetrode and bad channels.
        # Assumes the following channel structure 0-3 (first tetrode), 4-7 (second tetrode), etc.

        # Arrange channel numbers as they appear on tetrodes
        tet_chans = np.arange(len(self.LFPs) * 4, dtype=np.int16).reshape(len(self.LFPs), 4)
        other_mean_lfp = [None] * len(self.LFPs)
        other_median_lfp = [None] * len(self.LFPs)
        for ntet in range(len(self.LFPs)): # Find mean of other channels separately for each tetrode
            print('Referencing tetrode ' + str(ntet + 1) + ' out of ' + str(len(self.LFPs)))
            # Find out which group this tetrode belongs to
            for ngroup in range(len(self.tetrode_groups)):
                if sum(self.tetrode_groups[ngroup] == ntet) > 0:
                    tet_group = ngroup
            # List channels on other groups
            other_group_chans = []
            for ngroup in range(len(self.tetrode_groups)):
                if ngroup != tet_group:
                    other_group_chans.append(self.channel_groups[ngroup])
            other_group_chans = np.concatenate(other_group_chans)
            # Find indices of channels that are on the same tetrode or on the bad channel list
            # or channels that belong to another group that this tetrode
            dont_use_chans = tet_chans[ntet,:]
            if len(self.tetrode_groups) > 1:
                dont_use_chans = np.concatenate((dont_use_chans, other_group_chans))
            if len(self.badChan) > 0:
                dont_use_chans = np.concatenate((dont_use_chans, np.array(self.badChan)))
            dont_use_chans = np.in1d(tet_chans, dont_use_chans)
            otherchans = np.invert(dont_use_chans.reshape(tet_chans.shape))
            otherchan_idx = np.where(otherchans) # Get the indices of channels to use

            # Compute the mean of the other channels, using their indices to find them in the LFPs lists
            other_mean_lfp[ntet] = (np.zeros(self.n_samples, dtype=np.float32))
            otherchan_total = np.sum(otherchans)
            for nchan in range(otherchan_total):
                sub_ntet = otherchan_idx[0][nchan]
                sub_nchan = otherchan_idx[1][nchan]
                other_mean_lfp[ntet] = other_mean_lfp[ntet] + np.float32(self.LFPs[sub_ntet][sub_nchan])
            other_mean_lfp[ntet] = np.float32(other_mean_lfp[ntet]) / otherchan_total            
        # Substract the mean of other channels on other tetrodes from each channel on each tetrode
        for ntet in range(len(self.LFPs)):
            for nchan in range(4):
                if np.sum(np.array(self.badChan) == tet_chans[ntet,nchan]) == 0:
                    self.LFPs[ntet][nchan] = np.int16(np.float32(self.LFPs[ntet][nchan]) - other_mean_lfp[ntet])

        #     # Compute the median of other channels
        #     other_median_lfp[ntet] = np.zeros(self.n_samples, dtype=np.int16)
        #     otherchan_total = np.sum(otherchans)
        #     for chunk in chunks(range(self.n_samples), 10 ** 6):
        #         chunk_LFPs = np.zeros((otherchan_total, len(chunk)), dtype=np.int16)
        #         for nchan in range(otherchan_total):
        #             sub_ntet = otherchan_idx[0][nchan]
        #             sub_nchan = otherchan_idx[1][nchan]
        #             chunk_LFPs[nchan,:] = self.LFPs[sub_ntet][sub_nchan][chunk]
        #         other_median_lfp[ntet][chunk] = np.median(chunk_LFPs, axis=0)
        # # Substract the mean of other channels on other tetrodes from each channel on each tetrode
        # for ntet in range(len(self.LFPs)):
        #     for nchan in range(4):
        #         if np.sum(np.array(self.badChan) == tet_chans[ntet,nchan]) == 0:
        #             self.LFPs[ntet][nchan] = self.LFPs[ntet][nchan] - other_median_lfp[ntet]


    def filterThisChan(self, ntet, ntchan):
        # This function is called in multiple threads to speed processing
        self.LFPs[ntet][ntchan] = np.int16(Filter(np.float64(self.LFPs[ntet][ntchan]), 
                                           highpass_frequency=self.highpass_frequency))


    def load_data(self):
        # Loads all files and performes necessary preprocessing
        # Load data for all channels into memory
        for ntet in range(len(self.LFPs)):
            print('Loading tetrode ' + str(ntet + 1) + ' of ' + str(len(self.LFPs)))
            for ntchan in range(len(self.LFPs[ntet])):
                chan_nr = self.tetrode_channels_int[ntet][ntchan]
                fullfilepath = self.fpath + '/' + self.fileNames[self.channel_numbers_int.index(chan_nr)]
                OEdict = OpenEphys.loadContinuous(fullfilepath, dtype=np.int16, verbose=False)
                self.LFPs[ntet][ntchan] = OEdict['data']
                self.bitVolts = OEdict['header']['bitVolts']
                self.samplingRate = OEdict['header']['sampleRate']
                self.n_samples = self.LFPs[ntet][ntchan].size
                # Set data to zero if BadChan
                if np.sum(np.array(self.badChan) == chan_nr) > 0:
                    print(str(chan_nr) + ' set to zero')
                    self.LFPs[ntet][ntchan] = np.zeros(self.LFPs[ntet][ntchan].shape, dtype=np.int16)
        # Perform referencing (common average) if box checked
        if self.cb_reference.isChecked() == True:
            self.Referencing()
        # Perform filtering (band pass) if box checked
        if self.cb_filter.isChecked() == True:
            for ntet in range(len(self.LFPs)):
                print('Filtering channels on tetrode ' + str(ntet + 1) + ' of ' + str(len(self.LFPs)))
                FilteringThreads = []
                for ntchan in range(len(self.LFPs[ntet])):
                    FilteringThreads.append(Process(target=self.filterThisChan, args=(ntet, ntchan)))
                    FilteringThreads[ntchan].start()
                for fThread in FilteringThreads:
                    fThread.join()
        # Find position data edges, so spikes without position data could be ignored
        self.get_position_data_edges()
        print('Files loaded.')


    def listBadChannels(self):
        # Find file BadChan in the directory and extract numbers from each row
        badChanFile = self.fpath + '/BadChan'
        if os.path.exists(badChanFile):
            with open(badChanFile) as file:
                content = file.readlines()
            content = [x.strip() for x in content]
            self.badChan = list(np.array(map(int, content)) - 1)
        else:
            self.badChan = []


    def initialize_lw_tetrodes(self):
        # Populates the listbox with tetrode numbres and corresponding channel numbers
        self.lw_tetrodes.clear()
        itemstrings = []
        for ntet in range(len(self.tetrode_numbers_int)):
            # Create a list where each element lists the channels on that tetrode
            string = 'T' + self.tetrode_numbers_str[ntet] + ' C ' + \
                     ' '.join(self.tetrode_channels_str[ntet])
            string = string.replace('[','')
            string = string.replace(']','')
            itemstrings.append(string)
        # Populate the listbox
        self.lw_tetrodes.addItems(itemstrings)
        # Connect the tetrode selection action to function update_plots()
        self.lw_tetrodes.itemSelectionChanged.connect(lambda:self.update_plots())
        
        
    def next_channel(self):
        # Moves the selection in the listbox to next item, circles back to front
        ntet = self.lw_tetrodes.currentRow()
        max_rows = len(self.tetrode_numbers_int)
        new_row = ntet + 1
        if new_row > max_rows - 1:
            new_row = 0
        self.lw_tetrodes.setCurrentRow(new_row)
        
        
    def previous_channel(self):
        # Moves the selectioni in the listbox to the previous item, circles back to the end
        ntet = self.lw_tetrodes.currentRow()
        max_rows = len(self.tetrode_numbers_int)
        new_row = ntet - 1
        if new_row < 0:
            new_row = max_rows - 1
        self.lw_tetrodes.setCurrentRow(new_row)
        
        
    def get_position_data_edges(self):
        # Get position data file
        posLog_fileName = 'PosLogComb.csv'
        if os.path.exists(self.fpath + '/' + posLog_fileName):
            # Get data from CSV file
            pos_csv = np.genfromtxt(self.fpath + '/' + posLog_fileName, delimiter=',')
            pos_timestamps = np.array(pos_csv[:,0], dtype=np.float64)
            lfp_timestamps = np.arange(self.n_samples, dtype=np.float64) / self.samplingRate
            idx_first = np.abs(lfp_timestamps - pos_timestamps[0]).argmin()
            idx_last = np.abs(lfp_timestamps - pos_timestamps[-1]).argmin()
            self.pos_edges = [idx_first, idx_last]

        
    def extract_waveforms(self, ntet, ntchan):
        # Extract the waveforms for this channel from the signal based on set threshold
        # Detect spikes differently depending on whether the threshold is negative or positive
        if self.thresholds[ntet][ntchan] < 0:
            spiketimes = self.LFPs[ntet][ntchan] < np.int16(np.round((self.thresholds[ntet][ntchan] / self.bitVolts)))
            spiketimes = np.where(spiketimes)[0]
        elif self.thresholds[ntet][ntchan] > 0:
            spiketimes = self.LFPs[ntet][ntchan] > np.int16(np.round((self.thresholds[ntet][ntchan] / self.bitVolts)))
            spiketimes = np.where(spiketimes)[0]
        else:
            # These items being None are used to identify that no spikes were detected on this channels
            spiketimes = []
            self.spiketimes[ntet][ntchan] = spiketimes
        # Ignore spiketimes beofre and after position data
        if len(spiketimes) > 0:
            idx_outside = spiketimes < self.pos_edges[0]
            idx_outside = np.logical_or(idx_outside, spiketimes > self.pos_edges[1])
            if np.sum(idx_outside) > 0:
                spiketimes = spiketimes[np.logical_not(idx_outside)]
                self.spiketimes[ntet][ntchan] = spiketimes
        # Remove duplicates based on temporal proximity
        if len(spiketimes) > 0:
            tooclose = np.int32(np.round(float(self.samplingRate) * float(self.waveform_exemption)))
            spike_diff = np.concatenate((np.array([0]),np.diff(spiketimes)))
            tooclose_idx = spike_diff < tooclose
            spiketimes = np.delete(spiketimes, np.where(tooclose_idx)[0])
        if len(spiketimes) > 0:
            # Using spiketimes create an array of indices (windows) to extract waveforms from LFP trace
            spiketimes = np.int32(np.expand_dims(spiketimes, 1))
            winsize_before = np.int32(np.round((self.waveform_width[0] * self.samplingRate)))
            winsize_after = np.int32(np.round((self.waveform_width[1] * self.samplingRate)))
            windows = np.arange(winsize_before + winsize_after + 1, dtype=np.int32) - winsize_before
            windows = np.tile(windows, (spiketimes.size,1))
            windows = windows + np.tile(spiketimes, (1,windows.shape[1]))
            # Skip windows that are too close to edge of signal
            tooearly = windows < 0
            toolate = windows > (self.LFPs[ntet][ntchan].size - 1)
            idx_delete = np.any(np.logical_or(tooearly, toolate), axis=1)
            windows = np.delete(windows, np.where(idx_delete)[0], axis=0)
            spiketimes = np.delete(spiketimes, np.where(idx_delete)[0], axis=0)
            # Remove spikes that have waveforms with ridiculous amplitudes
            noise_threshold_int = np.int16(float(str(self.pt_noise_threshold.toPlainText())) * 1000 / self.bitVolts)
            idx_delete = np.zeros(spiketimes.size, dtype=bool)
            for sub_ntchan in range(len(self.LFPs[ntet])):
                waves = self.LFPs[ntet][sub_ntchan][windows]
                idx_delete = np.logical_or(idx_delete, np.any(np.abs(waves) > noise_threshold_int, axis=1))
            percentage_too_big = np.sum(idx_delete, dtype=np.float32) / idx_delete.size
            if percentage_too_big > 5:
                print('Caution! ' + str(percentage_too_big) + ' percent of waveforms in noise level')
            windows = np.delete(windows, np.where(idx_delete)[0], axis=0)
            spiketimes = np.delete(spiketimes, np.where(idx_delete)[0], axis=0)
            # Keep copy of windows in self for use when plotting or saving waveforms
            self.waveform_windows[ntet][ntchan] = windows
            self.spiketimes[ntet][ntchan] = spiketimes
        if len(spiketimes) > 0:
            self.spikes_detected[ntet][ntchan] = True
        else:
            self.spikes_detected[ntet][ntchan] = False
            print('No spikes detected on tetrode ' + str(ntet) + ' channel ' + str(ntchan))
        
        
    def plot_waveforms(self, ntet, ntchan):
        # Get number of channels that have LFP on this tetrode
        nsubchannels = len(self.LFPs[ntet])
        minvals = np.zeros(nsubchannels)
        maxvals = np.zeros(nsubchannels)
        for nsubchan in range(4):
            if nsubchan + 1 <= nsubchannels and len(self.waveform_windows[ntet][ntchan]) > 0:
                # Extract waveforms from LFP
                waveforms = np.float32(self.LFPs[ntet][nsubchan][self.waveform_windows[ntet][ntchan]]) * self.bitVolts
                minvals[nsubchan] = np.amin(waveforms)
                maxvals[nsubchan] = np.amax(waveforms)
                # Describe waveforms in form of connected points (Allows for faster plotting)
                lines = waveforms.shape[0]
                pointsPerLine = waveforms.shape[1]
                y = np.reshape(waveforms, waveforms.size)
                x = np.empty((lines, pointsPerLine))
                x[:] = np.arange(pointsPerLine)
                x = x.reshape(y.size)
                winsize_before = np.int32(np.round((self.waveform_width[0] * self.samplingRate)))
                winsize_after = np.int32(np.round((self.waveform_width[1] * self.samplingRate)))
                x = ((x - winsize_before) / self.samplingRate) * 1000 # Set to milliseconds from spike
                connect = np.ones(lines * pointsPerLine, dtype=np.ubyte)
                connect[pointsPerLine-1::pointsPerLine] = 0  #  disconnect segment between lines
                # Prepare drawing
                path = pg.arrayToQPath(x, y, connect)
                item = pg.QtGui.QGraphicsPathItem(path)
                item.setPen(pg.mkPen('w'))
                self.axesIDs[ntchan][nsubchan].setHidden(False)
                self.axesIDs[ntchan][nsubchan].clear()
                # Do the drawing
                self.axesIDs[ntchan][nsubchan].addItem(item)
                # Set X axis range
                xlims = [-winsize_before / self.samplingRate * 1000, winsize_after / self.samplingRate * 1000]
                self.axesIDs[ntchan][nsubchan].setXRange(xlims[0], xlims[1], padding=0)
                # Plot the threshold line
                self.axesIDs[ntchan][ntchan].addLine(y=self.thresholds[ntet][ntchan], pen=pg.mkPen('r'))
                # Update the title for this channel plots
                num_waveforms = self.waveform_windows[ntet][ntchan].shape[0]
                Ch_str = 'Channel: ' + self.tetrode_channels_str[ntet][ntchan] + ', waveforms: ' + str(num_waveforms)
                self.gbChIDs[ntchan].setTitle(Ch_str)
            else:
                # If no spikes were detected on this channel, hide empty plot
                self.axesIDs[ntchan][nsubchan].clear()
                self.axesIDs[ntchan][nsubchan].setHidden(True)
        # Ensure axes are shown or hidden according to the setting
        self.toggle_axes()
        # Save minimum and maximum value for all plots
        self.minvals[ntet][ntchan] = np.amin(minvals)
        self.maxvals[ntet][ntchan] = np.amax(maxvals)
        # Set Y axis range automatically if set to 0 on GUI. Otherwise use user input.
        if self.ymax_IDs[ntchan].value() == 0 or self.ymin_IDs[ntchan].value() == 0:
            self.ymin_IDs[ntchan].setValue(np.int16(self.minvals[ntet][ntchan]))
            self.ymax_IDs[ntchan].setValue(np.int16(self.maxvals[ntet][ntchan]))
        for nsubchan in range(nsubchannels):
            self.axesIDs[ntchan][nsubchan].setYRange(self.ymin_IDs[ntchan].value(), self.ymax_IDs[ntchan].value(), padding=0)

            
    def toggle_axes(self):
        # Show axes on plots if checkbox is ticket, otherwise hide axes
        if self.cb_toggle_axes.isChecked() == True:
            for ntchan in range(4):
                for nsubchan in range(4):
                    self.axesIDs[ntchan][nsubchan].showAxis('left')
                    self.axesIDs[ntchan][nsubchan].showAxis('bottom')
        elif self.cb_toggle_axes.isChecked() == False:
            for ntchan in range(4):
                for nsubchan in range(4):
                    self.axesIDs[ntchan][nsubchan].hideAxis('left')
                    self.axesIDs[ntchan][nsubchan].hideAxis('bottom')


    def update_plots(self,ntchans=range(4)):
        # Extracts waveforms for current tetrode with current thresholds and plots them
        for ntchan in ntchans:
            ntet = self.lw_tetrodes.currentRow()
            # If threshold has not been set previously, start from default
            if not self.thresholds[ntet][ntchan]:
                self.thresholds[ntet][ntchan] = np.int16(str(self.pt_default_threshold.toPlainText()))
            self.thresholdBoxes[ntchan].setValue(self.thresholds[ntet][ntchan])
            self.extract_waveforms(ntet, ntchan)
            self.plot_waveforms(ntet, ntchan)

    
    def mouseclick_on_plot(self,ntchan):
        # Get currently selected tetrode
        ntet = self.lw_tetrodes.currentRow()
        # Get Y-Coordinate in pixels
        y_coord_pix = self.axesIDs[ntchan][ntchan].lastMousePos[1]
        # Get Plot height in pixels
        tmp = self.axesIDs[ntchan][ntchan].getPlotItem()
        v = tmp.scene().views()[0]
        b = tmp.vb.mapRectToScene(tmp.vb.boundingRect())
        wr = v.mapFromScene(b).boundingRect()
        pos = v.mapToGlobal(v.pos())
        wr.adjust(pos.x(), pos.y(), pos.x(), pos.y())
        plot_height_pix = float(wr.bottom()-wr.y())
        # Compute multiplier from pixels to plot y-axis values
        y_range = self.ymax_IDs[ntchan].value() - self.ymin_IDs[ntchan].value()
        converter = y_range / plot_height_pix
        # Compute cursor location on y-axis
        y_coord = y_coord_pix * converter
        y_coord = self.ymax_IDs[ntchan].value() - y_coord
        # Change the save threshold for this channel
        self.thresholds[ntet][ntchan] = np.int16(y_coord)
        # Update plots
        self.update_plots([ntchan])
        
        
    def save_waveforms(self):
        # Save waveforms detected on all channels
        files_created = []
        for ntet in range(len(self.waveform_windows)):
            # Optain spiketimes and waveform windows for all spikes detected on the tetrode
            spiketimes = self.spiketimes[ntet]
            waveform_windows = self.waveform_windows[ntet]
            no_spikes_chans = np.where(np.invert(np.array(self.spikes_detected[ntet])))[0]
            for sub_nchan in no_spikes_chans[::-1]:
                del spiketimes[sub_nchan]
                del waveform_windows[sub_nchan]
            # Only continue if spikes were detected on any of the channels on this tetrode
            if len(spiketimes) > 0:
                spiketimes = np.concatenate(spiketimes, axis=0)
                waveform_windows = np.concatenate(waveform_windows, axis=0)
                # Sort spikes in the order of occurrence
                idx = np.argsort(spiketimes, axis=0)[:,0]
                spiketimes = spiketimes[idx]
                waveform_windows = waveform_windows[idx,:]
                # Remove likely duplicates based on temporal proximity
                tooclose = np.int32(np.round(float(self.samplingRate) * float(self.waveform_exemption)))
                idx_tooclose = np.array([1], dtype=bool)
                while np.any(idx_tooclose):
                    diff_spiketimes = np.diff(spiketimes, axis=0)
                    idx_tooclose = diff_spiketimes <= tooclose
                    idx_tooclose = np.concatenate((np.zeros((1,1), dtype=bool), idx_tooclose), axis=0)
                    spiketimes = np.delete(spiketimes, np.where(idx_tooclose)[0], axis=0)
                    waveform_windows = np.delete(waveform_windows, np.where(idx_tooclose)[0], axis=0)
                # Extract waveforms for all channels
                waveforms = np.zeros((waveform_windows.shape[0],waveform_windows.shape[1],len(self.LFPs[ntet])), dtype=np.int16)
                for ntchan in range(len(self.LFPs[ntet])):
                    waveforms[:,:,ntchan] = self.LFPs[ntet][ntchan][waveform_windows]
                    chan_nr = self.tetrode_channels_int[ntet][ntchan]
                # Save waveforms to file
                # Create name for saved file, combining all channel numbers
                filename = self.fileNames[self.channel_numbers_int.index(chan_nr)]
                numstart = filename.find('_CH') + 3
                channel_nrs_str = '_'.join(self.tetrode_channels_str[ntet])
                filename = filename[:numstart] + channel_nrs_str + '.waveforms'
                full_filename = self.fpath + '/' + filename
                # Create a dictionary to hold all the infromation to be saved
                waveform_data = {'waveforms': waveforms, 'spiketimes': spiketimes, 
                                 'nr_tetrode': self.tetrode_numbers_int[ntet], 
                                 'tetrode_channels': self.tetrode_channels_int[ntet], 
                                 'lfp_filename': full_filename, 
                                 'thresholds': self.thresholds[ntet], 
                                 'high_pass_frequency': self.highpass_frequency, 
                                 'waveform_width': self.waveform_width, 
                                 'waveform_exemption': self.waveform_exemption, 
                                 'badChan': self.badChan, 
                                 'sampling_rate': self.samplingRate, 
                                 'bitVolts': self.bitVolts}
                # Use cPickle to save the data
                with open(full_filename,'wb') as file:
                    pickle.dump(waveform_data, file)
                files_created.append(filename)
        # Display message listing the files saved
        show_message('Files were saved successfully to: ' + self.fpath,os.linesep.join(files_created))

        return files_created


    def plotTrace(self):
        # Plots the full recording of the specified tetrode channel in a new window.
        # The trace is filtered and/or referenced as specified by ticks.
        # Allows for multiple traces from different tetrodes and channels:
        # To load channel 2 from tetrode 3 and channel 1 from tetrode 8, use the following input
        # Set Tet: text box to 2,8 and Ch: text box to 3,1

        # Get user input
        ntet_str = str(self.pt_lfp_tet.toPlainText())
        nchan_str = str(self.pt_lfp_ch.toPlainText())
        if ntet_str.find(',') > -1: # If more than one channel specified
            # Find all values tetrode and channel values listed
            ntet_str = ntet_str.split(',')
            nchan_str = nchan_str.split(',')
        else:
            ntet_str = [ntet_str]
            nchan_str = [nchan_str]
        # Compute timestamps
        timestamps = np.arange(self.n_samples, dtype=np.float64) / self.samplingRate
        # Prepare plot
        tracesPlot = pg.plot()
        linecolors = ['c', 'g', 'b', 'y', 'r', 'm', 'w']
        for plot_chan_nr in range(len(ntet_str)):
            # Plot data for all requested tetrode channels
            ntet = int(ntet_str[plot_chan_nr]) - 1
            nchan = int(nchan_str[plot_chan_nr]) - 1
            print('Plotting channel ' + self.tetrode_channels_str[ntet][nchan])
            lfp_trace = np.float32(self.LFPs[ntet][nchan]) * self.bitVolts
            tracesPlot.plot(timestamps, lfp_trace, pen=linecolors[plot_chan_nr])

    def start_klusta(self):
        # Start ApplyKlustakwikGUI and load waveforms edited in this session
        files_created = self.save_waveforms() # Resave all waveforms and get list of file names
        from ApplyKlustakwikGUI import ApplyKlustakwikGUI
        self.AKKGUI = ApplyKlustakwikGUI(self)
        self.AKKGUI.show()
        self.AKKGUI.fpath = self.fpath
        self.AKKGUI.fileNames = files_created
        self.AKKGUI.load_waveforms()        
        

# The following is the default ending for a QtGui application script
def main():
    app = QtGui.QApplication(sys.argv)
    form = DetectWaveforms()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main()