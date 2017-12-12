### This script creates Waveform GUI compatible files for all selected .waveform files.

### Note, speed data on Waveform GUI will be incorrect, due to issue with conversion of
### the unit of position data. Spatial correlograms work fine though.

### By Sander Tanni, April 2017, UCL

import sys
from PyQt4 import QtGui, QtCore
import ntpath
import os
import numpy as np
import cPickle as pickle
import re
import shutil
from scipy import interpolate
from scipy.spatial.distance import euclidean
import subprocess
import NWBio
import HelperFunctions as hfunct


# def getAllFiles(fpath, file_basenames):
#     # Find .clu and position files and incorporate into dictionary of fileNames
#     fileNames = {'waveforms': [None] * len(file_basenames), 'clufiles': [None] * len(file_basenames), 
#                  'posfile': [None]}
#     for nfile in range(len(file_basenames)):
#         waveforms_string = file_basenames[nfile] + '.spikes.p'
#         fileNames['waveforms'][nfile] = waveforms_string
#     # Get all available clu file names
#     for nfile in range(len(file_basenames)):
#         clu_string = file_basenames[nfile] + '.clu.0'
#         if os.path.exists(fpath + '/' + clu_string):
#             fileNames['clufiles'][nfile] = clu_string
#     # Get position data file
#     posLog_fileName = 'PosLogComb.csv'
#     if os.path.exists(fpath + '/' + posLog_fileName):
#         'PosLogComb.csv' = posLog_fileName
        
#     return fileNames
    
    
def interpolate_waveforms(waves, nr_targetbins=50):
    # Waveforms are interpolated to 50 Hz, assuming the original waveforms were 1000 ms long
    original_bins = np.linspace(0, 1, num=waves.shape[1], dtype=np.float32)
    target_bins = np.linspace(0, 1, num=nr_targetbins, dtype=np.float32)
    new_waves = np.zeros((waves.shape[0], nr_targetbins), dtype=np.int8)
    for nwave in range(waves.shape[0]):
        # Interpolate each waveform
        interfunct = interpolate.interp1d(original_bins, waves[nwave,:])
        new_waves[nwave,:] = interfunct(target_bins)
        
    return new_waves
    
    
def create_DACQ_waveform_data(waveform_data, pos_edges):
    # Create DACQ data tetrode format
    waveform_data_dacq = []
    dacq_waveform_dtype = [('ts', '>i'), ('waveform', '50b')]
    # dacq_waveform_timestamps_dtype = '>i4'
    dacq_waveform_waves_dtype = '>i1'
    dacq_waveform_timestamps_dtype = '>i'
    # dacq_waveform_waves_dtype = '50b'
    dacq_sampling_rate = 96000
    hfunct.print_progress(0, len(waveform_data), prefix = 'Converting Waveforms:', initiation=True)
    for ntet in range(len(waveform_data)):
        tet_waveform_data = waveform_data[ntet]
        # Align spiketimes to beginning of position data
        tet_waveform_data['spiketimes'] = tet_waveform_data['spiketimes'] - pos_edges[0]
        # Get waveforms
        waves = np.array(tet_waveform_data['waveforms'], dtype=np.float32)
        nspikes = waves.shape[0]
        # Set waveforms values on this tetrode to range -127 to 127
        if nspikes > 1:
            mean_waves = np.mean(waves, axis=0)
            wave_peak = mean_waves.max()
            maxchan = np.where(mean_waves == wave_peak)[1][0]
            maxidx = np.where(mean_waves == wave_peak)[0][0]
            wave_peak_std = np.std(waves[:,maxidx,maxchan])
            max_range = wave_peak + 4 * wave_peak_std # This sets int8 range
            waves = waves / max_range
            waves = waves * 127
            waves[waves > 127] = 127
            waves[waves < -127] = -127
        waves = waves.astype(np.int8)
        # Where channels are missing, add 0 values to waveform values
        if waves.shape[2] < 4:
            waves = np.concatenate((waves, np.zeros((waves.shape[0], waves.shape[1], 4 - waves.shape[2]))), axis=2)
        # Reshape 3D waveform matrix into 2D matrix such that waveforms for 
        # first spike from all four channels are on consecutive rows.
        waves = np.reshape(np.ravel(np.transpose(waves, (0, 2, 1)), 'C'), (nspikes * 4, waves.shape[1]))
        # Interpolate waveforms to 48000 Hz resolution
        waves = interpolate_waveforms(waves=waves, nr_targetbins=50)
        # Create DACQ datatype structured array
        tmp_waveform_data_dacq = np.zeros(nspikes * 4, dtype=dacq_waveform_dtype)
        # Input waveform values, leaving a trailing end of zeros due to lower sampling rate
        waves = waves.astype(dtype=dacq_waveform_waves_dtype)
        tmp_waveform_data_dacq['waveform'][:,:waves.shape[1]] = waves
        # Arrange timestamps into a vector where timestamp for a spike is
        # repeated for each of the 4 channels
        timestamps = np.array(tet_waveform_data['spiketimes'])
        timestamps = np.ravel(np.repeat(np.reshape(timestamps, (len(timestamps), 1)), 4, axis=1),'C')
        # Convert OpenEphys timestamp sampling rate to DACQ sampling rate
        timestamps = timestamps * float(dacq_sampling_rate)
        timestamps_dacq = np.round(timestamps).astype(dtype=dacq_waveform_timestamps_dtype)
        # Input timestamp values to the dacq data matrix
        tmp_waveform_data_dacq['ts'] = timestamps_dacq
        waveform_data_dacq.append(tmp_waveform_data_dacq)
        hfunct.print_progress(ntet, len(waveform_data), prefix = 'Converting Waveforms:')
        
    return waveform_data_dacq
    
    
def create_DACQ_pos_data(posfile):
    # Create DACQ data pos format
    dacq_pos_samplingRate = 50 # Sampling rate in Hz
    dacq_pos_dtype = [('ts', '>i'), ('pos', '>8h')]
    # dacq_pos_timestamp_dtype = '>i4'
    dacq_pos_xypos_dtype = '>i2'
    dacq_pos_timestamp_dtype = '>i'
    # dacq_pos_xypos_dtype = '>8h'
    # Get data from CSV file
    pos_csv = np.genfromtxt(posfile, delimiter=',')
    timestamps = np.array(pos_csv[:,0], dtype=np.float32)
    xy_pos = pos_csv[:,1:5]
    # Realign position data start to 0
    timestamps = timestamps - timestamps[0]
    # Interpolate position data to 50Hz
    countstamps = np.arange(np.floor(timestamps[-1] * dacq_pos_samplingRate))
    dacq_timestamps = countstamps / dacq_pos_samplingRate
    xy_pos_interp = np.zeros((countstamps.size, xy_pos.shape[1]), dtype=np.float32)
    for ncoord in range(xy_pos.shape[1]):
        xy_pos_interp[:,ncoord] = np.interp(dacq_timestamps, timestamps, xy_pos[:,ncoord])
    xy_pos = xy_pos_interp
    # Set position data to range between 0 and 600 (upsamples data for smaller arenas)
    # Also set NaN values to 1023
    xy_pos = xy_pos - np.nanmin(xy_pos)
    xy_pos = xy_pos * (600 / np.nanmax(xy_pos))
    xy_pos[np.isnan(xy_pos)] = 1023
    xy_pos = np.int16(np.round(xy_pos))
    # Convert datatypes to DACQ data type
    countstamps = countstamps.astype(dtype=dacq_pos_timestamp_dtype)
    xy_pos = xy_pos.astype(dtype=dacq_pos_xypos_dtype)
    # Arrange xy_pos as in the DACQ data matrix including the dot size values
    xy_pos = np.concatenate((xy_pos, 20 * np.ones((xy_pos.shape[0], 1), dtype=dacq_pos_xypos_dtype)), axis=1)
    xy_pos = np.concatenate((xy_pos, 10 * np.ones((xy_pos.shape[0], 1), dtype=dacq_pos_xypos_dtype)), axis=1)
    xy_pos = np.concatenate((xy_pos, 30 * np.ones((xy_pos.shape[0], 1), dtype=dacq_pos_xypos_dtype)), axis=1)
    xy_pos = np.concatenate((xy_pos, np.zeros((xy_pos.shape[0], 1), dtype=dacq_pos_xypos_dtype)), axis=1)
    # Create DACQ datatype structured array
    pos_data_dacq = np.zeros(countstamps.size, dtype=dacq_pos_dtype)
    # Input timestamps and pos_xy to DACQ array
    pos_data_dacq['ts'] = countstamps
    pos_data_dacq['pos'] = xy_pos
    
    return pos_data_dacq

def create_DACQ_eeg_data(fpath, OpenEphys_SamplingRate, dacq_eeg_samplingRate, pos_edges, eegChan):
    # Load EEG data of second channel
    data = np.array(NWBio.load_continuous(os.path.join(fpath,'experiment_1.nwb'))['continuous'][:,eegChan])
    timestamps = np.array(NWBio.load_continuous(os.path.join(fpath,'experiment_1.nwb'))['timestamps'])
    # Crop data outside position data
    idx_outside_pos_data = timestamps < pos_edges[0]
    idx_outside_pos_data = np.logical_or(idx_outside_pos_data, timestamps > pos_edges[1])
    idx_outside_pos_data = np.where(idx_outside_pos_data)[0]
    data = np.delete(data, idx_outside_pos_data, 0)
    # Adjust EEG data format and range
    data = data.astype(np.float32)
    data = hfunct.Filter(data, sampling_rate=30000, highpass_frequency=1, lowpass_frequency=300)
    data = data - np.mean(data)
    data = data / 2000 # Set data range to between 1000 microvolts
    data = data * 127
    data[data > 127] = 127
    data[data < -127] = -127
    # Create DACQ data eeg format
    data = data[::int(np.round(OpenEphys_SamplingRate/dacq_eeg_samplingRate))]
    dacq_eeg_dtype = [('eeg', '=b')]
    dacq_eeg_data_dtype = '=b'
    dacq_eeg = data.astype(dtype=dacq_eeg_data_dtype)
    eeg_data_dacq = np.zeros(dacq_eeg.size, dtype=dacq_eeg_dtype)
    eeg_data_dacq['eeg'] = dacq_eeg

    return eeg_data_dacq


def header_templates(htype):
    # This function returns the default header necessary for the waveform and pos files
    if htype == 'waveforms':
        keyorder = ['trial_date', 'trial_time', 'experimenter', 'comments', \
              'duration', 'sw_version', 'num_chans', 'timebase', \
              'bytes_per_timestamp', 'samples_per_spike', 'sample_rate', \
              'bytes_per_sample', 'spike_format', 'num_spikes']
            
        header = {'trial_date': 'Thursday, 3 Mar 2016', 
                  'trial_time': '11:28:24', 
                  'experimenter': 'ST', 
                  'comments': 'R2410', 
                  'duration': '1', 
                  'sw_version': '1.2.2.14', 
                  'num_chans': '4', 
                  'timebase': '96000 hz', 
                  'bytes_per_timestamp': '4', 
                  'samples_per_spike': '50', 
                  'sample_rate': '48000 hz',
                  'bytes_per_sample': '1', 
                  'spike_format': 't,ch1,t,ch2,t,ch3,t,ch4', 
                  'num_spikes': '1'}
    elif htype == 'pos':
        keyorder = ['trial_date', 'trial_time', 'experimenter', 'comments', \
              'duration', 'sw_version', 'num_colours', 'min_x', 'max_x', \
              'min_y', 'max_y', 'window_min_x', 'window_max_x', \
              'window_min_y', 'window_max_y', 'timebase', \
              'bytes_per_timestamp', 'sample_rate', \
              'EEG_samples_per_position', 'bearing_colour_1', \
              'bearing_colour_2', 'bearing_colour_3', 'bearing_colour_4', \
              'pos_format', 'bytes_per_coord', 'pixels_per_metre', \
              'num_pos_samples']
            
        header = {'trial_date': 'Thursday, 3 Mar 2016', 
                  'trial_time': '11:28:24', 
                  'experimenter': 'ST', 
                  'comments': 'R2410', 
                  'duration': '1', 
                  'sw_version': '1.2.2.14', 
                  'num_colours': '4', 
                  'min_x': '0', 
                  'max_x': '600', 
                  'min_y': '0', 
                  'max_y': '600', 
                  'window_min_x':'0', 
                  'window_max_x': '600', 
                  'window_min_y': '0', 
                  'window_max_y': '600', 
                  'timebase': '50 hz', 
                  'bytes_per_timestamp': '4', 
                  'sample_rate': '50.0 hz', 
                  'EEG_samples_per_position': '5', 
                  'bearing_colour_1': '0', 
                  'bearing_colour_2': '0', 
                  'bearing_colour_3': '0', 
                  'bearing_colour_4': '0', 
                  'pos_format': 't,x1,y1,x2,y2,numpix1,numpix2', 
                  'bytes_per_coord': '2', 
                  'pixels_per_metre': '600', 
                  'num_pos_samples': '1'}
    elif htype == 'eeg':
        keyorder = ['trial_date', 'trial_time', 'experimenter', 'comments', \
              'duration', 'sw_version', 'num_chans', 'sample_rate', 'EEG_samples_per_position', \
              'bytes_per_sample', 'num_EEG_samples']

        header = {'trial_date': 'Tuesday, 14 Dec 2010', 
                  'trial_time': '10:44:07', 
                  'experimenter': 'cb', 
                  'comments': '1838', 
                  'duration': '1201', 
                  'sw_version': '1.0.2', 
                  'num_chans': '1', 
                  'sample_rate': '250.0 hz', 
                  'EEG_samples_per_position': '5', 
                  'bytes_per_sample': '1', 
                  'num_EEG_samples': '300250'}
                  
    return header, keyorder
    
    
def getExperimentInfo(fpath):
    # This function gets some information on the currently selected recording
    rootfolder_name = os.path.split(fpath)[1]
    trial_time = rootfolder_name[rootfolder_name.find('_') + 1:]
    datefolder = os.path.split(os.path.split(fpath)[0])[1]
    animalfolder = os.path.split(os.path.split(os.path.split(fpath)[0])[0])[1]
    
    experiment_info = {'trial_date': datefolder, 
                       'trial_time': trial_time, 
                       'animal': animalfolder}
    
    return experiment_info


def get_speed_data(posfile):
    pos_csv = np.genfromtxt(posfile, delimiter=',')
    pos_timestamps = np.array(pos_csv[:,0], dtype=np.float32)
    xy_pos = pos_csv[:,1:3]
    # Compute distance travelled between each pos datapoint
    distance = np.zeros(xy_pos.shape[0], dtype=np.float32)
    for npos in range(xy_pos.shape[0] - 1):
        distance[npos] = euclidean(xy_pos[npos,:], xy_pos[npos + 1,:])
    # Compute total distance traveled in 1 second around each pos datapoint
    winsize = float(1.0)
    distance_sum = np.zeros(distance.shape, dtype=np.float32)
    startpos = np.where(np.diff(pos_timestamps > (pos_timestamps[0] + (winsize / 2))))[0][0] + 1
    endpos = np.where(np.diff(pos_timestamps < (pos_timestamps[-1] - (winsize / 2))))[0][0]
    for npos in range(distance.size)[startpos:endpos]:
        # Find beginning and end of window
        pos_time = pos_timestamps[npos]
        time_differences = np.abs(pos_timestamps - pos_time)
        window_edges = np.where(np.diff(time_differences > winsize / 2))[0]
        distance_sum[npos] = np.sum(distance[window_edges[0]:window_edges[1]])
    distance_sum[:startpos] = distance_sum[startpos]
    distance_sum[endpos:] = distance_sum[endpos - 1]
    # Convert distance covered to speed cm/s
    speed = distance_sum / winsize

    return pos_timestamps, speed


def createAxonaData(OpenEphysDataPath, waveform_data, subfolder='AxonaData', eegChan=1):
    print('Converting data')
    # if type(fileNames) is not str:
    #     # Loads waveforms from Pickle files selected with the openFileDialog
    #     waveform_data = []
    #     for filename in fileNames['waveforms']:
    #         # Load all selected waveform files
    #         full_filename = fpath + '/' + filename
    #         with open(full_filename, 'rb') as file:
    #             tmp = pickle.load(file)
    #         waveform_data.append(tmp)
    #     # Extract tetrode numbers and order data by tetrode numbers
    #     tetrode_numbers_int = []
    #     for wavedat in waveform_data:
    #         tetrode_numbers_int.append(wavedat['nr_tetrode'])
    #     file_order = np.argsort(np.array(tetrode_numbers_int))
    #     fileNames['waveforms'] = [fileNames['waveforms'][x] for x in file_order]
    #     waveform_data = [waveform_data[x] for x in file_order]
    #     fileNames['clufiles'] = [fileNames['clufiles'][x] for x in file_order]
    # Get position data start and end times
    pos_edges = hfunct.get_position_data_edges(OpenEphysDataPath)
    # Convert data to DACQ format
    waveform_data_dacq = create_DACQ_waveform_data(waveform_data, pos_edges)
    pos_data_dacq = create_DACQ_pos_data(os.path.join(OpenEphysDataPath,'PosLogComb.csv'))
    OpenEphys_SamplingRate = 30000
    dacq_eeg_samplingRate = 250 # Sampling rate in Hz
    eeg_data_dacq = create_DACQ_eeg_data(OpenEphysDataPath, OpenEphys_SamplingRate, dacq_eeg_samplingRate, pos_edges, eegChan)
    # Get headers for both datatypes
    header_wave, keyorder_wave = header_templates('waveforms')
    header_pos, keyorder_pos = header_templates('pos')
    header_eeg, keyorder_eeg = header_templates('eeg')
    # Update headers
    experiment_info = getExperimentInfo(OpenEphysDataPath)
    trial_duration = str(int(float(len(eeg_data_dacq['eeg'])) / dacq_eeg_samplingRate))
    header_wave['trial_date'] = experiment_info['trial_date']
    header_wave['trial_time'] = experiment_info['trial_time']
    header_wave['comments'] = experiment_info['animal']
    header_wave['duration'] = trial_duration
    nspikes_tet = []
    for waves in waveform_data_dacq:
        nspikes_tet.append(str(int(len(waves) / 4)))
    # header_pos['window_min_x'] = str(int(np.min(pos_data_dacq['pos'][:,0])))
    # header_pos['window_max_x'] = str(int(np.max(pos_data_dacq['pos'][:,0])))
    # header_pos['window_min_y'] = str(int(np.min(pos_data_dacq['pos'][:,1])))
    # header_pos['window_max_y'] = str(int(np.max(pos_data_dacq['pos'][:,1])))
    header_pos['trial_date'] = experiment_info['trial_date']
    header_pos['trial_time'] = experiment_info['trial_time']
    header_pos['comments'] = experiment_info['animal']
    header_pos['duration'] = trial_duration
    header_pos['num_pos_samples'] = str(len(pos_data_dacq))
    header_eeg['trial_date'] = experiment_info['trial_date']
    header_eeg['trial_time'] = experiment_info['trial_time']
    header_eeg['comments'] = experiment_info['animal']
    header_eeg['duration'] = trial_duration
    header_eeg['EEG_samples_per_position'] = str(int(np.round(len(eeg_data_dacq) / len(pos_data_dacq))))
    header_eeg['num_EEG_samples'] = str(len(eeg_data_dacq))
    # Generate base name for files
    file_basename = 'experiment_1'
    # Set data start and end tokens
    DATA_START_TOKEN = 'data_start'
    DATA_END_TOKEN = '\r\ndata_end\r\n'
    # Create subdirectory or rewrite existing
    AxonaDataPath = os.path.join(OpenEphysDataPath, subfolder)
    if not os.path.exists(AxonaDataPath):
        os.mkdir(AxonaDataPath)
    # Write WAVEFORM data for each tetrode into DACQ format
    hfunct.print_progress(0, len(waveform_data_dacq), prefix = 'Writing tetrode files:', initiation=True)
    for ntet in range(len(waveform_data_dacq)):
        fname = os.path.join(AxonaDataPath, file_basename + '.' + str(waveform_data[ntet]['nr_tetrode'] + 1))
        with open(fname, 'wb') as f:
            # Write header in the correct order
            for key in keyorder_wave:
                if 'num_spikes' in key:
                    # Replicate spaces following num_spikes in original dacq files
                    stringval = nspikes_tet[ntet]
                    while len(stringval) < 10:
                        stringval += ' '
                    f.write(key + ' ' + stringval + '\r\n')
                elif 'duration' in key:
                    # Replicate spaces following duration in original dacq files
                    stringval = header_wave[key]
                    while len(stringval) < 10:
                        stringval += ' '
                    f.write(key + ' ' + stringval + '\r\n')
                else:
                    f.write(key + ' ' + header_wave[key] + '\r\n')
            # Write the start token string
            f.write(DATA_START_TOKEN)
            # Write the data into the file in binary format
            waveform_data_dacq[ntet].tofile(f)
            # Write the end token string
            f.write(DATA_END_TOKEN)
        hfunct.print_progress(ntet + 1, len(waveform_data_dacq), prefix = 'Writing tetrode files:')
    # Write POSITION data into DACQ format
    fname = os.path.join(AxonaDataPath, file_basename + '.pos')
    with open(fname, 'wb') as f:
        # Write header in the correct order
        for key in keyorder_pos:
            if 'num_pos_samples' in key:
                # Replicate spaces following num_pos_samples in original dacq files
                stringval = header_pos[key]
                while len(stringval) < 10:
                    stringval += ' '
                f.write(key + ' ' + stringval + '\r\n')
            elif 'duration' in key:
                # Replicate spaces following duration in original dacq files
                stringval = header_pos[key]
                while len(stringval) < 10:
                    stringval += ' '
                f.write(key + ' ' + stringval + '\r\n')
            else:
                f.write(key + ' ' + header_pos[key] + '\r\n')
        # Write the start token string
        f.write(DATA_START_TOKEN)
        # Write the data into the file in binary format
        pos_data_dacq.tofile(f)
        # Write the end token string
        f.write(DATA_END_TOKEN)
    # Write EEG data into DACQ format
    fname = os.path.join(AxonaDataPath, file_basename + '.eeg')
    with open(fname, 'wb') as f:
        # Write header in the correct order
        for key in keyorder_eeg:
            if 'duration' in key:
                # Replicate spaces following duration in original dacq files
                stringval = header_eeg[key]
                while len(stringval) < 10:
                    stringval += ' '
                f.write(key + ' ' + stringval + '\r\n')
            else:
                f.write(key + ' ' + header_eeg[key] + '\r\n')
        # Write the start token string
        f.write(DATA_START_TOKEN)
        # Write the data into the file in binary format
        eeg_data_dacq.tofile(f)
        # Write the end token string
        f.write(DATA_END_TOKEN)
    # Write CLU files
    for ntet in range(len(waveform_data_dacq)):
        clufileName = os.path.join(AxonaDataPath, file_basename + '.clu.' + str(waveform_data[ntet]['nr_tetrode'] + 1))
        lines = [str(waveform_data[ntet]['clusterIDs'].size) + '\r\n']
        for nclu in list(waveform_data[ntet]['clusterIDs']):
            lines.append(str(nclu) + '\r\n')
        with open(clufileName, 'wb') as file:
                file.writelines(lines)
    print('Waveform data was generated for ' + str(len(waveform_data_dacq)) + ' tetrodes.')
    # Make a copy of a random .set file into the converted data folder
    sourcefile = 'SetFileBase.set'
    fname = os.path.join(AxonaDataPath, file_basename + '.set')
    shutil.copy(sourcefile, fname)
    # Rewrite the .set file to correct the trial duration
    with open(fname, 'rb') as file:
        lines = file.readlines()
    lines[4] = lines[4][:9] + trial_duration + lines[4][9 + len(trial_duration):]
    with open(fname, 'wb') as file:
        file.writelines(lines)
    # Opens recording folder with Ubuntu file browser
    subprocess.Popen(['xdg-open', AxonaDataPath])
        
        
class MainWindow(QtGui.QWidget):

    def __init__(self):
        # create GUI
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle('File picker')
        # Set the window dimensions
        self.resize(300,75)
        # vertical layout for widgets
        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)
        # Create a label which displays the path to our chosen file
        self.lbl = QtGui.QLabel('Select one of the waveform files of interest.\n' + \
                                'All necessary files for Waveform GUI will be \n' + \
                                'created and put into subdirectory WaveformGUI.')
        self.vbox.addWidget(self.lbl)
        # Create text box to display selected path
        self.pt_fpath = QtGui.QPlainTextEdit()
        self.pt_fpath.readOnly = True
        self.vbox.addWidget(self.pt_fpath)
        # Create a push button labelled 'choose' and add it to our layout
        self.pb_choose_file = QtGui.QPushButton('Choose file on path', self)
        self.vbox.addWidget(self.pb_choose_file)
        # Connect the clicked signal to the get_fname handler
        self.connect(self.pb_choose_file, QtCore.SIGNAL('clicked()'), self.get_fname)
        # Add spin box for showing speed limit
        self.spinbox = QtGui.QSpinBox()
        self.spinbox.setPrefix('Speed cut: ')
        self.spinbox.setSuffix(' cm/s')
        self.vbox.addWidget(self.spinbox)
        # Add button to display speed data
        self.pb_display_speed = QtGui.QPushButton('Display speed data', self)
        self.vbox.addWidget(self.pb_display_speed)
        # Connect the clicked signal to the display speed plot function
        self.connect(self.pb_display_speed, QtCore.SIGNAL('clicked()'), self.plot_speed)
        # Create a push button labelled 'Create Data' and add it to our layout
        self.pb_create_data = QtGui.QPushButton('Create Waveform GUI data', self)
        self.vbox.addWidget(self.pb_create_data)
        # Connect the clicked signal to the get_fname handler
        self.connect(self.pb_create_data, QtCore.SIGNAL('clicked()'), self.create_data)


    def get_fname(self):
        # Pops up a GUI to select a single file. All others with same prefix will be loaded
        dialog = QtGui.QFileDialog(self, caption='Select one from the set of waveform files in the recording folder')
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        dialog.setNameFilters(['(*.spikes.p)'])
        dialog.setViewMode(QtGui.QFileDialog.List) # or Detail
        if dialog.exec_():
            # Get path and file name of selection
            tmp = dialog.selectedFiles()
            selected_file = str(tmp[0])
            self.fpath = ntpath.dirname(selected_file)
            self.pt_fpath.setPlainText(self.fpath)
            f_s_name = str(ntpath.basename(selected_file))
            # Check if file name is correct format
            numstart = f_s_name.find('_CH') + 3
            numend = f_s_name.find('.spikes.p')
            if numstart == -1 or numend == -1:
                print('Error: Unexpected file name')
            else:
                # Get prefix for this set of LFPs
                fprefix = f_s_name[:f_s_name.index('_')]
                # Get list of all other files and channel numbers from this recording
                self.fileNames = []
                for fname in os.listdir(self.fpath):
                    if fname.startswith(fprefix) and fname.endswith('.spikes.p'):
                        self.fileNames.append(fname[:-9])
            self.fileNames = getAllFiles(self.fpath, self.fileNames)


    def create_data(self):
        createAxonaData(fpath=self.fpath, fileNames=self.fileNames)
        app.instance().quit()


    def plot_speed(self):
        import pyqtgraph as pg
        tracesPlot = pg.plot()
        pos_timestamps, speed = get_speed_data(self.fpath + '/' + 'PosLogComb.csv')
        tracesPlot.plot(pos_timestamps, speed)
        pen = pg.mkPen('y', width=3, style=QtCore.Qt.DashLine)
        

# The following is the default ending for a QtGui application script
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    gui = MainWindow()
    gui.show()
    app.exec_()