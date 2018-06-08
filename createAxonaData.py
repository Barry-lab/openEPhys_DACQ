### This script creates Waveform GUI compatible files for all selected .waveform files.

### Note, speed data on Waveform GUI will be incorrect, due to issue with conversion of
### the unit of position data. Spatial correlograms work fine though.

### By Sander Tanni, April 2017, UCL

import sys
import os
import numpy as np
import shutil
from scipy import interpolate
import subprocess
import NWBio
import HelperFunctions as hfunct
from datetime import datetime
import argparse


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

def create_DACQ_waveform_data_for_single_tetrode(tet_waveform_data, pos_edges):
    # Create DACQ data tetrode format
    waveform_data_dacq = []
    dacq_waveform_dtype = [('ts', '>i'), ('waveform', '50b')]
    dacq_waveform_waves_dtype = '>i1'
    dacq_waveform_timestamps_dtype = '>i'
    dacq_sampling_rate = 96000
    # Align timestamps to beginning of position data
    tet_waveform_data['timestamps'] = tet_waveform_data['timestamps'] - pos_edges[0]
    # Get waveforms
    waves = np.array(tet_waveform_data['waveforms'], dtype=np.float32)
    waves = -waves
    waves = np.swapaxes(waves,1,2)
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
    waveform_data_dacq = np.zeros(nspikes * 4, dtype=dacq_waveform_dtype)
    # Input waveform values, leaving a trailing end of zeros due to lower sampling rate
    waves = waves.astype(dtype=dacq_waveform_waves_dtype)
    waveform_data_dacq['waveform'][:,:waves.shape[1]] = waves
    # Arrange timestamps into a vector where timestamp for a spike is
    # repeated for each of the 4 channels
    timestamps = np.array(tet_waveform_data['timestamps'])
    timestamps = np.ravel(np.repeat(np.reshape(timestamps, (len(timestamps), 1)), 4, axis=1),'C')
    # Convert OpenEphys timestamp sampling rate to DACQ sampling rate
    timestamps = timestamps * float(dacq_sampling_rate)
    timestamps_dacq = np.round(timestamps).astype(dtype=dacq_waveform_timestamps_dtype)
    # Input timestamp values to the dacq data matrix
    waveform_data_dacq['ts'] = timestamps_dacq
    
    return waveform_data_dacq

def create_DACQ_waveform_data(waveform_data, pos_edges):
    print('Converting Waveforms...')
    input_args = []
    for tet_waveform_data in waveform_data:
        input_args.append((tet_waveform_data, pos_edges))
    multiprocessor = hfunct.multiprocess()
    waveform_data_dacq = multiprocessor.map(create_DACQ_waveform_data_for_single_tetrode, input_args)
    print('Converting Waveforms Successful.')

    return waveform_data_dacq

def create_DACQ_pos_data(OpenEphysDataPath):
    posdata = NWBio.load_tracking_data(OpenEphysDataPath, subset='ProcessedPos')
    xy_pos = posdata[:,1:5].astype(np.float32)
    timestamps = posdata[:,0].astype(np.float32)
    # Realign position data start to 0
    timestamps = timestamps - timestamps[0]
    # Create DACQ data pos format
    dacq_pos_samplingRate = 50 # Sampling rate in Hz
    dacq_pos_dtype = [('ts', '>i'), ('pos', '>8h')]
    # dacq_pos_timestamp_dtype = '>i4'
    dacq_pos_xypos_dtype = '>i2'
    dacq_pos_timestamp_dtype = '>i'
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


def create_DACQ_eeg_data(fpath, OpenEphys_SamplingRate, dacq_eeg_samplingRate, lowpass_frequency, pos_edges, eegChan, bitVolts=0.195):
    '''
    EEG is downsampled to dacq_eeg_samplingrate and inverted to same polarity as spikes in AxonaFormat.
    EEG is also rescaled to microvolt values.
    '''
    # Load EEG data of selected channel
    continuous_data = NWBio.load_continuous(fpath)
    if not (continuous_data is None):
        timestamps = np.array(continuous_data['timestamps'])# Lowpass filter data
        data = np.array(continuous_data['continuous'][:,eegChan], dtype=np.float64)
        data = hfunct.butter_lowpass_filter(data, sampling_rate=30000.0, lowpass_frequency=lowpass_frequency, filt_order=4)
    else:
        # If no raw data available, load downsampled data for that tetrode
        lowpass_data = NWBio.load_tetrode_lowpass(fpath)
        timestamps = lowpass_data['tetrode_lowpass_timestamps']
        data = lowpass_data['tetrode_lowpass'][:,hfunct.channels_tetrode(eegChan)]
        lowpass_downsampling = int([i for i in  lowpass_data['tetrode_lowpass_info'] if 'downsampling ' in i][0][13:])
        OpenEphys_SamplingRate = OpenEphys_SamplingRate / lowpass_downsampling
        lowpass_data_filter_frequency = float([i for i in  lowpass_data['tetrode_lowpass_info'] if 'lowpass_frequency ' in i][0][18:])
        if lowpass_data_filter_frequency > lowpass_frequency:
            data = hfunct.butter_lowpass_filter(data, sampling_rate=OpenEphys_SamplingRate, lowpass_frequency=lowpass_frequency, filt_order=4)
    # Resample data to dacq_eeg sampling rate
    data = data[::int(np.round(OpenEphys_SamplingRate/dacq_eeg_samplingRate))]
    # Invert data
    data = -data * bitVolts
    # Crop data outside position data
    idx_outside_pos_data = timestamps < pos_edges[0]
    idx_outside_pos_data = np.logical_or(idx_outside_pos_data, timestamps > pos_edges[1])
    idx_outside_pos_data = np.where(idx_outside_pos_data)[0]
    data = np.delete(data, idx_outside_pos_data, 0)
    # Adjust EEG data format and range
    data = data - np.mean(data)
    data = data / 4000 # Set data range to between 4000 microvolts
    data = data * 127
    data[data > 127] = 127
    data[data < -127] = -127
    # Create DACQ data eeg format
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
    if NWBio.check_if_settings_available(fpath,'/General/animal/'):
        animal = NWBio.load_settings(fpath,'/General/animal/')
    else:
        animal = 'unknown'
    if NWBio.check_if_settings_available(fpath,'/Time/'):
        full_timestring = NWBio.load_settings(fpath,'/Time/')
        timeobject = datetime.strptime(full_timestring, '%Y-%m-%d_%H-%M-%S')
    else:
        timeobject = datetime.now()
    datestring = timeobject.strftime('%d-%m-%y')
    timestring = timeobject.strftime('%H-%M-%S')
    experiment_info = {'trial_date': datestring, 
                       'trial_time': timestring, 
                       'animal': animal}
    
    return experiment_info

def createAxonaData(OpenEphysDataPath, waveform_data=None, spike_name='spikes', subfolder='AxonaData', eegChan=1, open_output_folder=True):
    if waveform_data is None:
        print('Loading waveform data from file')
        waveform_data = NWBio.load_spikes(OpenEphysDataPath, spike_name=spike_name, use_idx_keep=True, use_badChan=True)
    else:
        # Ensure idx_keep has been applied to incoming data
        for spike_data_tet in waveform_data:
            if spike_data_tet['idx_keep'].size == spike_data_tet['waveforms'].shape[0]:
                spike_data_tet['waveforms'] = spike_data_tet['waveforms'][spike_data_tet['idx_keep'], :, :]
                spike_data_tet['timestamps'] = spike_data_tet['timestamps'][spike_data_tet['idx_keep']]
    print('Converting data')
    # Get position data start and end times
    pos_edges = NWBio.get_processed_tracking_data_timestamp_edges(OpenEphysDataPath)
    # Convert data to DACQ format
    waveform_data_dacq = create_DACQ_waveform_data(waveform_data, pos_edges)
    print('Converting position data')
    pos_data_dacq = create_DACQ_pos_data(OpenEphysDataPath)
    OpenEphys_SamplingRate = 30000
    dacq_eeg_samplingRate = 250 # Sampling rate in Hz
    lowpass_frequency = 125.0
    print('Converting LFP to EEG data')
    eeg_data_dacq = create_DACQ_eeg_data(OpenEphysDataPath, OpenEphys_SamplingRate, dacq_eeg_samplingRate, lowpass_frequency, pos_edges, eegChan)
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
    AxonaDataPath = os.path.join(os.path.dirname(OpenEphysDataPath), subfolder)
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
    print('Writing CLU files')
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
    if open_output_folder:
        subprocess.Popen(['xdg-open', AxonaDataPath])

# The following is the default ending for a QtGui application script
if __name__ == "__main__":
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Export data into Axona format.')
    parser.add_argument('paths', type=str, nargs='*', 
                        help='recording data folder(s) (can enter multiple paths separated by spaces to KlustaKwik simultaneously)')
    parser.add_argument('--eegChan', type=int, nargs = 1, 
                        help='enter channel number to use for creating EEG data')
    parser.add_argument('--subfolder', type=str, nargs = 1, 
                        help='enter the name of subfolder to use for AxonaData')
    parser.add_argument('--spike_name', type=str, nargs = 1, 
                        help='enter the name of spike set in NWB file to use for AxonaData')
    args = parser.parse_args()
    # Get paths to recording files
    OpenEphysDataPaths = args.paths
    if isinstance(OpenEphysDataPaths, basestring):
        OpenEphysDataPaths = [OpenEphysDataPaths]
    # If directories entered as paths, attempt creating path to file by appending experiment_1.nwb
    for ndata, OpenEphysDataPath in enumerate(OpenEphysDataPaths):
        if not os.path.isfile(OpenEphysDataPath):
            new_path = os.path.join(OpenEphysDataPath, 'experiment_1.nwb')
            if os.path.isfile(new_path):
                OpenEphysDataPaths[ndata] = new_path
            else:
                raise ValueError('The following path does not lead to a NWB data file:\n' + OpenEphysDataPath)
    # Get optional arguemtn values
    if args.eegChan:
        eegChan = args.eegChan[0]
    else:
        eegChan = 1
    if args.subfolder:
        subfolder = args.subfolder[0]
    else:
        subfolder = 'AxonaData'
    if args.spike_name:
        spike_name = args.spike_name[0]
    else:
        spike_name = 'spikes'
    # Run main script
    for OpenEphysDataPath in OpenEphysDataPaths:
        createAxonaData(OpenEphysDataPath, spike_name=spike_name, subfolder=subfolder, eegChan=eegChan)
