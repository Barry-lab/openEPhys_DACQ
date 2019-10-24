'''
This script creates Waveform GUI compatible files for all selected .waveform files.

Note, absolute values of position data and therefore speed information will be incorrect, 
due to issue with conversion of position data. Spatial correlograms work fine though.
'''

import os
import numpy as np
from scipy import interpolate
import subprocess
import openEPhys_DACQ.NWBio as NWBio
import openEPhys_DACQ.HelperFunctions as hfunct
from datetime import datetime
import argparse

from openEPhys_DACQ.package_configuration import package_path


def AxonaDataEEG_SamplingRate():
    return 250

def AxonaDataEGF_SamplingRate():
    return 4800

def interpolate_waveforms(waves, input_sampling_frequency=30000, 
                          output_sampling_frequency=48000, output_timestemps=50):
    '''
    Resamples waves to output_sampling_frequency.
    Input waves must have enough timesteps to cover output_timesteps.
    '''
    input_sample_step = 1000.0 / float(input_sampling_frequency)
    original_bins = np.arange(0.0, input_sample_step * waves.shape[1], 
                              input_sample_step).astype(np.float32)
    output_sample_step = 1000.0 / float(output_sampling_frequency)
    target_bins = np.arange(0.0, output_sample_step * output_timestemps, 
                            output_sample_step).astype(np.float32)
    if target_bins[-1] > original_bins[-1]:
        raise ValueError('Input waves do not have enough samples to interpolate requested output.')
    new_waves = np.zeros((waves.shape[0], output_timestemps), dtype=np.int8)
    for nwave in range(waves.shape[0]):
        # Interpolate each waveform
        interfunct = interpolate.interp1d(original_bins, waves[nwave,:].astype(np.float32))
        new_waves[nwave,:] = np.int8(np.round(interfunct(target_bins)))

    return new_waves

def create_DACQ_waveform_data_for_single_tetrode(spike_data_tet, data_time_edges, 
                                                 input_sampling_frequency=30000, 
                                                 output_sampling_frequency=48000, 
                                                 output_timestemps=50):
    '''
    spike_data - dict - with fields:
                 'waveforms' - numpy array of shape (n_spikes, 4, n_datapoints)
                               waveforms for all spikes on the 4 channels of the tetrode.
                 'timestamps' - numpy one dimensional array with spike times in seconds
    data_time_edges - tuple with two elements: start and end time of data (in seconds).
                      This is used with timestamps to discard spikes outside data range.

    Default values of the following are compatible with AxonaData:
        input_sampling_frequency - int or float - sampling frequency of input waveforms
        output_sampling_frequency - int or float - target sampling frequency of waveforms
        output_timestemps - int or float - number of timesteps required in output data

    Note: The length of input waveforms must be at least as long as output waveforms.
          In case of default values for output sampling frequency and timesteps, 
          the input waveforms need to be at least 1.03 milliseconds long.
    '''
    # Create DACQ data tetrode format
    waveform_data_dacq = []
    dacq_waveform_dtype = [('ts', '>i'), ('waveform', '50b')]
    dacq_waveform_waves_dtype = '>i1'
    dacq_waveform_timestamps_dtype = '>i'
    dacq_sampling_rate = 96000
    # Align timestamps to start from 0 at data_time_edges[0]
    spike_data_tet['timestamps'] = spike_data_tet['timestamps'] - data_time_edges[0]
    # Get waveforms
    waves = np.array(spike_data_tet['waveforms'], dtype=np.float32)
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
    waves = interpolate_waveforms(waves, input_sampling_frequency, 
                                  output_sampling_frequency, output_timestemps)
    # Create DACQ datatype structured array
    waveform_data_dacq = np.zeros(nspikes * 4, dtype=dacq_waveform_dtype)
    # Input waveform values, leaving a trailing end of zeros due to lower sampling rate
    waves = waves.astype(dtype=dacq_waveform_waves_dtype)
    waveform_data_dacq['waveform'][:,:waves.shape[1]] = waves
    # Arrange timestamps into a vector where timestamp for a spike is
    # repeated for each of the 4 channels
    timestamps = np.array(spike_data_tet['timestamps'])
    timestamps = np.ravel(np.repeat(np.reshape(timestamps, (len(timestamps), 1)), 4, axis=1),'C')
    # Convert OpenEphys timestamp sampling rate to DACQ sampling rate
    timestamps = timestamps * float(dacq_sampling_rate)
    timestamps_dacq = np.round(timestamps).astype(dtype=dacq_waveform_timestamps_dtype)
    # Input timestamp values to the dacq data matrix
    waveform_data_dacq['ts'] = timestamps_dacq
    
    return waveform_data_dacq

def create_DACQ_waveform_data(spike_data, data_time_edges, 
                              input_sampling_frequency=30000, 
                              output_sampling_frequency=48000, 
                              output_timestemps=50):
    '''
    This is a function to create_AxonaData waveforms using multiprocessing.
    See create_DACQ_waveform_data_for_single_tetrode() for description of input arguments.

    spike_data - list of spike_data_tet - see create_DACQ_waveform_data_for_single_tetrode()
    '''
    input_args = []
    for spike_data_tet in spike_data:
        input_args.append((spike_data_tet, data_time_edges, input_sampling_frequency, 
                           output_sampling_frequency, output_timestemps))
    multiprocessor = hfunct.multiprocess()
    waveform_data_dacq = multiprocessor.map(create_DACQ_waveform_data_for_single_tetrode, 
                                            len(input_args), args_list=input_args, 
                                            max_memory_usage=0.66)

    return waveform_data_dacq

def AxonaData_max_pix_value():
    return 600

def convert_position_data_from_cm_to_pixels(xy_pos):
    '''
    Converts cm positions to pixel values in AxonaData acceptable range
    Computes the conversion pixels_per_metre value
    '''
    max_pix = AxonaData_max_pix_value()
    xy_pos = xy_pos - np.nanmin(xy_pos)
    max_cm = np.nanmax(xy_pos)
    pix_per_cm = float(max_pix) / float(max_cm)
    xy_pos = xy_pos * pix_per_cm
    #pixels_per_metre = int(np.round(pix_per_cm * 100))
    pixels_per_metre = max_pix

    return xy_pos, pixels_per_metre

def create_DACQ_pos_data(posdata, data_time_edges=None, pixels_per_metre=None, dacq_pos_samplingRate=50):
    '''
    posdata - numpy array as type np.float32 with shape (N x 5) - 
              where there are N samples and colums are:
              timestamp (s), xpos1, ypos1, xpos2, ypos2
              Missing datapoints should be NaN. If no second LED, then NaN columns.

    data_time_edges - tuple with two elements: start and end time of synthetic position data (in seconds)
    
    If pixels_per_metre is entered, it is assumed that ProcessedPos data is
    in pixel values, not centimeters. Otherwise, position data values are scaled 
    into range [0, AxonaData_max_pix_value()] and pixels_per_metre set to AxonaData_max_pix_value().

    dacq_pos_samplingRate - float or int - sampling rate of AxonaData position data
                            default value is compatible with AxonaData.
    '''
    # Create DACQ data pos format
    dacq_pos_dtype = [('ts', '>i'), ('pos', '>8h')]
    # dacq_pos_timestamp_dtype = '>i4'
    dacq_pos_xypos_dtype = '>i2'
    dacq_pos_timestamp_dtype = '>i'
    if posdata is None:
        print('Warning! No position data provided, creating fake position data for AxonaData.')
        if data_time_edges is None:
            raise ValueError('data_time_edges as start and end of recording in seconds \n' + \
                             'is required to create synthetic data')
        countstamps = np.arange(np.floor((data_time_edges[1] - data_time_edges[0]) * float(dacq_pos_samplingRate)))
        dacq_timestamps = np.float64(countstamps) / float(dacq_pos_samplingRate)
        xy_pos = np.repeat(np.linspace(0, 100, dacq_timestamps.size)[:, None], 4, axis=1).astype(np.float32)
    else:
        # Extract xy values and timestamps from posdata
        xy_pos = posdata[:,1:5].astype(np.float32)
        timestamps = posdata[:,0].astype(np.float32)
        # Crop position data outside data_time_edges
        idx_outside_data_time = timestamps < data_time_edges[0]
        idx_outside_data_time = np.logical_or(idx_outside_data_time, timestamps > data_time_edges[1])
        idx_outside_data_time = np.where(idx_outside_data_time)[0]
        timestamps = np.delete(timestamps, idx_outside_data_time, 0)
        xy_pos = np.delete(xy_pos, idx_outside_data_time, 0)
        # Realign position data start to 0 at data_time_edges[0]
        timestamps = timestamps - data_time_edges[0]
        # Set minumum value of x and y to be 0
        x_min = np.nanmin(xy_pos[:, 0])
        y_min = np.nanmin(xy_pos[:, 1])
        if not all(np.isnan(xy_pos[:, 2])):
            x_min = min(x_min, np.nanmin(xy_pos[:, 2]))
        if not all(np.isnan(xy_pos[:, 3])):
            y_min = min(y_min, np.nanmin(xy_pos[:, 3]))
        xy_pos[:, 0] = xy_pos[:, 0] - x_min
        xy_pos[:, 2] = xy_pos[:, 2] - x_min
        xy_pos[:, 1] = xy_pos[:, 1] - y_min
        xy_pos[:, 3] = xy_pos[:, 3] - y_min
        # Interpolate position data to 50Hz
        countstamps = np.arange(np.floor(timestamps[-1] * float(dacq_pos_samplingRate)))
        dacq_timestamps = np.float64(countstamps) / float(dacq_pos_samplingRate)
        xy_pos_interp = np.zeros((dacq_timestamps.size, xy_pos.shape[1]), dtype=np.float32)
        for ncoord in range(xy_pos.shape[1]):
            xy_pos_interp[:,ncoord] = np.interp(dacq_timestamps, timestamps, xy_pos[:,ncoord])
        xy_pos = xy_pos_interp
    # If no pixels_per_metre provided, convert position data to pixel values
    # and calculate pixels_per_metre for range 600
    if pixels_per_metre is None:
        xy_pos, pixels_per_metre = convert_position_data_from_cm_to_pixels(xy_pos)
    # Round values and turn into integers
    xy_pos = np.int16(np.round(xy_pos))
    # Find minimum and maximum values for x and y values
    limits = {'x_max': max([np.nanmax(xy_pos[:, 0]), np.nanmax(xy_pos[:, 2])]), 
              'x_min': min([np.nanmin(xy_pos[:, 0]), np.nanmin(xy_pos[:, 2])]), 
              'y_max': max([np.nanmax(xy_pos[:, 1]), np.nanmax(xy_pos[:, 3])]), 
              'y_min': min([np.nanmin(xy_pos[:, 1]), np.nanmin(xy_pos[:, 3])])}
    # Set NaN values to 1023
    xy_pos[np.isnan(xy_pos)] = 1023
    # Convert datatypes to DACQ data type
    dacq_timestamps = dacq_timestamps.astype(dtype=dacq_pos_timestamp_dtype)
    xy_pos = xy_pos.astype(dtype=dacq_pos_xypos_dtype)
    # Arrange xy_pos as in the DACQ data matrix including the dot size values
    xy_pos = np.concatenate((xy_pos, 20 * np.ones((xy_pos.shape[0], 1), dtype=dacq_pos_xypos_dtype)), axis=1)
    xy_pos = np.concatenate((xy_pos, 10 * np.ones((xy_pos.shape[0], 1), dtype=dacq_pos_xypos_dtype)), axis=1)
    xy_pos = np.concatenate((xy_pos, 30 * np.ones((xy_pos.shape[0], 1), dtype=dacq_pos_xypos_dtype)), axis=1)
    xy_pos = np.concatenate((xy_pos, np.zeros((xy_pos.shape[0], 1), dtype=dacq_pos_xypos_dtype)), axis=1)
    # Create DACQ datatype structured array
    pos_data_dacq = np.zeros(dacq_timestamps.size, dtype=dacq_pos_dtype)
    # Input timestamps and pos_xy to DACQ array
    pos_data_dacq['ts'] = dacq_timestamps
    pos_data_dacq['pos'] = xy_pos

    return pos_data_dacq, pixels_per_metre, limits

def create_DACQ_eeg_or_egf_data(eeg_or_egf, data, data_time_edges, target_range=1000):
    '''
    EEG is lowpass filtered to half the target sampling rate, 
    downsampled to dacq_eeg_samplingrate and 
    inverted to same polarity as spikes in AxonaFormat.
    The data is also clipped to specified range in values and time.

    The returned array is in correct format to written to binary Axona file.

    eeg_or_egf - str - 'eeg' or 'egf' specifies ouput data format and sampling rate
    data - dict with following fields:
           'data' - numpy  array in dtype=numpy.float32 with dimensions (N x n_chan) - 
                  LFP data to convert.
           'timestamps' - numpy one dimensional array in dtype=numpy.float32 - 
                        timestamps in seconds for each of the datapoints in data
           'sampling_rate' - int or float - sampling rate of data
    data_time_edges - tuple with two elements: start and end time of data (in seconds).
                      This is used with timestamps to crop EEG data outside data range.
    target_range - int or float - EEG data with voltage values above this will be clipped.
    '''
    # AxonaData eeg data parameters
    if eeg_or_egf == 'eeg':
        output_SamplingRate = AxonaDataEEG_SamplingRate()
    elif eeg_or_egf == 'egf':
        output_SamplingRate = AxonaDataEGF_SamplingRate()
    if output_SamplingRate > data['sampling_rate']:
        raise ValueError('Input data sampling rate is lower than requested output data.')
    if data['data'].dtype  != np.float32:
        raise ValueError('Input data dtype is not numpy.float32.')
    lowpass_frequency = output_SamplingRate / 2.0
    # Filter data with lowpass butter filter
    data_in_processing = []
    for n_chan in range(data['data'].shape[1]):
        data_in_processing.append(hfunct.butter_lowpass_filter(data['data'][:, n_chan].copy(), 
                                                               sampling_rate=float(data['sampling_rate']), 
                                                               lowpass_frequency=lowpass_frequency, 
                                                               filt_order=4))
    # Crop data outside data_time_edges
    idx_outside_data_time = data['timestamps'] < data_time_edges[0]
    idx_outside_data_time = np.logical_or(idx_outside_data_time, data['timestamps'] > data_time_edges[1])
    idx_outside_data_time = np.where(idx_outside_data_time)[0]
    cropped_timestamps = np.delete(data['timestamps'].copy(), idx_outside_data_time, 0)
    for n_chan in range(len(data_in_processing)):
        data_in_processing[n_chan] = np.delete(data_in_processing[n_chan], idx_outside_data_time, 0)
    # Resample data to dacq_eeg sampling rate
    original_timestamps = (cropped_timestamps - cropped_timestamps[0])
    target_timestamps = np.arange(0, original_timestamps[-1], 1.0 / float(output_SamplingRate))
    for n_chan in range(len(data_in_processing)):
        interfunct = interpolate.interp1d(original_timestamps, data_in_processing[n_chan])
        data_in_processing[n_chan] = interfunct(target_timestamps)
    # Invert data
    for n_chan in range(len(data_in_processing)):
        data_in_processing[n_chan] = -data_in_processing[n_chan]
    # Adjust EEG data format and range
    for n_chan in range(len(data_in_processing)):
        data_in_processing[n_chan] = data_in_processing[n_chan] - np.mean(data_in_processing[n_chan])
    for n_chan in range(len(data_in_processing)):
        data_in_processing[n_chan] = data_in_processing[n_chan] / target_range
    for n_chan in range(len(data_in_processing)):
        data_in_processing[n_chan] = data_in_processing[n_chan] * 127
    for n_chan in range(len(data_in_processing)):
        data_in_processing[n_chan][data_in_processing[n_chan] > 127] = 127
        data_in_processing[n_chan][data_in_processing[n_chan] < -127] = -127
    # Create DACQ data eeg format
    if eeg_or_egf == 'eeg':
        dacq_eeg_dtype = [('eeg', '=b')]
        dacq_eeg_data_dtype = '=b'
    elif eeg_or_egf == 'egf':
        dacq_eeg_dtype = [('eeg', np.int16)]
        dacq_eeg_data_dtype = np.int16
    eeg_data_dacq = [None] * len(data_in_processing)
    for n_chan in range(len(data_in_processing)):
        dacq_eeg = data_in_processing[n_chan].astype(dtype=dacq_eeg_data_dtype)
        eeg_data_dacq[n_chan] = np.zeros(dacq_eeg.size, dtype=dacq_eeg_dtype)
        eeg_data_dacq[n_chan]['eeg'] = dacq_eeg

    return eeg_data_dacq

def create_DACQ_eeg_data(data, data_time_edges, target_range=1000):
    return create_DACQ_eeg_or_egf_data('eeg', data, data_time_edges, 
                                       target_range=target_range)

def create_DACQ_egf_data(data, data_time_edges, target_range=1000):
    return create_DACQ_eeg_or_egf_data('egf', data, data_time_edges, 
                                       target_range=target_range)

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
                  'num_spikes': '0'}

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
                  'max_x': str(AxonaData_max_pix_value()), 
                  'min_y': '0', 
                  'max_y': str(AxonaData_max_pix_value()), 
                  'window_min_x':'0', 
                  'window_max_x': str(AxonaData_max_pix_value()), 
                  'window_min_y': '0', 
                  'window_max_y': str(AxonaData_max_pix_value()), 
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
                  'pixels_per_metre': '100', 
                  'num_pos_samples': '0'}
    
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
                  'num_EEG_samples': '0'}
    
    elif htype == 'egf':
        keyorder = ['trial_date', 'trial_time', 'experimenter', 'comments', \
              'duration', 'sw_version', 'num_chans', 'sample_rate', \
              'bytes_per_sample', 'num_EGF_samples']

        header = {'trial_date': 'Tuesday, 14 Dec 2010', 
                  'trial_time': '10:44:07', 
                  'experimenter': 'cb', 
                  'comments': '1838', 
                  'duration': '1201', 
                  'sw_version': '1.0.2', 
                  'num_chans': '1', 
                  'sample_rate': '4800 hz', 
                  'bytes_per_sample': '2', 
                  'num_EGF_samples': '0'}
                  
    return header, keyorder

def update_header(header, experiment_info, new_values_dict):
    header['trial_date'] = experiment_info['trial_date']
    header['trial_time'] = experiment_info['trial_time']
    header['comments'] = experiment_info['animal']
    header['duration'] = experiment_info['duration']
    for key in new_values_dict.keys():
        header[key] = new_values_dict[key]

    return header

def getExperimentInfo(fpath=None):
    '''
    Extract experiment info from NWB file and sets to correc tformat for
    createAxonaData() function.

    If fpath is not provided, sham data is provided that is compatible
    with createAxonaData() function.
    '''
    if not (fpath is None) and NWBio.check_if_settings_available(fpath,'/General/animal/'):
        animal = NWBio.load_settings(fpath,'/General/animal/')
    else:
        animal = 'unknown'
    if not (fpath is None) and NWBio.check_if_settings_available(fpath,'/Time/'):
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

def get_first_available_spike_data(OpenEphysDataPath, tetrode_nrs, use_idx_keep, use_badChan, 
                                   clustering_name=None):
    _, spike_names = NWBio.processing_method_and_spike_name_combinations()
    spike_data_available = False
    for spike_name in spike_names:
        spike_data = NWBio.load_spikes(OpenEphysDataPath, tetrode_nrs=tetrode_nrs, 
                                       spike_name=spike_name, use_idx_keep=use_idx_keep, 
                                       use_badChan=use_badChan, clustering_name=clustering_name)
        if len(spike_data) > 0:
            spike_data_available = True
            break
    if not spike_data_available:
        raise Exception('Spike data is not available in file: ' + OpenEphysDataPath + '\nChecked for following spike_name: ' + str(spike_names))

    return spike_data

def write_file_in_axona_format(filename, header, header_keyorder, data):
    '''
    Writes data in axona format
    '''
    # Set data start and end tokens
    DATA_START_TOKEN = 'data_start'
    DATA_END_TOKEN = '\r\ndata_end\r\n'
    with open(filename, 'wb') as f:
        # Write header in the correct order
        for key in header_keyorder:
            if 'num_spikes' in key:
                # Replicate spaces following num_spikes in original dacq files
                stringval = header[key]
                while len(stringval) < 10:
                    stringval += ' '
                f.write(hfunct.encode_bytes(key + ' ' + stringval + '\r\n'))
            elif 'num_pos_samples' in key:
                # Replicate spaces following num_pos_samples in original dacq files
                stringval = header[key]
                while len(stringval) < 10:
                    stringval += ' '
                f.write(hfunct.encode_bytes(key + ' ' + stringval + '\r\n'))
            elif 'duration' in key:
                # Replicate spaces following duration in original dacq files
                stringval = header[key]
                while len(stringval) < 10:
                    stringval += ' '
                f.write(hfunct.encode_bytes(key + ' ' + stringval + '\r\n'))
            else:
                f.write(hfunct.encode_bytes(key + ' ' + header[key] + '\r\n'))
        # Write the start token string
        f.write(hfunct.encode_bytes(DATA_START_TOKEN))
        # Write the data into the file in binary format
        data.tofile(f)
        # Write the end token string
        f.write(hfunct.encode_bytes(DATA_END_TOKEN))


def write_clusterIDs_in_CLU_format(clusterIDs, cluFileName):
    lines = [hfunct.encode_bytes(str(max(clusterIDs)) + '\r\n')]
    for nclu in list(clusterIDs):
        lines.append(hfunct.encode_bytes(str(nclu) + '\r\n'))
    with open(cluFileName, 'wb') as file:
        file.writelines(lines)

def write_set_file(setFileName, new_values_dict):
    sourcefile = os.path.join(package_path, 'Utils', 'SetFileBase.set')
    # Read in base .set file
    with open(sourcefile, 'rb') as file:
        lines = file.readlines()
    # Correct lines based on new_values_dict
    for key in new_values_dict.keys():
        for nl, line in enumerate(lines):
            if hfunct.encode_bytes(key + ' ') in line:
                lines[nl] = hfunct.encode_bytes(key + ' ' + new_values_dict[key] + '\r\n')
                break
    # Write the .set file with corrected lines
    with open(setFileName, 'wb') as file:
        file.writelines(lines)

def load_eegData(fpath, eegChans, bitVolts=0.195):
    # Load EEG data of selected channel
    print([hfunct.time_string(), 'DEBUG: load from file'])
    continuous_data = NWBio.load_continuous_as_array(fpath, eegChans)
    if not (continuous_data is None):
        print([hfunct.time_string(), 'DEBUG: extract timestamps'])
        timestamps = np.array(continuous_data['timestamps'])
        print([hfunct.time_string(), 'DEBUG: transforming into float32'])
        data = continuous_data['continuous'].astype(np.float32)
        sampling_rate = NWBio.OpenEphys_SamplingRate()
    else:
        # If no raw data available, load downsampled data for that tetrode
        lowpass_data = NWBio.load_tetrode_lowpass(fpath)
        timestamps = np.array(lowpass_data['tetrode_lowpass_timestamps'])
        sampling_rate = NWBio.OpenEphys_SamplingRate() / float(lowpass_downsampling)
        if not isinstance(eegChans, list):
            eegChans = [eegChans]
        tetrodes = []
        for eegChan in eegChans:
            tetrodes.append(hfunct.channels_tetrode(eegChan))
        data = np.array(lowpass_data['tetrode_lowpass'][:, tetrodes]).astype(np.float32)
    # Scale data with bitVolts value to convert from raw data to voltage values
    print([hfunct.time_string(), 'DEBUG: converting to bitVolts'])
    data = data * bitVolts

    return {'data': data, 'timestamps': timestamps, 'sampling_rate': sampling_rate}

def createAxonaData_for_NWBfile(OpenEphysDataPath, spike_name='first_available', channel_map=None, 
                                subfolder='AxonaData', eegChans=None, pixels_per_metre=None, 
                                show_output=False, clustering_name=None):
    # Construct path for AxonaData and get experiment info
    AxonaDataPath = os.path.join(os.path.dirname(OpenEphysDataPath), subfolder)
    experiment_info = getExperimentInfo(OpenEphysDataPath)
    # Get channel_map for this dataset
    if channel_map is None:
        if NWBio.check_if_settings_available(OpenEphysDataPath,'/General/channel_map/'):
            channel_map = NWBio.load_settings(OpenEphysDataPath,'/General/channel_map/')
        else:
            raise ValueError('Channel map could not be generated. Enter channels to process.')
    # Get position data for this recording
    data_time_edges = NWBio.get_processed_tracking_data_timestamp_edges(OpenEphysDataPath)
    if NWBio.check_if_processed_position_data_available(OpenEphysDataPath):
        posdata = NWBio.load_processed_tracking_data(OpenEphysDataPath)
    else:
        posdata = None
    # Create AxonaData separately for each recording area
    for area in channel_map.keys():
        # Load spike data
        tetrode_nrs = hfunct.get_tetrode_nrs(channel_map[area]['list'])
        print('Loading spikes for tetrodes nr: ' +  ', '.join(map(str, tetrode_nrs)))
        if spike_name == 'first_available':
            spike_data = get_first_available_spike_data(OpenEphysDataPath, tetrode_nrs, 
                                                        use_idx_keep=True, use_badChan=True,
                                                        clustering_name=clustering_name)
        else:
            spike_data = NWBio.load_spikes(OpenEphysDataPath, tetrode_nrs=tetrode_nrs, 
                                           spike_name=spike_name, use_idx_keep=True, 
                                           use_badChan=True, clustering_name=clustering_name)
        # Load eeg data
        if eegChans is None:
            eegData = None
        else:
            eegChansInArea_Bool = [x in channel_map[area]['list'] for x in eegChans]
            if any(eegChansInArea_Bool):
                eegChansInArea = [x for (x,y) in zip(eegChans, eegChansInArea_Bool) if y]
                print('Loading LFP data for channels: ' +  ', '.join(map(str, eegChansInArea)))
                eegData = load_eegData(OpenEphysDataPath, eegChansInArea)
            else:
                eegData = None
        createAxonaData(AxonaDataPath, spike_data, data_time_edges, posdata=posdata, 
                        experiment_info=experiment_info, axona_file_name=area, 
                        eegData=eegData, pixels_per_metre=pixels_per_metre, 
                        show_output=show_output)

def concatenate_posdata_across_recordings(posdata, data_time_edges, recording_edges):
    # Only keep timestamps and x,y for led1 and led2
    posdata = [x[:, :5] for x in posdata]
    for n_rec in range(len(posdata)):
        # Crop data outside data_time_edges and transform timestamps to continuous recording
        idx_outside_data_time = posdata[n_rec][:, 0] < data_time_edges[n_rec][0]
        idx_outside_data_time = np.logical_or(idx_outside_data_time, 
                                              posdata[n_rec][:, 0] > data_time_edges[n_rec][1])
        idx_outside_data_time = np.where(idx_outside_data_time)[0]
        posdata[n_rec] = np.delete(posdata[n_rec], idx_outside_data_time, axis=0)
        posdata[n_rec][:, 0] = posdata[n_rec][:, 0] - data_time_edges[n_rec][0] + recording_edges[n_rec][0]
    # Concatenate posdata
    posdata = np.concatenate(posdata, axis=0)

    return posdata

def concatenate_spike_data_across_recordings(spike_data, data_time_edges, recording_edges):
    new_spike_data = [None] * len(spike_data[0])
    for n_tet in range(len(new_spike_data)):
        print([hfunct.time_string(), 'DEBUG: concatenating data for tetrode ', n_tet])
        waveforms = [data[n_tet]['waveforms'] for data in spike_data]
        clusterIDs = [data[n_tet]['clusterIDs'] for data in spike_data]
        timestamps = [data[n_tet]['timestamps'] for data in spike_data]
        for n_rec in range(len(timestamps)):
            # Transform timestamps to continuous recordings
            timestamps[n_rec] = timestamps[n_rec] - data_time_edges[n_rec][0] + recording_edges[n_rec][0]
        print([hfunct.time_string(), 'DEBUG: The concatenation'])
        new_spike_data[n_tet] = {'waveforms': np.concatenate(waveforms, axis=0), 
                                 'clusterIDs': np.concatenate(clusterIDs, axis=0), 
                                 'timestamps': np.concatenate(timestamps, axis=0)}

    return new_spike_data

def concatenate_eegData_across_recordings(eegData, data_time_edges, recording_edges):
    for n_rec in range(len(eegData)):
        # Crop data outside data_time_edges and transform timestamps to continuous recording
        idx_outside_data_time = eegData[n_rec]['timestamps'] < data_time_edges[n_rec][0]
        idx_outside_data_time = np.logical_or(idx_outside_data_time, 
                                              eegData[n_rec]['timestamps'] > data_time_edges[n_rec][1])
        idx_outside_data_time = np.where(idx_outside_data_time)[0]
        eegData[n_rec]['timestamps'] = np.delete(eegData[n_rec]['timestamps'], idx_outside_data_time, axis=0)
        eegData[n_rec]['timestamps'] = eegData[n_rec]['timestamps'] - data_time_edges[n_rec][0] + recording_edges[n_rec][0]
        eegData[n_rec]['data'] = np.delete(eegData[n_rec]['data'], idx_outside_data_time, axis=0)
    # Concatenate data and timestamps
    print([hfunct.time_string(), 'DEBUG: The concatenation'])
    new_eegData = {'data': np.concatenate([x['data'] for x in eegData], axis=0), 
                   'timestamps': np.concatenate([x['timestamps'] for x in eegData], axis=0), 
                   'sampling_rate': eegData[0]['sampling_rate']}

    return new_eegData

def createAxonaData_for_multiple_NWBfiles(OpenEphysDataPaths, AxonaDataPath, 
                                          spike_name='first_available', channel_map=None, 
                                          eegChans=None, pixels_per_metre=None, 
                                          show_output=False, clustering_name=None):
    # Get experiment info
    if len(OpenEphysDataPaths) > 1:
        print('Using experiment_info from first recording only.')
    experiment_info = getExperimentInfo(OpenEphysDataPaths[0])
    # Get channel_map for this dataset
    if channel_map is None:
        if len(OpenEphysDataPaths) > 1:
            print('Using channel_map from first recording only.')
        if NWBio.check_if_settings_available(OpenEphysDataPaths[0],'/General/channel_map/'):
            channel_map = NWBio.load_settings(OpenEphysDataPaths[0],'/General/channel_map/')
        else:
            raise ValueError('Channel map could not be generated. Enter channels to process.')
    # Compute start and end times of each segment of the recording
    data_time_edges = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        data_time_edges.append(NWBio.get_processed_tracking_data_timestamp_edges(OpenEphysDataPath))
    recording_edges = []
    recording_duration = 0
    for dte in data_time_edges:
        end_of_this_recording = recording_duration + (dte[1] - dte[0])
        recording_edges.append([recording_duration, end_of_this_recording])
        recording_duration = end_of_this_recording
    combined_data_time_edges = [recording_edges[0][0], recording_edges[-1][1]]
    # Get position data for these recordings
    print('Loading position data.')
    posdata = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        if NWBio.check_if_processed_position_data_available(OpenEphysDataPath):
            posdata.append(NWBio.load_processed_tracking_data(OpenEphysDataPath))
        else:
            posdata.append(None)
    if any([x is None for x in posdata]):
        posdata = None
    else:
        posdata = concatenate_posdata_across_recordings(posdata, data_time_edges, 
                                                        recording_edges)
    # Create AxonaData separately for each recording area
    for area in channel_map.keys():
        # Load spike data
        tetrode_nrs = hfunct.get_tetrode_nrs(channel_map[area]['list'])
        print('Loading spikes for tetrodes nr: ' +  ', '.join(map(str, tetrode_nrs)))
        spike_data = []
        for OpenEphysDataPath in OpenEphysDataPaths:
            if spike_name == 'first_available':
                spike_data.append(get_first_available_spike_data(OpenEphysDataPath, tetrode_nrs, 
                                                                 use_idx_keep=True, use_badChan=True))
            else:
                print([hfunct.time_string(), 'DEBUG: loading spikes of tet ', tetrode_nrs, ' from ', OpenEphysDataPath])
                spike_data.append(NWBio.load_spikes(OpenEphysDataPath, tetrode_nrs=tetrode_nrs, 
                                                    spike_name=spike_name, use_idx_keep=True, 
                                                    use_badChan=True, clustering_name=clustering_name))
        spike_data = concatenate_spike_data_across_recordings(spike_data, data_time_edges, 
                                                              recording_edges)
        # Load eeg data
        if eegChans is None:
            eegData = None
        else:
            eegChansInArea_Bool = [x in channel_map[area]['list'] for x in eegChans]
            if any(eegChansInArea_Bool):
                eegChansInArea = [x for (x,y) in zip(eegChans, eegChansInArea_Bool) if y]
                print('Loading LFP data for channels: ' +  ', '.join(map(str, eegChansInArea)))
                eegData = []
                for OpenEphysDataPath in OpenEphysDataPaths:
                    print([hfunct.time_string(), 'DEBUG: loading eegData for ', OpenEphysDataPath])
                    eegData.append(load_eegData(OpenEphysDataPath, eegChansInArea))
                print([hfunct.time_string(), 'DEBUG: concatenating eeg data'])
                eegData = concatenate_eegData_across_recordings(eegData, data_time_edges, 
                                                                recording_edges)
            else:
                eegData = None
        createAxonaData(AxonaDataPath, spike_data, combined_data_time_edges, 
                        posdata=posdata, experiment_info=experiment_info, 
                        axona_file_name=area, eegData=eegData, 
                        pixels_per_metre=pixels_per_metre, show_output=show_output)
    with open(os.path.join(AxonaDataPath, 'recording_edges'), 'w') as file:
        for edges, OpenEphysDataPath in zip(recording_edges, OpenEphysDataPaths):
            file.write(str(edges) + ' path: ' + OpenEphysDataPath + '\n')

def createAxonaData(AxonaDataPath, spike_data, data_time_edges, posdata=None, 
                    experiment_info=None, axona_file_name='datafile', eegData=None, 
                    pixels_per_metre=None, show_output=False):
    '''
    AxonaDataPath - folder path where to store AxonaData files
    spike_data - see create_DACQ_waveform_data() for description
    posdata - create_DACQ_pos_data() for description
    data_time_edges - beginning and end time of data to convert to AxonaData. Rest is discarded.
                      posdata and eegData, if provided, must cover this range entirely.
    experiment_info - see getExperimentInfo() for description
    axona_file_name - str - filename prefix for output AxonaData files
    eegData - see create_DACQ_eeg_or_egf_data() for description
    pixels_per_metre - see create_DACQ_pos_data() for description
    show_output - bool - if True, destination folder is opened with xdg-open when finished

    Note! - timestamps info in spike_data, posdata and eegData must be aligned.
    '''
    n_tetrodes = len(spike_data)
    # Convert data to DACQ format
    print('Converting spike data')
    waveform_data_dacq = create_DACQ_waveform_data(spike_data, data_time_edges)
    print('Converting position data')
    pos_data_dacq, pixels_per_metre, limits = create_DACQ_pos_data(posdata, data_time_edges, 
                                                                   pixels_per_metre)
    if not (eegData is None):
        print('Converting LFP to EEG data')
        eeg_data_dacq = create_DACQ_eeg_data(eegData, data_time_edges)
        print('Converting LFP to EGF data')
        egf_data_dacq = create_DACQ_egf_data(eegData, data_time_edges)
    else:
        EEG_samples_per_position = (data_time_edges[1] - data_time_edges[0]) / len(pos_data_dacq)
        num_EEG_samples = int(np.round(AxonaDataEEG_SamplingRate() * (data_time_edges[1] - data_time_edges[0])))
        num_EGF_samples = int(np.round(AxonaDataEGF_SamplingRate() * (data_time_edges[1] - data_time_edges[0])))
    # Get recording specific header data for both datatypes
    if experiment_info is None:
        experiment_info = getExperimentInfo()
    experiment_info['duration'] = str(data_time_edges[1] - data_time_edges[0])
    nvds = {'num_spikes': []}
    num_spikes = []
    for ntet in range(n_tetrodes):
        nvds['num_spikes'].append(str(int(len(waveform_data_dacq[ntet]) / 4)))
    nvds['num_pos_samples'] = str(len(pos_data_dacq))
    nvds['pixels_per_metre'] = str(pixels_per_metre)
    if not (eegData is None):
        nvds['EEG_samples_per_position'] = str(int(np.round(len(eeg_data_dacq[0]) / len(pos_data_dacq))))
        nvds['num_EEG_samples'] = str(len(eeg_data_dacq[0]))
        nvds['num_EGF_samples'] = str(len(egf_data_dacq[0]))
    else:
        nvds['EEG_samples_per_position'] = EEG_samples_per_position
        nvds['num_EEG_samples'] = num_EEG_samples
        nvds['num_EGF_samples'] = num_EGF_samples
    # Get waveform file headers templates and update with recording specific data
    headers_wave = []
    for ntet in range(n_tetrodes):
        header_wave, keyorder_wave = header_templates('waveforms')
        new_values_dict = {'num_spikes': nvds['num_spikes'][ntet]}
        header_wave = update_header(header_wave, experiment_info, new_values_dict)
        headers_wave.append(header_wave)
    # Get position file header templates and update with recording specific data
    header_pos, keyorder_pos = header_templates('pos')
    new_values_dict = {'num_pos_samples': nvds['num_pos_samples'], 
                       'pixels_per_metre': nvds['pixels_per_metre'], 
                       'max_x': str(int(limits['x_max'])), 
                       'max_y': str(int(limits['y_max'])), 
                       'window_max_x': str(int(limits['x_max'])), 
                       'window_max_y': str(int(limits['y_max'])), 
                       'min_x': str(int(limits['x_min'])), 
                       'min_y': str(int(limits['y_min'])), 
                       'window_min_x': str(int(limits['x_min'])), 
                       'window_min_y': str(int(limits['y_min']))}
    header_pos = update_header(header_pos, experiment_info, new_values_dict)
    # Get EEG file header templates and update with recording specific data
    header_eeg, keyorder_eeg = header_templates('eeg')
    new_values_dict = {'EEG_samples_per_position': nvds['EEG_samples_per_position'], 
                       'num_EEG_samples': nvds['num_EEG_samples']}
    header_eeg = update_header(header_eeg, experiment_info, new_values_dict)
    # Get EGF file header templates and update with recording specific data
    header_egf, keyorder_egf = header_templates('egf')
    new_values_dict = {'num_EGF_samples': nvds['num_EGF_samples']}
    header_egf = update_header(header_egf, experiment_info, new_values_dict)
    # Create AxonaDataPath directory if it does not exist
    if not os.path.exists(AxonaDataPath):
        os.mkdir(AxonaDataPath)
    # Write WAVEFORM data for each tetrode into DACQ format
    print('Writing files to disk')
    for ntet in range(n_tetrodes):
        fname = os.path.join(AxonaDataPath, axona_file_name + '.' + str(ntet + 1))
        write_file_in_axona_format(fname, headers_wave[ntet], keyorder_wave, waveform_data_dacq[ntet])
    # Write POSITION data into DACQ format
    fname = os.path.join(AxonaDataPath, axona_file_name + '.pos')
    write_file_in_axona_format(fname, header_pos, keyorder_pos, pos_data_dacq)
    # Write EEG data into DACQ format
    if not (eegData is None):
      for i, eeg_data in enumerate(eeg_data_dacq):
          fname = os.path.join(AxonaDataPath, axona_file_name + '.eeg')
          if i > 0:
              fname += str(i + 1)
          write_file_in_axona_format(fname, header_eeg, keyorder_eeg, eeg_data)
    # Write EGF data into DACQ format
    if not (eegData is None):
      for i, egf_data in enumerate(egf_data_dacq):
          fname = os.path.join(AxonaDataPath, axona_file_name + '.egf')
          if i > 0:
              fname += str(i + 1)
          write_file_in_axona_format(fname, header_egf, keyorder_egf, egf_data)
    # Write CLU files
    for ntet in range(n_tetrodes):
        cluFileName = os.path.join(AxonaDataPath, axona_file_name + '.clu.' + str(ntet + 1))
        if (isinstance(spike_data[ntet]['clusterIDs'], np.ndarray) 
                and len(spike_data[ntet]['clusterIDs']) > 0):
            write_clusterIDs_in_CLU_format(spike_data[ntet]['clusterIDs'], cluFileName)
    # Write SET file
    setFileName = os.path.join(AxonaDataPath, axona_file_name + '.set')
    duration_string = experiment_info['duration'] + (9 - len(experiment_info['duration'])) * ' '
    new_values_dict = {'duration': duration_string}
    write_set_file(setFileName, new_values_dict)
    # Opens recording folder with Ubuntu file browser
    if show_output:
        subprocess.Popen(['xdg-open', AxonaDataPath])
    print('Finished creating AxonaData.')


def main():
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Export data into Axona format.')
    parser.add_argument('paths', type=str, nargs='*', 
                        help='recording data folder(s) (can enter multiple paths separated by spaces to KlustaKwik simultaneously)')
    parser.add_argument('--chan', type=int, nargs = 2, 
                        help='list the first and last channel to process (counting starts from 1)')
    parser.add_argument('--eegChans', type=int, nargs = '*', 
                        help='enter channel number to use for creating EEG data')
    parser.add_argument('--subfolder', type=str, nargs = 1, 
                        help='enter the name of subfolder to use for AxonaData')
    parser.add_argument('--spike_name', type=str, nargs = 1, 
                        help='enter the name of spike set in NWB file to use for AxonaData')
    parser.add_argument('--ppm', type=int, nargs = 1, 
                        help='enter pixels_per_metre to assume position data is in pixels')
    parser.add_argument('--concatenatedDataPath', type=str, nargs = 1, 
                        help='enter the path to folder where to write data concatenated from all specified recordings')
    parser.add_argument('--show_output', action='store_true', 
                        help='to open AxonaData output folder after processing')
    parser.add_argument('--clustering_name', type=str, nargs = 1, 
                        help='specify cluster identities path to use in NWB file')
    args = parser.parse_args()
    # Get paths to recording files
    OpenEphysDataPaths = args.paths
    if isinstance(OpenEphysDataPaths, str):
        OpenEphysDataPaths = [OpenEphysDataPaths]
    if len(OpenEphysDataPaths) == 0:
        raise ValueError('Paths to folder(s) required. Use --help for more info.')
    # If directories entered as paths, attempt creating path to file by appending experiment_1.nwb
    for ndata, OpenEphysDataPath in enumerate(OpenEphysDataPaths):
        if not os.path.isfile(OpenEphysDataPath):
            new_path = os.path.join(OpenEphysDataPath, 'experiment_1.nwb')
            if os.path.isfile(new_path):
                OpenEphysDataPaths[ndata] = new_path
            else:
                raise ValueError('The following path does not lead to a NWB data file:\n' + OpenEphysDataPath)
    # Get chan input variable
    if args.chan:
        chan = [args.chan[0] - 1, args.chan[1]]
        if np.mod(chan[1] - chan[0], 4) != 0:
            raise ValueError('Channel range must cover full tetrodes')
        area_name = 'Chan' + str(args.chan[0]) + '-' + str(args.chan[1])
        channel_map = {area_name: {'list': list(range(chan[0], chan[1], 1))}}
    else:
        channel_map = None
    # Get eegChans variable
    if args.eegChans:
        eegChans = [x - 1 for x in args.eegChans]
    else:
        eegChans = None
    # Get subfolder variable
    if args.subfolder:
        subfolder = args.subfolder[0]
    else:
        subfolder = 'AxonaData'
    # Get spike_name variable
    if args.spike_name:
        spike_name = args.spike_name[0]
    else:
        spike_name = 'first_available'
    # Get ppm variable
    if args.ppm:
        pixels_per_metre = args.ppm[0]
    else:
        pixels_per_metre = None
    # Get concatenatedDataPath variable
    if args.concatenatedDataPath:
        concatenatedDataPath = args.concatenatedDataPath[0]
    else:
        concatenatedDataPath = None
    # Get show_output variable
    if args.show_output:
        show_output = True
    else:
        show_output = False
    # Get clustering_name variable
    if args.clustering_name:
        clustering_name = args.clustering_name[0]
    else:
        clustering_name = None
    # Run main script for each NWB file or to concatenate them if requested
    if concatenatedDataPath is None:
        for OpenEphysDataPath in OpenEphysDataPaths:
            createAxonaData_for_NWBfile(OpenEphysDataPath, spike_name=spike_name, 
                                        channel_map=channel_map, subfolder=subfolder, 
                                        eegChans=eegChans, pixels_per_metre=pixels_per_metre, 
                                        show_output=show_output, clustering_name=clustering_name)
    else:
        createAxonaData_for_multiple_NWBfiles(OpenEphysDataPaths, concatenatedDataPath, 
                                              spike_name=spike_name, channel_map=channel_map, 
                                              eegChans=eegChans, pixels_per_metre=pixels_per_metre, 
                                              show_output=show_output, clustering_name=clustering_name)


if __name__ == '__main__':
    main()
