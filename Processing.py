try:
    import matlab.engine
    matlab_available = True
except:
    matlab_available = False
# To install matlab engine, go to folder /usr/local/MATLAB/R2017a/extern/engines/python
# and run terminal command: sudo python setup.py installfrom TrackingDataProcessing import process_tracking_data
import argparse
import numpy as np
import os
import tempfile
import shutil
import copy
import NWBio
from createAxonaData import createAxonaData_for_NWBfile
import HelperFunctions as hfunct
from KlustaKwikWrapper import applyKlustaKwik_on_spike_data_tet
from TrackingDataProcessing import process_tracking_data
import h5py

class continuous_data_preloader(object):
    '''
    This class loads into memory and preprocesses continuous data.
    Data for specific channels can then be queried.
        channels - continuous list of channels to prepare, e.g. range(0,16,1) for first 4 tetrodes
    '''
    def __init__(self, OpenEphysDataPath, channels):
        self.chan_nrs = list(channels)
        # Load data
        print('Loading continuous data from NWB file.')
        data = NWBio.load_continuous(OpenEphysDataPath)
        self.timestamps = np.array(data['timestamps'])
        self.continuous = np.array(data['continuous'][:, channels])
        data['file_handle'].close()
        self.continuous = np.transpose(self.continuous)
        # Set bad channels to 0
        self.badChan = NWBio.listBadChannels(OpenEphysDataPath)
        self.continuous = set_bad_chan_to_0(self.continuous, self.chan_nrs, self.badChan)
        print('Loading continuous data from NWB file successful.')

    def get_chan_nrs_idx_in_continuous(self, chan_nrs_req):
        '''
        Finds rows in self.continuous corresponding to chan_nrs requested
        '''
        chan_nrs_idx = []
        for chan_nr in chan_nrs_req:
            chan_nrs_idx.append(self.chan_nrs.index(chan_nr))

        return chan_nrs_idx

    def prepare_referencing(self, referencing_method='other_channels'):
        '''
        Prepares variables for channel specific use of method continuous_data_preloader.referenced
            referencing_method options: 'other_channels' or 'all_channels'
        '''
        self.referencing_method = referencing_method
        if referencing_method == 'other_channels':
            self.continuous_sum = np.sum(self.continuous, axis=0)
        elif referencing_method == 'all_channels':
            self.continuous_mean = np.mean(self.continuous, axis=0).astype(np.int16)

    def referenced_continuous(self, chan_nrs_req):
        '''
        Returns referenced continuous data for channels in list chan_nrs_req.
        If prepare_referencing method has not been called, calls it with default options.
        '''
        if not hasattr(self, 'referencing_method'):
            self.prepare_referencing()
        chan_nrs_idx = self.get_chan_nrs_idx_in_continuous(chan_nrs_req)
        # Compute referencing array according to method
        if self.referencing_method == 'other_channels':
            continuous_other_sum = self.continuous_sum - np.sum(self.continuous[chan_nrs_idx], axis=0)
            continuous_other_mean = continuous_other_sum.astype(np.float64) / np.float64(self.continuous.shape[0])
            reference_array = np.int16(continuous_other_mean)
        elif self.referencing_method == 'all_channels':
            reference_array = self.continuous_mean
        # Extract data and apply referencing array
        data = np.zeros((len(chan_nrs_req), self.continuous.shape[1]), dtype=self.continuous.dtype)
        for idx, chan_nr in zip(chan_nrs_idx, chan_nrs_req):
            if not (chan_nr in self.badChan):
                data[chan_nrs_req.index(chan_nr), :] = self.continuous[idx, :] - reference_array

        return data

    def filter_signal(self, signal_in, filter_freqs):
        signal_out = hfunct.butter_bandpass_filter(signal_in, sampling_rate=30000.0, 
                                                   highpass_frequency=filter_freqs[0], 
                                                   lowpass_frequency=filter_freqs[1], 
                                                   filt_order=4)
        return signal_out

    def get_channels(self, chan_nrs_req, referenced=False, filter_freqs=False, no_badChan=False):
        '''
        Returns an array where each row is continuous signal for en element in list chan_nrs_req.
            referenced=True - output data is referenced with method referenced_continuous
            filter_freqs=[300, 6000] - sets band-pass filtering frequency band limits
        '''
        if referenced:
            data = self.referenced_continuous(chan_nrs_req)
        else:
            chan_nrs_idx = self.get_chan_nrs_idx_in_continuous(chan_nrs_req)
            data = self.continuous[chan_nrs_idx, :]
        if no_badChan:
            badChan_idx = []
            for i, chan_nr in enumerate(chan_nrs_req):
                if chan_nr in self.badChan:
                    badChan_idx.append(i)
            data = np.delete(data, badChan_idx, axis=0)
        if filter_freqs:
            for nchan in range(data.shape[0]):
                data[nchan, :] = self.filter_signal(data[nchan, :], filter_freqs)            

        return data

    def close(self):
        '''
        Deletes pre-loaded data from virtual memory
        '''
        if hasattr(self, 'continuous'):
            del self.continuous
        if hasattr(self, 'timestamps'):
            del self.timestamps
        if hasattr(self, 'continuous_sum'):
            del self.continuous_sum
        if hasattr(self, 'continuous_mean'):
            del self.continuous_mean

def set_bad_chan_to_0(data, chan_nrs, badChan):
    '''
    Sets channels (rows in matrix data) to 0 values according to lists chan_nrs and badChan
    If badChan is an empty list, data will be unchanged
    '''
    if len(badChan) > 0:
        for nchanBad in badChan:
            if nchanBad in chan_nrs:
                nchan = chan_nrs.index(nchanBad)
                data[nchan,:] = np.int16(0)

    return data

def detect_threshold_crossings_on_tetrode(continuous_tetrode_data, threshold, tooclose, detection_method='negative'):
    '''
    Finds all threshold crossings on a tetrode. Returns an empty array if no threshold crossings detected
        continuous_tetrode_data - 4 x N processed continuous data array for 4 channels at N datapoints
        threshold - threshold value in microvolts
        tooclose - minimum latency allowed between spikes (in datapoints)
        detection_method - 'negative', 'positive' or 'both' - the polarity of threshold crossing detection
    '''
    threshold_int16 = np.int16(np.round(threshold / 0.195))
    # Find threshold crossings for each channel
    spike_indices = np.array([], dtype=np.int64)
    for nchan in range(continuous_tetrode_data.shape[0]):
        if detection_method == 'negative':
            tmp = continuous_tetrode_data[nchan,:] < -threshold_int16
        elif detection_method == 'positive':
            tmp = continuous_tetrode_data[nchan,:] > threshold_int16
        elif detection_method == 'both':
            tmp = np.abs(continuous_tetrode_data[nchan,:]) > threshold_int16
        spike_indices = np.append(spike_indices, np.where(tmp)[0])
    # Sort the spike indices
    if len(spike_indices) > 0: 
        spike_indices = np.sort(spike_indices)
    # Remove duplicates based on temporal proximity
    if len(spike_indices) > 0:
        spike_diff = np.append(np.array([0]),np.diff(spike_indices))
        tooclose_idx = spike_diff < tooclose
        spike_indices = np.delete(spike_indices, np.where(tooclose_idx)[0])

    return spike_indices

def extract_spikes_from_tetrode(continuous_tetrode_data, spike_indices, waveform_length=[6, 34]):
    '''
    Extracts spike waveforms from continuous_data based on spike timestamps
        continuous_tetrode_data - 4 x N processed continuous data array for 4 channels at N datapoints
        spike_indices - indices for threshold crossing in the continuous_data
        waveform_length - [before, after] number of datapoints to include in the waveform
    '''
    # Using spike_indices create an array of indices (windows) to extract waveforms from LFP trace
    # The following values are chosen to match OpenEphysGUI default window
    spike_indices = np.expand_dims(spike_indices, 1)
    # Create windows for indexing all samples for a waveform
    windows = np.arange(waveform_length[0] + waveform_length[1], dtype=np.int64) - np.int64(waveform_length[0])
    windows = np.tile(windows, (spike_indices.size,1))
    windows = windows + np.tile(spike_indices, (1,windows.shape[1]))
    # Skip windows that are too close to edge of signal
    tooearly = windows < 0
    toolate = windows > (continuous_tetrode_data.shape[1] - 1)
    idx_delete = np.any(np.logical_or(tooearly, toolate), axis=1)
    windows = np.delete(windows, np.where(idx_delete)[0], axis=0)
    spike_indices = np.delete(spike_indices, np.where(idx_delete)[0], axis=0)
    # Create indexing for channels and spikes
    # windows and windows_channels shape must be  nspikes x nchans x windowsize
    windows = np.repeat(windows[:,:,np.newaxis], 4, axis=2)
    windows = np.swapaxes(windows,1,2)
    windows_channels = np.arange(4, dtype=np.int64)
    windows_channels = np.tile(windows_channels[np.newaxis,:,np.newaxis], 
                               (windows.shape[0], 1, windows.shape[2]))
    waveforms = continuous_tetrode_data[windows_channels,windows]
    
    return waveforms, spike_indices

def filter_spike_data(spike_data_tet, pos_edges, threshold, noise_cut_off, verbose=True):
    '''
    Filters data on all tetrodes according to multiple criteria
    Inputs:
        spike_data [list] - list of dictionaries with 'waveforms' and 'timestamps' for each tetrode
        pos_edges [list] - first and last position data timestamp
        threshold - threshold value in microvolts
        noise_cut_off - value for removing spikes above this cut off in microvolts
    '''
    # Create idx_keep all True for all tetrodes. Then turn values to False with each filter
    idx_keep = np.ones(spike_data_tet['waveforms'].shape[0], dtype=np.bool)
    # Include spikes occured during position data
    idx = np.logical_and(spike_data_tet['timestamps'] > pos_edges[0], 
                         spike_data_tet['timestamps'] < pos_edges[1])
    idx = idx.squeeze()
    idx_keep = np.logical_and(idx_keep, idx)
    # Include spikes above threshold
    threshold_int16 = np.int16(np.round(threshold / 0.195))
    idx = np.any(np.any(spike_data_tet['waveforms'] < -threshold_int16, axis=2), axis=1)
    idx_keep = np.logical_and(idx_keep, idx)
    if verbose and np.sum(idx) < idx.size:
        percentage_above_threshold = np.sum(idx) / float(idx.size) * 100
        print('{:.1f}% of spikes on tetrode {} were above threshold'.format(percentage_above_threshold, spike_data_tet['nr_tetrode'] + 1))
    # Include spikes below noise cut off
    if noise_cut_off and (noise_cut_off != 0):
        noise_cut_off_int16 = np.int16(np.round(noise_cut_off / 0.195))
        idx = np.all(np.all(np.abs(spike_data_tet['waveforms']) < noise_cut_off_int16, axis=2), axis=1)
        idx_keep = np.logical_and(idx_keep, idx)
        if verbose and np.sum(idx) < idx.size:
            percentage_too_big = (1 - np.sum(idx) / float(idx.size)) * 100
            print('{:.1f}% of spikes removed on tetrode {}'.format(percentage_too_big, spike_data_tet['nr_tetrode'] + 1))
    idx_keep = idx_keep.squeeze()

    return idx_keep

def clarify_OpenEphysDataPaths(OpenEphysDataPaths):
    # If directories entered as paths, attempt creating path to file by appending experiment_1.nwb
    for ndata, OpenEphysDataPath in enumerate(OpenEphysDataPaths):
        if not os.path.isfile(OpenEphysDataPath):
            new_path = os.path.join(OpenEphysDataPath, 'experiment_1.nwb')
            if os.path.isfile(new_path):
                OpenEphysDataPaths[ndata] = new_path
            else:
                raise ValueError('The following path does not lead to a NWB data file:\n' + OpenEphysDataPath)

    return OpenEphysDataPaths

def check_if_channel_maps_are_same(channel_map_1, channel_map_2):
    '''
    Determines if two channel maps are identical
    '''
    # Check that there are same number of areas in the dictionary
    if len(channel_map_1) != len(channel_map_2):
        return False
    # Sort the area names because dictionary is not ordered
    channel_map_1_keys = channel_map_1.keys()
    channel_map_1_keys.sort()
    channel_map_2_keys = channel_map_2.keys()
    channel_map_2_keys.sort()
    # Check that the areas have the same name
    for n_area in range(len(channel_map_1_keys)):
        if channel_map_1_keys[n_area] != channel_map_2_keys[n_area]:
            return False
    # Check that the channel lists are the same
    for area in channel_map_1_keys:
        if not all(channel_map_1[area]['list'] == channel_map_2[area]['list']):
            return False

    return True

def get_channel_map(OpenEphysDataPaths):
    # Get channel maps for all datasets
    channel_maps = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        if NWBio.check_if_settings_available(OpenEphysDataPaths[0],'/General/channel_map/'):
            channel_maps.append(NWBio.load_settings(OpenEphysDataPaths[0],'/General/channel_map/'))
        else:
            raise Exception('No channel map for: ' + OpenEphysDataPath)
    # Ensure that channel map is the same in all datasets
    if not all([check_if_channel_maps_are_same(channel_maps[0], x) for x in channel_maps]):
        raise Exception('Not all channel maps are the same in: ' + str(OpenEphysDataPaths))
    channel_map = channel_maps[0]
    # Check if the channel list fully coveres a set of tetrodes
    for area in channel_map.keys():
        if np.mod(len(channel_map[area]['list']), 4) != 0:
            raise ValueError('Channel map range must map to full tetrodes.')

    return channel_map

def ensure_processed_position_data_is_available(OpenEphysDataPath):
    if not NWBio.check_if_processed_position_data_available(OpenEphysDataPath):
        if NWBio.check_if_tracking_data_available(OpenEphysDataPath):
            print('Processing tracking data for: ' + OpenEphysDataPath)
            ProcessedPos = process_tracking_data(OpenEphysDataPath)
            NWBio.save_tracking_data(OpenEphysDataPath, ProcessedPos, ProcessedPos=True)
            print('ProcessedPos saved to ' + OpenEphysDataPath)
        elif NWBio.check_if_binary_pos(OpenEphysDataPath):
            NWBio.use_binary_pos(OpenEphysDataPath, postprocess=False)
            print('Using binary position data for: ' + OpenEphysDataPath)
        else:
            print('Proceeding without position data for: ' + OpenEphysDataPath)

def ensure_data_available_for_all_tetrodes(spike_data, tetrode_nrs):
    # Check which tetrodes have data missing in the recording
    tetrodes_missing_in_spike_data = copy.deepcopy(tetrode_nrs)
    for data in spike_data:
        if 'waveforms' in data.keys():
            tetrodes_missing_in_spike_data.remove(data['nr_tetrode'])
    if len(tetrodes_missing_in_spike_data) > 0:
        raise Exception('No data for tetrodes: ' + ', '.join(map(str, tetrodes_missing_in_spike_data)))

def combine_spike_datas_tet(spike_datas_tet):
    # Combine all spike_datas into a single spike_data
    orig_spike_datas_tet = copy.deepcopy(spike_datas_tet)
    spike_datas_tet_comb = orig_spike_datas_tet[0]
    for tmp_spike_data in orig_spike_datas_tet[1:]:
        spike_datas_tet_comb['waveforms'] = np.append(spike_datas_tet_comb['waveforms'], 
                                                      tmp_spike_data['waveforms'], axis=0)
        spike_datas_tet_comb['timestamps'] = np.append(spike_datas_tet_comb['timestamps'], 
                                                       tmp_spike_data['timestamps'], axis=0)
        spike_datas_tet_comb['idx_keep'] = np.append(spike_datas_tet_comb['idx_keep'], 
                                                     tmp_spike_data['idx_keep'], axis=0)

    return spike_datas_tet_comb

def uncombine_spike_datas_tet(spike_datas_tet_comb, spike_datas_tet):
    # Extract clusters for each original spike_data
    for ndata in range(len(spike_datas_tet)):
        # Get clusterIDs for this dataset and tetrode
        nspikes = sum(spike_datas_tet[ndata]['idx_keep'])
        clusterIDs = spike_datas_tet_comb['clusterIDs'][range(nspikes)]
        spike_datas_tet[ndata]['clusterIDs'] = clusterIDs
        # Remove these clusterIDs from the list
        spike_datas_tet_comb['clusterIDs'] = np.delete(spike_datas_tet_comb['clusterIDs'], range(nspikes), axis=0)

    return spike_datas_tet

def applyKlustaKwik_to_combined_recordings(spike_datas_tet):
    spike_datas_tet_comb = combine_spike_datas_tet(spike_datas_tet)
    clusterIDs = applyKlustaKwik_on_spike_data_tet(spike_datas_tet_comb)
    spike_datas_tet_comb['clusterIDs'] = clusterIDs
    spike_datas_tet = uncombine_spike_datas_tet(spike_datas_tet_comb, spike_datas_tet)

    return spike_datas_tet

def split_KiloSort_output(datas_comb_clusterIDs, datas_comb_spike_indices, datas_tet_shape):
    datas_clusterIDs = []
    datas_spike_indices = []
    data_start_pos = 0
    for data_shape in datas_tet_shape:
        nspikes = sum(datas_comb_spike_indices < data_start_pos + data_shape[1])
        datas_spike_indices.append(datas_comb_spike_indices[:nspikes] - data_start_pos)
        datas_clusterIDs.append(datas_comb_clusterIDs[:nspikes])
        datas_comb_spike_indices = np.delete(datas_comb_spike_indices, range(nspikes))
        datas_comb_clusterIDs = np.delete(datas_comb_clusterIDs, range(nspikes))
        data_start_pos += data_shape[1]

    return datas_clusterIDs, datas_spike_indices

def save_spike_data_to_disk(OpenEphysDataPath, processing_method, tetrode_nr, waveforms=None, timestamps=None, idx_keep=None, clusterIDs=None):
    spike_name = NWBio.get_spike_name_for_processing_method(processing_method)
    if not (waveforms is None) and not (timestamps is None):
        NWBio.save_spikes(OpenEphysDataPath, tetrode_nr, waveforms, 
                          timestamps, spike_name=spike_name)
    if not (idx_keep is None):
        NWBio.save_tetrode_idx_keep(OpenEphysDataPath, tetrode_nr, idx_keep, 
                                    spike_name=spike_name, overwrite=True)
    if not (clusterIDs is None):
        NWBio.save_tetrode_clusterIDs(OpenEphysDataPath, tetrode_nr, clusterIDs, 
                                      spike_name=spike_name, overwrite=True)

def process_available_spikes_using_klustakwik(OpenEphysDataPaths, channels, noise_cut_off=1000, threshold=50):
    tetrode_nrs = hfunct.get_tetrode_nrs(channels)
    # Load spikes
    spike_datas = [range(len(tetrode_nrs)) for i in range(len(OpenEphysDataPaths))]
    for n_dataset, OpenEphysDataPath in enumerate(OpenEphysDataPaths):
        print('Loading data for processing: ' + OpenEphysDataPath)
        spike_data = NWBio.load_spikes(OpenEphysDataPath, tetrode_nrs=tetrode_nrs, use_badChan=True)
        ensure_data_available_for_all_tetrodes(spike_data, tetrode_nrs)
        # Find eligible spikes on all tetrodes
        pos_edges = NWBio.get_processed_tracking_data_timestamp_edges(OpenEphysDataPath)
        for spike_data_tet in spike_data:
            spike_data_tet['idx_keep'] = filter_spike_data(spike_data_tet, pos_edges, threshold, noise_cut_off)
        # Combine into spike_datas
        for n_tet, spike_data_tet in enumerate(spike_data): 
            spike_datas[n_dataset][n_tet] = spike_data_tet
    # Cluster each tetrode using KlustaKwik. This creates 'clusterIDs' field in spike_data dictionaries.
    hfunct.print_progress(0, len(tetrode_nrs), prefix='Applying KlustaKwik:', suffix=' T: 0/' + str(len(tetrode_nrs)), initiation=True)
    for n_tet in range(len(tetrode_nrs)):
        if len(spike_datas) == 1:
            clusterIDs = applyKlustaKwik_on_spike_data_tet(spike_datas[0][n_tet])
            spike_datas[0][n_tet]['clusterIDs'] = np.int16(clusterIDs).squeeze()
        elif len(spike_datas) > 1:
            # If multiple datasets included, apply KlustaKwik on combined spike_datas_tet
            spike_datas_tet = [spike_data[n_tet] for spike_data in spike_datas]
            spike_datas_tet = applyKlustaKwik_to_combined_recordings(spike_datas_tet)
            # Put this tetrode from all datasets into spike_datas
            for n_dataset, spike_data_tet in enumerate(spike_datas_tet):
                spike_datas[n_dataset][n_tet] = spike_data_tet
        hfunct.print_progress(n_tet + 1, len(tetrode_nrs), prefix='Applying KlustaKwik:', suffix=' T: ' + str(n_tet + 1) + '/' + str(len(tetrode_nrs)))
    # Overwrite clusterIDs on disk
    for OpenEphysDataPath, spike_data in zip(OpenEphysDataPaths, spike_datas):
        print('Saving processing output to: ' + OpenEphysDataPath)
        for spike_data_tet in spike_data:
            save_spike_data_to_disk(OpenEphysDataPath, 'klustakwik', spike_data_tet['nr_tetrode'], 
                                    idx_keep=spike_data_tet['idx_keep'], 
                                    clusterIDs=spike_data_tet['clusterIDs'])

    return spike_datas

def process_spikes_from_raw_data_using_klustakwik(OpenEphysDataPaths, channels, noise_cut_off=1000, threshold=50):
    tetrode_nrs = hfunct.get_tetrode_nrs(channels)
    tooclose = 30
    spike_datas = [range(len(tetrode_nrs)) for i in range(len(OpenEphysDataPaths))]
    # Preload continuous data
    preloaded_datas = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        print('Loading data for processing: ' + OpenEphysDataPath)
        preloaded_data = continuous_data_preloader(OpenEphysDataPath, channels)
        preloaded_data.prepare_referencing('other_channels')
        preloaded_datas.append(preloaded_data)
    hfunct.print_progress(0, len(tetrode_nrs), prefix='Extract & KlustaKwik:', suffix=' T: 0/' + str(len(tetrode_nrs)), initiation=True)
    for n_tet, tetrode_nr in enumerate(tetrode_nrs):
        # Load this tetrode for all datasets
        spike_datas_tet = []
        for n_dataset in range(len(OpenEphysDataPaths)):
            data_tet = preloaded_datas[n_dataset].get_channels(hfunct.tetrode_channels(tetrode_nr), 
                                                              referenced=True, filter_freqs=[300, 6000])
            spike_indices = detect_threshold_crossings_on_tetrode(data_tet, threshold, tooclose, 
                                                                  detection_method='negative')
            waveforms, spike_indices = extract_spikes_from_tetrode(data_tet, spike_indices, 
                                                                   waveform_length=[6, 34])
            # Arrange waveforms, timestamps and tetrode number into a dictionary
            timestamps = preloaded_datas[n_dataset].timestamps[spike_indices].squeeze()
            spike_data_tet = {'waveforms': np.int16(waveforms), 
                              'timestamps': np.float64(timestamps), 
                              'nr_tetrode': tetrode_nr}
            # Create idx_keep field for this tetrode
            pos_edges = NWBio.get_processed_tracking_data_timestamp_edges(OpenEphysDataPaths[n_dataset])
            spike_data_tet['idx_keep'] = filter_spike_data(spike_data_tet, pos_edges, 
                                                           threshold, noise_cut_off, verbose=False)
            spike_datas_tet.append(spike_data_tet)
        # Apply KlustaKwik to this tetrode
        if len(spike_datas_tet) == 1:
            clusterIDs = applyKlustaKwik_on_spike_data_tet(spike_datas_tet[0])
            spike_datas_tet[0]['clusterIDs'] = np.int16(clusterIDs).squeeze()
        elif len(spike_datas_tet) > 1:
            spike_datas_tet = applyKlustaKwik_to_combined_recordings(spike_datas_tet)
        # Put this tetrode from all datasets into spike_datas
        for n_dataset, spike_data_tet in enumerate(spike_datas_tet):
                spike_datas[n_dataset][n_tet] = spike_data_tet
        hfunct.print_progress(n_tet + 1, len(tetrode_nrs), prefix='Extract & KlustaKwik:', suffix=' T: ' + str(n_tet + 1) + '/' + str(len(tetrode_nrs)))
    # Close pre-loaded datasets
    for preloaded_data in preloaded_datas:
        preloaded_data.close()
    # Save spike_datas to disk
    for OpenEphysDataPath, spike_data in zip(OpenEphysDataPaths, spike_datas):
        for data_tet in spike_data:
            save_spike_data_to_disk(OpenEphysDataPath, 'klustakwik_raw', data_tet['nr_tetrode'], 
                                    waveforms=data_tet['waveforms'], timestamps=data_tet['timestamps'], 
                                    idx_keep=data_tet['idx_keep'], clusterIDs=data_tet['clusterIDs'])

    return spike_datas

def process_raw_data_with_kilosort(OpenEphysDataPaths, channels, noise_cut_off=1000, threshold=5):
    KiloSortBinaryFileName = 'experiment_1.dat'
    tetrode_nrs = hfunct.get_tetrode_nrs(channels)
    spike_datas = [range(len(tetrode_nrs)) for i in range(len(OpenEphysDataPaths))]
    # Preload continuous data
    preloaded_datas = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        print('Loading data for processing: ' + OpenEphysDataPath)
        preloaded_data = continuous_data_preloader(OpenEphysDataPath, channels)
        preloaded_data.prepare_referencing('other_channels')
        preloaded_datas.append(preloaded_data)
    # Start matlab engine
    eng = matlab.engine.start_matlab()
    eng.cd('KiloSortScripts')
    # Worth through each tetrode
    hfunct.print_progress(0, len(tetrode_nrs), prefix='Applying KiloSort:', suffix=' T: 0/' + str(len(tetrode_nrs)), initiation=True)
    for n_tet, tetrode_nr in enumerate(tetrode_nrs):
        KiloSortProcessingFolder = tempfile.mkdtemp('KiloSortProcessing')
        # Load this tetrode for all datasets
        datas_tet_shape = []
        datas_tet = []
        for n_dataset in range(len(OpenEphysDataPaths)):
            data_tet = preloaded_datas[n_dataset].get_channels(hfunct.tetrode_channels(tetrode_nr), referenced=True, 
                                                               filter_freqs=False, no_badChan=True)
            datas_tet_shape.append(data_tet.shape)
            datas_tet.append(data_tet)
        # Save continuous data as a binary file
        datas_tet = np.concatenate(datas_tet, axis=1)
        datas_tet = np.transpose(datas_tet)
        datas_tet.tofile(os.path.join(KiloSortProcessingFolder, KiloSortBinaryFileName))
        del datas_tet
        # Run KiloSort
        eng.master_file(float(datas_tet_shape[0][0]), KiloSortProcessingFolder, nargout=0)
        eng.clear(nargout=0)
        # Load KiloSort output
        datas_comb_clusterIDs = np.load(os.path.join(KiloSortProcessingFolder, 'spike_clusters.npy'))[:,0]
        datas_comb_clusterIDs = np.int16(datas_comb_clusterIDs) + 1
        datas_comb_spike_indices = np.load(os.path.join(KiloSortProcessingFolder, 'spike_times.npy'))[:,0]
        datas_comb_spike_indices = np.int64(datas_comb_spike_indices)
        # Delete KiloSort Processing folder
        shutil.rmtree(KiloSortProcessingFolder)
        # Separate clusterIDs and spike indices to different datasets
        datas_clusterIDs, datas_spike_indices = split_KiloSort_output(datas_comb_clusterIDs, datas_comb_spike_indices, datas_tet_shape)
        # Create spike_data dictionary for this tetrode for each dataset
        for n_dataset in range(len(OpenEphysDataPaths)):
            data_tet = preloaded_datas[n_dataset].get_channels(hfunct.tetrode_channels(tetrode_nr), referenced=True, 
                                                               filter_freqs=[300, 6000], no_badChan=False)
            waveforms, spike_indices = extract_spikes_from_tetrode(data_tet, datas_spike_indices[n_dataset], 
                                                                   waveform_length=[6, 34])
            # Arrange waveforms, timestamps and tetrode number into a dictionary
            timestamps = preloaded_datas[n_dataset].timestamps[spike_indices].squeeze()
            spike_data_tet = {'waveforms': np.int16(waveforms), 
                              'timestamps': np.float64(timestamps), 
                              'clusterIDs': np.int16(datas_clusterIDs[n_dataset]), 
                              'nr_tetrode': tetrode_nr}
            # Create idx_keep field for this tetrode
            pos_edges = NWBio.get_processed_tracking_data_timestamp_edges(OpenEphysDataPaths[n_dataset])
            spike_data_tet['idx_keep'] = filter_spike_data(spike_data_tet, pos_edges, 
                                                           threshold, noise_cut_off, verbose=False)
            # Only keep clusterIDs based on idx_keep to conform to KlustaKwik processing format
            spike_data_tet['clusterIDs'] = spike_data_tet['clusterIDs'][spike_data_tet['idx_keep']]
            # Position spike_data_tet to the list of spike_data for each dataset
            spike_datas[n_dataset][n_tet] = spike_data_tet
        hfunct.print_progress(n_tet + 1, len(tetrode_nrs), prefix='Applying KiloSort:', suffix=' T: ' + str(n_tet + 1) + '/' + str(len(tetrode_nrs)))
    # Close pre-loaded datasets
    for preloaded_data in preloaded_datas:
        preloaded_data.close()
    # Save spike_datas to disk
    for OpenEphysDataPath, spike_data in zip(OpenEphysDataPaths, spike_datas):
        for data_tet in spike_data:
            save_spike_data_to_disk(OpenEphysDataPath, 'kilosort', data_tet['nr_tetrode'], 
                                    waveforms=data_tet['waveforms'], timestamps=data_tet['timestamps'], 
                                    idx_keep=data_tet['idx_keep'], clusterIDs=data_tet['clusterIDs'])

    return spike_datas

def main(OpenEphysDataPaths, processing_method='klustakwik', channel_map=None, noise_cut_off=1000, threshold=50, make_AxonaData=False, axonaDataArgs=(None, False)):
    # Ensure correct format for data paths
    if isinstance(OpenEphysDataPaths, basestring):
        OpenEphysDataPaths = [OpenEphysDataPaths]
    OpenEphysDataPaths = clarify_OpenEphysDataPaths(OpenEphysDataPaths)
    # Create ProcessedPos if not yet available
    for OpenEphysDataPath in OpenEphysDataPaths:
        ensure_processed_position_data_is_available(OpenEphysDataPath)
    # Get channel_map if not available
    if channel_map is None:
        channel_map = get_channel_map(OpenEphysDataPaths)
    # Process spikes using specified method
    area_spike_datas = []
    for area in channel_map.keys():
        channels = channel_map[area]['list']
        if processing_method == 'klustakwik':
            area_spike_datas.append(process_available_spikes_using_klustakwik(OpenEphysDataPaths, channels, 
                                                                              noise_cut_off=noise_cut_off, 
                                                                              threshold=threshold))
        elif processing_method == 'klustakwik_raw':
            area_spike_datas.append(process_spikes_from_raw_data_using_klustakwik(OpenEphysDataPaths, channels, 
                                                                                  noise_cut_off=noise_cut_off, 
                                                                                  threshold=threshold))
        elif processing_method == 'kilosort':
            if not matlab_available:
                raise Exception('Matlab not available. Can not process using KiloSort.')
            area_spike_datas.append(process_raw_data_with_kilosort(OpenEphysDataPaths, channels, 
                                                                   noise_cut_off=noise_cut_off, threshold=5))
    # Save data in Axona Format
    if make_AxonaData:
        spike_name = NWBio.get_spike_name_for_processing_method(processing_method)
        for OpenEphysDataPath in OpenEphysDataPaths:
            createAxonaData_for_NWBfile(OpenEphysDataPath, spike_name=spike_name, 
                                        channel_map=channel_map, pixels_per_metre=axonaDataArgs[0], 
                                        show_output=axonaDataArgs[1])

def process_data_tree(root_path, downsample=False):
    # Commence directory walk
    for dirName, subdirList, fileList in os.walk(root_path):
        for fname in fileList:
            if not ('Experiment' in dirName):
                fpath = os.path.join(dirName, fname)
                if fname == 'experiment_1.nwb':
                    AxonaDataExists = any(['AxonaData' in subdir for subdir in subdirList])
                    recordingKey = NWBio.get_recordingKey(fpath)
                    processorKey = NWBio.get_processorKey(fpath)
                    raw_data_path = '/acquisition/timeseries/' + recordingKey + \
                                    '/continuous/' + processorKey + '/data'
                    downsampled_data_path = '/acquisition/timeseries/' + recordingKey + \
                                            '/continuous/' + processorKey + '/tetrode_lowpass'
                    spike_data_path = '/acquisition/timeseries/' + recordingKey + '/spikes/'
                    with h5py.File(fpath, 'r') as h5file:
                        raw_data_available = raw_data_path in h5file
                        downsampled_data_available = downsampled_data_path in h5file
                        spikes_recorded = len(h5file[spike_data_path].items()) > 0
                    if not AxonaDataExists and spikes_recorded and \
                       (raw_data_available or downsampled_data_available):
                        main(fpath, processing_method='klustakwik', 
                            noise_cut_off=1000, threshold=50, make_AxonaData=True, 
                            axonaDataArgs=(None, False))
    if downsample:
        import DeleteRAWdata
        DeleteRAWdata.main(root_path)

if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Apply KlustaKwik and export into Axona format.')
    parser.add_argument('paths', type=str, nargs='*', 
                        help='recording data folder(s) (can enter multiple paths separated by spaces to KlustaKwik simultaneously)')
    parser.add_argument('--chan', type=int, nargs = 2, 
                        help='list the first and last channel to process (counting starts from 1)')
    parser.add_argument('--noisecut', type=int, nargs = 1, 
                        help='enter 0 to skip or value in microvolts for noise cutoff (default is 1000)')
    parser.add_argument('--threshold', type=int, nargs = 1, 
                        help='enter spike threshold in microvolts (default is 50)')
    parser.add_argument('--klustakwik', action='store_true',
                        help='use KlustaKwik to cluster spike data saved by OpenEphys GUI (default)')
    parser.add_argument('--klustakwik_raw', action='store_true',
                        help='use KlustaKwik to cluster spike data obtained from raw data')
    parser.add_argument('--kilosort', action='store_true',
                        help='use KiloSort to cluster spike data (this uses raw data only)')
    parser.add_argument('--noAxonaData', action='store_true',
                        help='to skip conversion into AxonaData format after processing')
    parser.add_argument('--ppm', type=int, nargs = 1, 
                        help='(for AxonaData) enter pixels_per_metre to assume position data is in pixels')
    parser.add_argument('--show_output', action='store_true', 
                        help='(for AxonaData) to open AxonaData output folder after processing')
    parser.add_argument('--datatree', action='store_true', 
                        help='to process a whole data tree with default arguments in method process_data_tree')
    parser.add_argument('--downsample', action='store_true', 
                        help='to downsample a whole data tree after processing')
    args = parser.parse_args()
    # Get paths to recording files
    OpenEphysDataPaths = args.paths
    # If datatree processing requested, use process_data_tree method
    if args.datatree:
        if args.downsample:
            downsample = True
        else:
            downsample = False
        process_data_tree(OpenEphysDataPaths[0], downsample)
    else:
        # Get chan input variable
        if args.chan:
            chan = [args.chan[0] - 1, args.chan[1]]
            if np.mod(chan[1] - chan[0], 4) != 0:
                raise ValueError('Channel range must cover full tetrodes')
            area_name = 'Chan' + str(args.chan[0]) + '-' + str(args.chan[1])
            channel_map = {area_name: {'list': range(chan[0], chan[1], 1)}}
        else:
            channel_map = None
        # Rewrite default noisecut if specified
        if args.noisecut:
            noise_cut_off = args.noisecut[0]
        else:
            noise_cut_off = 1000
        # Rewrite default threshold if specified
        if args.threshold:
            threshold = args.threshold[0]
        else:
            threshold = 50
        # Specify processing_method
        if args.klustakwik:
            processing_method = 'klustakwik'
        elif args.klustakwik_raw:
            processing_method = 'klustakwik_raw'
        elif args.kilosort:
            processing_method = 'kilosort'
        else:
            processing_method = 'klustakwik'
        # Specify axona data options
        if args.noAxonaData:
            make_AxonaData = False
        else:
            make_AxonaData = True
        if args.ppm:
            pixels_per_metre = args.ppm[0]
        else:
            pixels_per_metre = None
        if args.show_output:
            show_output = True
        else:
            show_output = False
        axonaDataArgs = (pixels_per_metre, show_output)
        # Run the script
        main(OpenEphysDataPaths, processing_method, channel_map, noise_cut_off, threshold, make_AxonaData, axonaDataArgs)
