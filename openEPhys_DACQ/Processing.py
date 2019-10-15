try:
    import matlab.engine
    matlab_available = True
except:
    matlab_available = False
# To install matlab engine, go to folder /usr/local/MATLAB/R2017a/extern/engines/python
# and run terminal command: sudo python setup.py install
import argparse
from time import sleep, time
import os
import tempfile
import shutil
import copy
from multiprocessing import Process

import numpy as np

from openEPhys_DACQ.package_configuration import package_config, package_path
from openEPhys_DACQ import NWBio
from openEPhys_DACQ.createAxonaData import createAxonaData_for_NWBfile
from openEPhys_DACQ import HelperFunctions as hfunct
from openEPhys_DACQ.KlustaKwikWrapper import applyKlustaKwik_on_spike_data_tet
from openEPhys_DACQ.TrackingDataProcessing import (remove_tracking_data_jumps,
                                                   iteratively_combine_multicamera_data_for_recording)

def use_binary_pos(filename, postprocess=False, maxjump=25):
    """
    Copies binary position data into tracking data
    Apply postprocessing with postprocess=True
    """
    recordingKey = NWBio.get_recordingKey(filename)
    # Load timestamps and position data
    with h5py.File(filename, 'r+') as h5file:
        timestamps = np.array(h5file['acquisition']['timeseries'][recordingKey]['events']['binary1']['timestamps'])
        xy = np.array(h5file['acquisition']['timeseries'][recordingKey]['events']['binary1']['data'][:,:2])
    data = {'xy': xy, 'timestamps': timestamps}
    # Construct data into repository familiar posdata format (single array with times in first column)
    posdata = np.append(data['timestamps'][:,None], data['xy'].astype(np.float64), axis=1)
    # Add NaNs for second LED if missing
    if posdata.shape[1] < 5:
        nanarray = np.zeros(data['xy'].shape, dtype=np.float64)
        nanarray[:] = np.nan
        posdata = np.append(posdata, nanarray, axis=1)
    # Postprocess the data if requested
    if postprocess:
        posdata = remove_tracking_data_jumps(posdata, maxjump)
    # Save data to ProcessedPos position in NWB file
    NWBio.save_tracking_data(filename, posdata, ProcessedPos=True, overwrite=True)


def process_tracking_data(filename, save_to_file=False, verbose=False):
    if verbose:
        print('Processing tracking data for {}'.format(filename))

    # Get CameraSettings
    try:
        CameraSettings = NWBio.load_settings(filename, '/CameraSettings/')
    except KeyError as e:
        raise Exception('Could not load CameraSettings for {}'.format(filename)
                        + '\nLikely older version on file format.')
    cameraIDs = sorted(CameraSettings['CameraSpecific'].keys())
    # Get global clock timestamps
    OE_GC_times = NWBio.load_GlobalClock_timestamps(filename)
    # Get arena_size
    arena_size = NWBio.load_settings(filename, '/General/arena_size/')
    # Load position data for all cameras
    posdatas = []
    for cameraID in cameraIDs:
        posdatas.append(NWBio.load_raw_tracking_data(filename, cameraID))
    ProcessedPos = iteratively_combine_multicamera_data_for_recording(
        CameraSettings, arena_size, posdatas, OE_GC_times, verbose=verbose)
    if save_to_file:
        # Save corrected data to file
        NWBio.save_tracking_data(filename, ProcessedPos, ProcessedPos=True, overwrite=True)

    return ProcessedPos


def recompute_tracking_data_for_all_files_in_directory_tree(root_path, verbose=False):

    # Commence directory walk
    processeses = []
    for dir_name, subdirList, fileList in os.walk(root_path):
        for fname in fileList:
            fpath = os.path.join(dir_name, fname)
            if fname == 'experiment_1.nwb':
                p = Process(target=process_tracking_data, args=(fpath,),
                            kwargs={'save_to_file': True, 'verbose': verbose})
                p.start()
                processeses.append(p)
    for p in processeses:
        p.join()


def lowpass_and_downsample_channel(
        fpath, chan, original_sampling_rate, target_sampling_rate):
    data = NWBio.load_continuous_as_array(fpath, chan)['continuous'].squeeze()
    data = hfunct.lowpass_and_downsample(data, original_sampling_rate, target_sampling_rate)

    return data


def lowpass_and_downsample_channels(
        fpath, channels, original_sampling_rate, target_sampling_rate):
    # Load, lowpass filter and downsample each channel in a separate process
    multiprocessor = hfunct.multiprocess()
    for chan in channels:
        hfunct.proceed_when_enough_memory_available(percent=0.50)
        print(hfunct.time_string() + ' Starting lowpass_and_downsample_channel for chan ' + str(chan))
        multiprocessor.run(lowpass_and_downsample_channel,
                           args=(fpath, chan, original_sampling_rate, target_sampling_rate))
        sleep(4)
    # Collect processed data from multiprocess class
    out = multiprocessor.results()

    return out


def lowpass_and_downsample_channel_on_each_tetrode(
        fpath, original_sampling_rate, target_sampling_rate, n_tetrodes, badChans):
    processed_chans = []
    processed_tets = []
    for n_tet in range(n_tetrodes):
        chans = hfunct.tetrode_channels(n_tet)
        chan = []
        for c in chans:
            if c not in badChans:
                chan.append(c)
        if len(chan) > 0:
            processed_chans.append(chan[0])
            processed_tets.append(n_tet)
    processed_data_list = lowpass_and_downsample_channels(
        fpath, processed_chans, original_sampling_rate, target_sampling_rate)
    new_data_size = processed_data_list[0].size
    processed_data_array = np.zeros((new_data_size, n_tetrodes), dtype=np.int16)
    for n_tet, processed_data in zip(processed_tets, processed_data_list):
        processed_data_array[:, n_tet] = np.int16(processed_data)

    return processed_data_array, processed_chans


def lowpass_and_downsample_AUX_data(fpath, n_tetrodes, original_sampling_rate, target_sampling_rate):
    aux_chan_list = NWBio.list_AUX_channels(fpath, n_tetrodes)
    processed_data_list = lowpass_and_downsample_channels(
        fpath, aux_chan_list, original_sampling_rate, target_sampling_rate)
    downsampled_AUX = np.concatenate([x[:, None] for x in processed_data_list], axis=1)
    downsampled_AUX = downsampled_AUX.astype(np.int16)

    return downsampled_AUX


def downsample_raw_timestamps(fpath, downsample_factor):
    return NWBio.load_raw_data_timestamps_as_array(fpath)[::downsample_factor]


def create_downsampled_data(fpath, n_tetrodes=32, downsample_factor=10):
    # Get original sampling rate and compute target rate based on downsampling factor
    original_sampling_rate = NWBio.OpenEphys_SamplingRate()
    target_sampling_rate = int(NWBio.OpenEphys_SamplingRate() / downsample_factor)
    # Get list of bad channels
    badChans = NWBio.listBadChannels(fpath)
    # Get downsampled data
    downsampled_data, used_chans = lowpass_and_downsample_channel_on_each_tetrode(
        fpath, original_sampling_rate, target_sampling_rate, n_tetrodes, badChans)
    downsampled_AUX = lowpass_and_downsample_AUX_data(fpath, n_tetrodes, original_sampling_rate, target_sampling_rate)
    downsampled_timestamps = downsample_raw_timestamps(fpath, downsample_factor)
    # Ensure timestamps and downsampled data have same number of samples
    assert downsampled_timestamps.size == downsampled_data.shape[0], \
        'Downsampled timestamps does not match number of samples in downsampled continuous data.'
    # Create downsampling_info
    downsampling_info = {'original_sampling_rate': original_sampling_rate,
                         'downsampled_sampling_rate': target_sampling_rate,
                         'downsampled_channels': np.array(used_chans)}
    # Save downsampled data to disk
    NWBio.save_downsampled_data_to_disk(
        fpath, downsampled_data, downsampled_timestamps, downsampled_AUX, downsampling_info)


def delete_raw_data(fpath, only_if_downsampled_data_available=True):
    print(hfunct.time_string() + ' Deleting raw data in ' + fpath)
    NWBio.delete_raw_data(fpath, only_if_downsampled_data_available=only_if_downsampled_data_available)
    print(hfunct.time_string() + ' Repacking NWB file ' + fpath)
    NWBio.repack_NWB_file(fpath, replace_original=True)


class ContinuousDataPreloader(object):
    """
    This class loads into memory and preprocesses continuous data.
    Data for specific channels can then be queried.
        channels - continuous list of channels to prepare, e.g. list(range(0,16,1)) for first 4 tetrodes
    """
    def __init__(self, OpenEphysDataPath, channels):
        self.chan_nrs = list(channels)
        # Load data
        print('Loading continuous data from NWB file.')
        data = NWBio.load_continuous_as_array(OpenEphysDataPath, self.chan_nrs)
        self.timestamps = data['timestamps']
        self.continuous = data['continuous']
        self.continuous = np.transpose(self.continuous)
        # Set bad channels to 0
        self.badChan = NWBio.listBadChannels(OpenEphysDataPath)
        self.continuous = set_bad_chan_to_0(self.continuous, self.chan_nrs, self.badChan)
        print('Loading continuous data from NWB file successful.')

    def get_chan_nrs_idx_in_continuous(self, chan_nrs_req):
        """
        Finds rows in self.continuous corresponding to chan_nrs requested
        """
        chan_nrs_idx = []
        for chan_nr in chan_nrs_req:
            chan_nrs_idx.append(self.chan_nrs.index(chan_nr))

        return chan_nrs_idx

    def prepare_referencing(self, referencing_method='other_channels'):
        """
        Prepares variables for channel specific use of method ContinuousDataPreloader.referenced
            referencing_method options: 'other_channels' or 'all_channels'
        """
        self.referencing_method = referencing_method
        if referencing_method == 'other_channels':
            self.continuous_sum = np.sum(self.continuous, axis=0)
        elif referencing_method == 'all_channels':
            self.continuous_mean = np.mean(self.continuous, axis=0).astype(np.int16)

    def referenced_continuous(self, chan_nrs_req):
        """
        Returns referenced continuous data for channels in list chan_nrs_req.
        If prepare_referencing method has not been called, calls it with default options.
        """
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
        """
        Returns an array where each row is continuous signal for en element in list chan_nrs_req.
            referenced=True - output data is referenced with method referenced_continuous
            filter_freqs=[300, 6000] - sets band-pass filtering frequency band limits
        """
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
        """
        Deletes pre-loaded data from virtual memory
        """
        if hasattr(self, 'continuous'):
            del self.continuous
        if hasattr(self, 'timestamps'):
            del self.timestamps
        if hasattr(self, 'continuous_sum'):
            del self.continuous_sum
        if hasattr(self, 'continuous_mean'):
            del self.continuous_mean

def set_bad_chan_to_0(data, chan_nrs, badChan):
    """
    Sets channels (rows in matrix data) to 0 values according to lists chan_nrs and badChan
    If badChan is an empty list, data will be unchanged
    """
    if len(badChan) > 0:
        for nchanBad in badChan:
            if nchanBad in chan_nrs:
                nchan = chan_nrs.index(nchanBad)
                data[nchan,:] = np.int16(0)

    return data

def detect_threshold_crossings_on_tetrode(continuous_tetrode_data, threshold, tooclose, detection_method='negative'):
    """
    Finds all threshold crossings on a tetrode. Returns an empty array if no threshold crossings detected
        continuous_tetrode_data - 4 x N processed continuous data array for 4 channels at N datapoints
        threshold - threshold value in microvolts
        tooclose - minimum latency allowed between spikes (in datapoints)
        detection_method - 'negative', 'positive' or 'both' - the polarity of threshold crossing detection
    """
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
    """
    Extracts spike waveforms from continuous_data based on spike timestamps
        continuous_tetrode_data - 4 x N processed continuous data array for 4 channels at N datapoints
        spike_indices - indices for threshold crossing in the continuous_data
        waveform_length - [before, after] number of datapoints to include in the waveform
    """
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
    # Prepare idx_keep array to return
    idx_keep = np.logical_not(idx_delete).squeeze()
    # Create indexing for channels and spikes
    # windows and windows_channels shape must be  nspikes x nchans x windowsize
    windows = np.repeat(windows[:,:,np.newaxis], 4, axis=2)
    windows = np.swapaxes(windows,1,2)
    windows_channels = np.arange(4, dtype=np.int64)
    windows_channels = np.tile(windows_channels[np.newaxis,:,np.newaxis], 
                               (windows.shape[0], 1, windows.shape[2]))
    waveforms = continuous_tetrode_data[windows_channels,windows]
    
    return waveforms, spike_indices, idx_keep

def filter_spike_data(spike_data_tet, pos_edges, threshold, noise_cut_off, verbose=True):
    """
    Filters data on all tetrodes according to multiple criteria
    Inputs:
        spike_data [list] - list of dictionaries with 'waveforms' and 'timestamps' for each tetrode
        pos_edges [list] - first and last position data timestamp
        threshold - threshold value in microvolts
        noise_cut_off - value for removing spikes above this cut off in microvolts
    """
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
    idx_keep = idx_keep.reshape(idx_keep.size)

    return idx_keep

def clarify_OpenEphysDataPaths(OpenEphysDataPaths):
    # If directories entered as paths, attempt creating path to file by appending experiment_1.nwb
    for ndata, OpenEphysDataPath in enumerate(OpenEphysDataPaths):
        if not os.path.isfile(OpenEphysDataPath):
            new_path = os.path.join(OpenEphysDataPath, 'experiment_1.nwb')
            if os.path.isfile(new_path):
                OpenEphysDataPaths[ndata] = new_path
            else:
                raise ValueError('The following path does not lead to a data file:\n' + OpenEphysDataPath)

    return OpenEphysDataPaths


def get_channel_map(OpenEphysDataPaths):
    # Get channel maps for all datasets
    channel_maps = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        if NWBio.check_if_settings_available(OpenEphysDataPaths[0],'/General/channel_map/'):
            channel_maps.append(NWBio.load_settings(OpenEphysDataPaths[0],'/General/channel_map/'))
        else:
            raise Exception('No channel map for: ' + OpenEphysDataPath)
    # Ensure that channel map is the same in all datasets
    if not all([NWBio.check_if_channel_maps_are_same(channel_maps[0], x) for x in channel_maps]):
        raise Exception('Not all channel maps are the same in: ' + str(OpenEphysDataPaths))
    channel_map = channel_maps[0]
    # Check if the channel list fully coveres a set of tetrodes
    for area in channel_map.keys():
        if np.mod(len(channel_map[area]['list']), 4) != 0:
            raise ValueError('Channel map range must map to full tetrodes.')

    return channel_map


def process_position_data(OpenEphysDataPath, postprocess=False, maxjump=25):
    if NWBio.check_if_tracking_data_available(OpenEphysDataPath):
        print('Processing tracking data for: ' + OpenEphysDataPath)
        ProcessedPos = process_tracking_data(OpenEphysDataPath, verbose=True)
        NWBio.save_tracking_data(OpenEphysDataPath, ProcessedPos, ProcessedPos=True, overwrite=True)
        print('ProcessedPos saved to ' + OpenEphysDataPath)
    elif NWBio.check_if_binary_pos(OpenEphysDataPath):
        use_binary_pos(OpenEphysDataPath, postprocess=postprocess, maxjump=maxjump)
        print('Using binary position data for: ' + OpenEphysDataPath)
    else:
        print('Proceeding without position data for: ' + OpenEphysDataPath)


def ensure_processed_position_data_is_available(OpenEphysDataPath, **kwargs):
    if not NWBio.check_if_processed_position_data_available(OpenEphysDataPath):
        process_position_data(OpenEphysDataPath, **kwargs)


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

def uncombine_spike_datas_tet_clusterIDs(clusterIDs, spike_datas_tet):
    clusterIDs = copy.copy(clusterIDs)
    # Extract clusters for each original spike_data
    for ndata in range(len(spike_datas_tet)):
        # Get clusterIDs for this dataset
        nspikes = sum(spike_datas_tet[ndata]['idx_keep'])
        spike_datas_tet[ndata]['clusterIDs'] = clusterIDs[list(range(nspikes))]
        # Remove these clusterIDs from the list
        clusterIDs = np.delete(clusterIDs, list(range(nspikes)), axis=0)

    return spike_datas_tet

def applyKlustaKwik_to_combined_recordings(spike_datas_tet, max_clusters=31):
    spike_datas_tet_comb = combine_spike_datas_tet(spike_datas_tet)
    clusterIDs = applyKlustaKwik_on_spike_data_tet(spike_datas_tet_comb, 
                                                   max_possible_clusters=max_clusters)
    spike_datas_tet = uncombine_spike_datas_tet_clusterIDs(clusterIDs, spike_datas_tet)

    return spike_datas_tet

def split_KiloSort_output(datas_comb_clusterIDs, datas_comb_spike_indices, datas_tet_shape):
    datas_clusterIDs = []
    datas_spike_indices = []
    data_start_pos = 0
    for data_shape in datas_tet_shape:
        nspikes = sum(datas_comb_spike_indices < data_start_pos + data_shape[1])
        datas_spike_indices.append(datas_comb_spike_indices[:nspikes] - data_start_pos)
        datas_clusterIDs.append(datas_comb_clusterIDs[:nspikes])
        datas_comb_spike_indices = np.delete(datas_comb_spike_indices, list(range(nspikes)))
        datas_comb_clusterIDs = np.delete(datas_comb_clusterIDs, list(range(nspikes)))
        data_start_pos += data_shape[1]

    return datas_clusterIDs, datas_spike_indices

def save_spike_data_to_disk(OpenEphysDataPath, processing_method, tetrode_nr, waveforms=None, timestamps=None, idx_keep=None, clusterIDs=None):
    spike_name = NWBio.get_spike_name_for_processing_method(processing_method)
    if not (waveforms is None) and not (timestamps is None):
        NWBio.save_spikes(OpenEphysDataPath, tetrode_nr, waveforms, 
                          timestamps, spike_name=spike_name, overwrite=True)
    if not (idx_keep is None):
        NWBio.save_tetrode_idx_keep(OpenEphysDataPath, tetrode_nr, idx_keep, 
                                    spike_name=spike_name, overwrite=True)
    if not (clusterIDs is None):
        NWBio.save_tetrode_clusterIDs(OpenEphysDataPath, tetrode_nr, clusterIDs, 
                                      spike_name=spike_name, overwrite=True)


class Multiprocess_KlustaKwik(object):

    def __init__(self):
        self.multiprocessor = hfunct.multiprocess()

    def add(self, spike_data_tet, max_clusters=31):
        self.multiprocessor.run(applyKlustaKwik_on_spike_data_tet, 
                                args=(spike_data_tet,), 
                                kwargs={'max_possible_clusters': max_clusters}, 
                                single_cpu_affinity=False)

    def get(self):
        return self.multiprocessor.results()


def process_combined_recordings_spike_datas(spike_datas, tetrode_nrs, max_clusters):
    mp_KlustaKwik = Multiprocess_KlustaKwik()
    for n_tet in range(len(tetrode_nrs)):
        hfunct.proceed_when_enough_memory_available(percent=0.60)
        spike_datas_tet = [spike_data[n_tet] for spike_data in spike_datas]
        spike_datas_tet_comb = combine_spike_datas_tet(spike_datas_tet)
        mp_KlustaKwik.add(spike_datas_tet_comb, max_clusters=max_clusters)
        hfunct.print_progress(n_tet + 1, len(tetrode_nrs), prefix='Applying KlustaKwik:', suffix=' T: ' + str(n_tet + 1) + '/' + str(len(tetrode_nrs)))
    for n_tet in range(len(tetrode_nrs)):
        # Get sorted combined data for one tetrode
        clusterIDs = mp_KlustaKwik.get()[n_tet]
        spike_datas_tet = [spike_data[n_tet] for spike_data in spike_datas]
        spike_datas_tet = uncombine_spike_datas_tet_clusterIDs(clusterIDs, spike_datas_tet)
        for n_dataset, spike_data_tet in enumerate(spike_datas_tet):
            spike_datas[n_dataset][n_tet] = spike_data_tet

    return spike_datas


def process_available_spikes_using_klustakwik(OpenEphysDataPaths, channels, 
                                              noise_cut_off=1000, threshold=50, 
                                              max_clusters=31):
    tetrode_nrs = hfunct.get_tetrode_nrs(channels)
    # Load spikes
    spike_datas = [list(range(len(tetrode_nrs))) for i in range(len(OpenEphysDataPaths))]
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
    if len(spike_datas) == 1:
        hfunct.print_progress(0, len(tetrode_nrs), prefix='Applying KlustaKwik:', suffix=' T: 0/' + str(len(tetrode_nrs)), initiation=True)
        mp_KlustaKwik = Multiprocess_KlustaKwik()
        for n_tet in range(len(tetrode_nrs)):
            mp_KlustaKwik.add(spike_datas[0][n_tet], max_clusters=max_clusters)
            hfunct.print_progress(n_tet + 1, len(tetrode_nrs), prefix='Applying KlustaKwik:', suffix=' T: ' + str(n_tet + 1) + '/' + str(len(tetrode_nrs)))
        for n_tet in range(len(tetrode_nrs)):
            spike_datas[0][n_tet]['clusterIDs'] = mp_KlustaKwik.get()[n_tet]
    elif len(spike_datas) > 1:
        spike_datas = process_combined_recordings_spike_datas(spike_datas, tetrode_nrs, max_clusters)
    # Overwrite clusterIDs on disk
    for OpenEphysDataPath, spike_data in zip(OpenEphysDataPaths, spike_datas):
        print('Saving processing output to: ' + OpenEphysDataPath)
        for spike_data_tet in spike_data:
            save_spike_data_to_disk(OpenEphysDataPath, 'klustakwik', spike_data_tet['nr_tetrode'], 
                                    idx_keep=spike_data_tet['idx_keep'], 
                                    clusterIDs=spike_data_tet['clusterIDs'])

    return spike_datas

def process_spikes_from_raw_data_using_klustakwik(OpenEphysDataPaths, channels, 
                                                  noise_cut_off=1000, threshold=50, 
                                                  max_clusters=31):
    tetrode_nrs = hfunct.get_tetrode_nrs(channels)
    tooclose = 30
    spike_datas = [list(range(len(tetrode_nrs))) for i in range(len(OpenEphysDataPaths))]
    # Preload continuous data
    preloaded_datas = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        print('Loading data for processing: ' + OpenEphysDataPath)
        preloaded_data = ContinuousDataPreloader(OpenEphysDataPath, channels)
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
            waveforms, spike_indices, _ = extract_spikes_from_tetrode(data_tet, spike_indices, 
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
            clusterIDs = applyKlustaKwik_on_spike_data_tet(spike_datas_tet[0], 
                                                           max_possible_clusters=max_clusters)
            spike_datas_tet[0]['clusterIDs'] = clusterIDs
        elif len(spike_datas_tet) > 1:
            spike_datas_tet = applyKlustaKwik_to_combined_recordings(spike_datas_tet, 
                                                                     max_clusters=max_clusters)
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

def process_raw_data_with_kilosort(OpenEphysDataPaths, channels, noise_cut_off=1000, threshold=5, 
                                   num_clusters=31):
    KiloSortBinaryFileName = 'experiment_1.dat'
    tetrode_nrs = hfunct.get_tetrode_nrs(channels)
    spike_datas = [list(range(len(tetrode_nrs))) for i in range(len(OpenEphysDataPaths))]
    # Preload continuous data
    preloaded_datas = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        print('Loading data for processing: ' + OpenEphysDataPath)
        preloaded_data = ContinuousDataPreloader(OpenEphysDataPath, channels)
        preloaded_data.prepare_referencing('other_channels')
        preloaded_datas.append(preloaded_data)
    # Start matlab engine
    eng = matlab.engine.start_matlab()
    eng.cd(os.path.join(package_path, 'Utils', 'KiloSortScripts'))
    eng.add_kilosort_paths(package_config['kilosort_path'], package_config['npy_matlab_path'])
    # Work through each tetrode
    for n_tet, tetrode_nr in enumerate(tetrode_nrs):
        print('Applying KiloSort to tetrode ' + str(n_tet + 1) + '/' + str(len(tetrode_nrs)))
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
        eng.master_file(float(datas_tet_shape[0][0]), KiloSortProcessingFolder, 
                        float(num_clusters), nargout=0)
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
            waveforms, spike_indices, idx_keep = extract_spikes_from_tetrode(data_tet, datas_spike_indices[n_dataset], 
                                                                             waveform_length=[6, 34])
            # Only keep clusterIDs for clusters that were included by extract_spikes_from_tetrode
            datas_clusterIDs[n_dataset] = (datas_clusterIDs[n_dataset][idx_keep]).squeeze()
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


def processing(OpenEphysDataPaths, processing_method='klustakwik', channel_map=None, 
               noise_cut_off=1000, threshold=50, make_AxonaData=False, 
               axonaDataArgs=(None, False), max_clusters=31, 
               force_position_processing=False, pos_data_processing_kwargs={}):
    # Ensure correct format for data paths
    if isinstance(OpenEphysDataPaths, str):
        OpenEphysDataPaths = [OpenEphysDataPaths]
    OpenEphysDataPaths = clarify_OpenEphysDataPaths(OpenEphysDataPaths)
    for fpath in OpenEphysDataPaths:
        if not NWBio.check_if_open_ephys_nwb_file(fpath):
            raise ValueError('Specified path {} does not lead to expected filetype.'.format(fpath))
    # Create ProcessedPos if not yet available
    for OpenEphysDataPath in OpenEphysDataPaths:
        if force_position_processing:
            process_position_data(OpenEphysDataPath, **pos_data_processing_kwargs)
        else:
            ensure_processed_position_data_is_available(OpenEphysDataPath, **pos_data_processing_kwargs)
    # Get channel_map if not available
    if channel_map is None:
        channel_map = get_channel_map(OpenEphysDataPaths)
    # Process spikes using specified method
    area_spike_datas = []
    print(hfunct.time_string(), 'DEBUG: Starting Processing', processing_method)
    DEBUG_Time = time()
    for area in channel_map.keys():
        channels = channel_map[area]['list']
        if processing_method == 'klustakwik':
            area_spike_datas.append(process_available_spikes_using_klustakwik(OpenEphysDataPaths, channels, 
                                                                              noise_cut_off=noise_cut_off, 
                                                                              threshold=threshold, 
                                                                              max_clusters=max_clusters))
        elif processing_method == 'klustakwik_raw':
            area_spike_datas.append(process_spikes_from_raw_data_using_klustakwik(OpenEphysDataPaths, channels, 
                                                                                  noise_cut_off=noise_cut_off, 
                                                                                  threshold=threshold, 
                                                                                  max_clusters=max_clusters))
        elif processing_method == 'kilosort':
            if not matlab_available:
                raise Exception('Matlab not available. Can not process using KiloSort.')
            area_spike_datas.append(process_raw_data_with_kilosort(OpenEphysDataPaths, channels, 
                                                                   noise_cut_off=noise_cut_off, threshold=5, 
                                                                   num_clusters=max_clusters))
    print(hfunct.time_string(), 'DEBUG: Finished Processing in ', time() - DEBUG_Time)
    # Save data in Axona Format
    del area_spike_datas
    if make_AxonaData:
        spike_name = NWBio.get_spike_name_for_processing_method(processing_method)
        for OpenEphysDataPath in OpenEphysDataPaths:
            createAxonaData_for_NWBfile(OpenEphysDataPath, spike_name=spike_name, 
                                        channel_map=channel_map, pixels_per_metre=axonaDataArgs[0], 
                                        show_output=axonaDataArgs[1])


def process_data_tree(root_path, only_keep_processor=None, downsample=False, delete_raw=False, max_clusters=31):
    # Create list of dirnames skipped
    dir_names_skipped = []
    # Commence directory walk
    for dir_name, subdirList, fileList in os.walk(root_path):
        for fname in fileList:
            if 'Experiment' in dir_name:
                if not (dir_name in dir_names_skipped):
                    dir_names_skipped.append(dir_name)
                    raise Warning('Experiment found in directory name, skipping: ' + dir_name)
            else:
                fpath = os.path.join(dir_name, fname)
                if fname == 'experiment_1.nwb':
                    AxonaDataExists = any(['AxonaData' in subdir for subdir in subdirList])
                    if not AxonaDataExists:
                        print(hfunct.time_string() + ' Applying KlustaKwik on ' + fpath)
                        processing(fpath, processing_method='klustakwik',
                                   noise_cut_off=1000, threshold=50, make_AxonaData=True,
                                   axonaDataArgs=(None, False), max_clusters=max_clusters)
                    if not (only_keep_processor is None):
                        # Get all processor paths in this file
                        processor_paths = NWBio.get_all_processor_paths(fpath)
                        # Check that the processor key requested to keep is available
                        if not any([path.endswith(only_keep_processor) for path in processor_paths]):
                            raise ValueError('{} not found in {}. Can not delete others.'.format(only_keep_processor,
                                                                                                 fpath))
                        # Delete all other processor paths in the file, if any available
                        if len(processor_paths) > 1:
                            for path in processor_paths:
                                if not path.endswith(only_keep_processor):
                                    print('Deleting path: {}\n    in file: {}'.format(path, fpath))
                                    NWBio.delete_path_in_file(fpath, path)
                    if downsample:
                        if not NWBio.check_if_downsampled_data_available(fpath):
                            if NWBio.check_if_raw_data_available(fpath):
                                print(hfunct.time_string() + ' Downsample ' + fpath)
                                create_downsampled_data(fpath, n_tetrodes=32, downsample_factor=10)
                            else:
                                print('Warning', 'Neither Downsampled or Raw data is available, skipping ' + fpath)
                        else:
                            print('Warning', 'Downsampled data already available, skipping ' + fpath)
                    if delete_raw:
                        if NWBio.check_if_raw_data_available(fpath):
                            print(hfunct.time_string() + ' Repack ' + fpath)
                            delete_raw_data(fpath, only_if_downsampled_data_available=True)
                        else:
                            print('Warning', 'No raw data to be deleted in ' + fpath)


def main():
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
    parser.add_argument('--force_position_processing', action='store_true',
                        help='instruct reprocessing position data and overwrite previous data')
    parser.add_argument('--position_postprocessing', action='store_true',
                        help='instruct position data postprocessing. Only applicable to binary position data.\n'
                             + '(default is no postprocessing)')
    parser.add_argument('--reprocess_tracking_in_directory', action='store_true',
                        help=('Re-process tracking data with default settings in all.\n'
                              + 'files named experiment_1.nwb in all sub-folders from input directory.'))
    parser.add_argument('--position_maxjump', type=float, nargs = 1, 
                        help='enter maximum allowed jump in position data values for position data postprocessing')
    parser.add_argument('--max_clusters', type=int, nargs = 1, 
                        help='Specifies the maximum number of cluster to find. Default is 31.')
    parser.add_argument('--show_output', action='store_true', 
                        help='(for AxonaData) to open AxonaData output folder after processing')
    parser.add_argument('--datatree', action='store_true', 
                        help='to process a whole data tree with default arguments in method process_data_tree')
    parser.add_argument('--only_keep_processor', type=str, nargs=1,
                        help=('All other continuous data subsets are deleted. Specify which one to keep.\n'
                              + 'Only available with datatree option.'))
    parser.add_argument('--downsample', action='store_true', 
                        help='to downsample a whole data tree after processing. Only available with datatree option.')
    parser.add_argument('--delete_raw', action='store_true',
                        help='to delete raw data after downsampling. Only available with datatree and downsample option.')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbosity of progress and warnings. Default is False (off).')
    args = parser.parse_args()
    # Get paths to recording files
    OpenEphysDataPaths = args.paths
    if len(OpenEphysDataPaths) == 0:
        raise ValueError('Paths to file(s) required. Use --help for more info.')
    # If reprocessing tracking in directory is requested, just do that
    if args.reprocess_tracking_in_directory:
        if len(OpenEphysDataPaths) > 1:
            raise Exception('Only one root path should be specified if reprocess_tracking_in_directory is set.')
        recompute_tracking_data_for_all_files_in_directory_tree(OpenEphysDataPaths[0],
                                                                verbose=args.verbose[0])
    # If datatree processing requested, use process_data_tree method
    if args.datatree:
        if args.only_keep_processor:
            only_keep_processor = args.only_keep_processor[0]
        else:
            only_keep_processor = None
        if args.downsample:
            downsample = True
        else:
            downsample = False
        if args.delete_raw:
            delete_raw = True
        else:
            delete_raw = False
        process_data_tree(OpenEphysDataPaths[0], only_keep_processor, downsample, delete_raw)
    else:
        # Get chan input variable
        if args.chan:
            chan = [args.chan[0] - 1, args.chan[1]]
            if np.mod(chan[1] - chan[0], 4) != 0:
                raise ValueError('Channel range must cover full tetrodes')
            area_name = 'Chan' + str(args.chan[0]) + '-' + str(args.chan[1])
            channel_map = {area_name: {'list': list(range(chan[0], chan[1], 1))}}
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
        # Specify position data processing options
        if args.force_position_processing:
            force_position_processing = True
        else:
            force_position_processing = False
        pos_data_processing_kwargs = {}
        if args.position_postprocessing:
            pos_data_processing_kwargs['postprocess'] = True
        else:
            pos_data_processing_kwargs['postprocess'] = False
        if args.position_maxjump:
            pos_data_processing_kwargs['maxjump'] = args.position_maxjump[0]
        # Specify axona data options
        if args.noAxonaData:
            make_AxonaData = False
        else:
            make_AxonaData = True
        if args.ppm:
            pixels_per_metre = args.ppm[0]
        else:
            pixels_per_metre = None
        if args.max_clusters:
            max_clusters = args.max_clusters[0]
        else:
            max_clusters = 31
        if args.show_output:
            show_output = True
        else:
            show_output = False
        axonaDataArgs = (pixels_per_metre, show_output)
        # Run the script
        processing(OpenEphysDataPaths, processing_method, channel_map, noise_cut_off, 
                   threshold, make_AxonaData, axonaDataArgs, max_clusters, 
                   force_position_processing, pos_data_processing_kwargs)


if __name__ == '__main__':
    main()
