# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
import sys
from openEPhys_DACQ.HelperFunctions import tetrode_channels, channels_tetrode, closest_argmin
from pprint import pprint
from copy import copy
import argparse
import importlib
from tqdm import tqdm


def OpenEphys_SamplingRate():
    return 30000


def bitVolts():
    return 0.195


def spike_waveform_leftwards_shift():
    """Returns the leftwards shift of waveforms from detection point in seconds.

    :return:
    :rtype: float
    """
    return 6 * (1.0 / OpenEphys_SamplingRate())


def get_filename(path):
    if not os.path.isfile(path):
        return os.path.join(path, 'experiment_1.nwb')
    else:
        return path


def delete_path_in_file(filename, path):
    with h5py.File(filename, 'r+') as h5file:
        del h5file[path]


def get_recordingKey(filename):
    with h5py.File(filename, 'r') as h5file:
        return list(h5file['acquisition']['timeseries'].keys())[0]


def get_all_processorKeys(filename):
    with h5py.File(filename, 'r') as h5file:
        return list(h5file['acquisition']['timeseries'][get_recordingKey(filename)]['continuous'].keys())


def get_processorKey(filename):
    return get_all_processorKeys(filename)[0]


def get_all_processor_paths(filename):
    return ['/acquisition/timeseries/' + get_recordingKey(filename)
            + '/continuous/' + processorKey
            for processorKey in get_all_processorKeys(filename)]


def get_processor_path(filename):
    return '/acquisition/timeseries/' + get_recordingKey(filename) \
           + '/continuous/' + get_processorKey(filename)


def check_if_open_ephys_nwb_file(filename):
    """
    Returns True if processor path can be identified
    in the file and False otherwise.
    """
    try:
        processor_path = get_processor_path(filename)
        with h5py.File(filename, 'r') as h5file:
            return processor_path in h5file
    except:
        return False


def get_downsampled_data_paths(filename):
    """
    Returns paths to downsampled data in NWB file.

    :param filename: path to NWB file
    :type filename: str
    :return: paths
    :rtype: dict
    """
    processor_path = get_processor_path(filename)
    return {'tetrode_data': processor_path + '/downsampled_tetrode_data/',
            'aux_data': processor_path + '/downsampled_AUX_data/',
            'timestamps': processor_path + '/downsampled_timestamps/',
            'info': processor_path + '/downsampling_info/'}


def check_if_downsampled_data_available(filename):
    """
    Checks if downsampled data is available in the NWB file.

    :param filename: path to NWB file
    :type filename: str
    :return: available
    :rtype: bool
    """
    paths = get_downsampled_data_paths(filename)
    with h5py.File(filename, 'r') as h5file:
        # START Workaround for older downsampled datasets
        if '/acquisition/timeseries/recording1/continuous/processor102_100/tetrode_lowpass' in h5file:
            return True
        # END Workaround for older downsampled datasets
        for path in [paths[key] for key in paths]:
            if not (path in h5file):
                return False
        if h5file[paths['tetrode_data']].shape[0] == 0:
            return False
        if h5file[paths['tetrode_data']].shape[0] \
                != h5file[paths['timestamps']].shape[0] \
                != h5file[paths['aux_data']].shape[0]:
            return False

    return True


def get_raw_data_paths(filename):
    """
    Returns paths to downsampled data in NWB file.

    :param filename: path to NWB file
    :type filename: str
    :return: paths
    :rtype: dict
    """
    processor_path = get_processor_path(filename)
    return {'continuous': processor_path + '/data',
            'timestamps': processor_path + '/timestamps'}


def check_if_raw_data_available(filename):
    """
    Returns paths to raw data in NWB file.

    :param filename:
    :type filename: str
    :return: paths
    :rtype: dict
    """
    paths = get_raw_data_paths(filename)
    if all([check_if_path_exists(filename, paths[key]) for key in paths]):
        return True
    else:
        return False


def save_downsampling_info_to_disk(filename, info):
    # Get paths to respective dataset locations
    paths = get_downsampled_data_paths(filename)
    # Ensure dictionary fields are in correct format
    info = {'original_sampling_rate': np.int64(info['original_sampling_rate']),
            'downsampled_sampling_rate': np.int64(info['downsampled_sampling_rate']),
            'downsampled_channels': np.array(info['downsampled_channels'], dtype=np.int64)}
    # Write data to disk
    with h5py.File(filename, 'r+') as h5file:
        recursively_save_dict_contents_to_group(h5file, paths['info'], info)


def save_downsampled_data_to_disk(filename, tetrode_data, timestamps, aux_data, info):
    # Get paths to respective dataset locations
    paths = get_downsampled_data_paths(filename)
    # Write data to disk
    save_downsampling_info_to_disk(filename, info)
    with h5py.File(filename, 'r+') as h5file:
        h5file[paths['tetrode_data']] = tetrode_data
        h5file[paths['timestamps']] = timestamps
        h5file[paths['aux_data']] = aux_data


def delete_raw_data(filename, only_if_downsampled_data_available=True):
    if only_if_downsampled_data_available:
        if not check_if_downsampled_data_available(filename):
            print('Warning', 'Downsampled data not available in NWB file. Raw data deletion aborted.')
            return None
    if not check_if_raw_data_available(filename):
        print('Warning', 'Raw data not available to be deleted in: ' + filename)
    else:
        raw_data_paths = get_raw_data_paths(filename)
        with h5py.File(filename,'r+') as h5file:
            for path in [raw_data_paths[key] for key in raw_data_paths]:
                del h5file[path]


def repack_NWB_file(filename, replace_original=True, check_validity_with_downsampled_data=True):
    # Create a repacked copy of the file
    os.system('h5repack ' + filename + ' ' + (filename + '.repacked'))
    # Check that the new file is not corrupted
    if check_validity_with_downsampled_data:
        if not check_if_downsampled_data_available(filename):
            raise Exception('Downsampled data cannot be found in repacked file. Original file not replaced.')
    # Replace original file with repacked file
    if replace_original:
        os.system('mv ' + (filename + '.repacked') + ' ' + filename)


def repack_all_nwb_files_in_directory_tree(folder_path, replace_original=True,
                                           check_validity_with_downsampled_data=True):
    # Commence directory walk
    for dir_name, subdirList, fileList in os.walk(folder_path):
        for fname in fileList:
            fpath = os.path.join(dir_name, fname)
            if fname == 'experiment_1.nwb':
                print('Repacking file {}'.format(fpath))
                repack_NWB_file(fpath, replace_original=replace_original,
                                check_validity_with_downsampled_data=check_validity_with_downsampled_data)


def list_AUX_channels(filename, n_tetrodes):
    data = load_continuous(filename)
    n_channels = data['continuous'].shape[1]
    data['file_handle'].close()
    aux_chan_list = range(n_tetrodes * 4 - 1, n_channels)

    return aux_chan_list


def load_continuous(filename):
    # Load data file
    h5file = h5py.File(filename, 'r')
    # Load timestamps and continuous data
    recordingKey = get_recordingKey(filename)
    processorKey = get_processorKey(filename)
    path = '/acquisition/timeseries/' + recordingKey + '/continuous/' + processorKey
    if check_if_path_exists(filename, path + '/data'):
        continuous = h5file[path + '/data'] # not converted to microvolts!!!! need to multiply by 0.195
        timestamps = h5file[path + '/timestamps']
        data = {'continuous': continuous, 'timestamps': timestamps, 'file_handle': h5file}
    else:
        data = None

    return data


def load_raw_data_timestamps_as_array(filename):
    data = load_continuous(filename)
    timestamps = np.array(data['timestamps']).squeeze()
    data['file_handle'].close()

    return timestamps


def load_data_columns_as_array(filename, data_path, first_column, last_column):
    """
    Loads a contiguous columns of dataset efficiently from HDF5 dataset.
    """
    with h5py.File(filename, 'r') as h5file:
        data = h5file[data_path]
        data = h5file[data_path][:, first_column:last_column]

    return data


def load_data_as_array(filename, data_path, columns):
    """
    Fast way of reading a single column or a set of columns.
    
    filename - str - full path to file
    columns  - list - column numbers to include (starting from 0).
               Single column can be given as a single list element or int.
               Columns in the list must be in sorted (ascending) order.
    """
    # Make columns variable into a list if int given
    if isinstance(columns, int):
        columns = [columns]
    # Check that all elements of columns are integers
    if isinstance(columns, list):
        for column in columns:
            if not isinstance(column, int):
                raise ValueError('columns argument must be a list of int values.')
    else:
        raise ValueError('columns argument must be list or int.')
    # Check that column number are sorted
    if sorted(columns) != columns:
        raise ValueError('columns was not in sorted (ascending) order.')
    # Check that data is available, otherwise return None
    if not check_if_path_exists(filename, data_path):
        raise ValueError('File ' + filename + '\n'
                         + 'Does not contain path ' + data_path)
    # Find contiguous column groups
    current_column = columns[0]
    column_groups = [current_column]
    for i in range(1, len(columns)):
        if (columns[i] - columns[i - 1]) == 1:
            column_groups.append(current_column)
        else:
            column_groups.append(columns[i])
            current_column = columns[i]
    # Find start and end column numbers for contiguous groups
    column_ranges = []
    for first_channel in sorted(set(column_groups)):
        last_channel = first_channel + column_groups.count(first_channel)
        column_ranges.append((first_channel, last_channel))
    # Get contiguous column segments for each group
    column_group_data = []
    for column_range in column_ranges:
        column_group_data.append(
            load_data_columns_as_array(filename, data_path, *column_range))
    # Concatenate column groups
    data = np.concatenate(column_group_data, axis=1)

    return data


def load_continuous_as_array(filename, channels):
    """
    Fast way of reading a single channel or a set of channels.
    
    filename - str - full path to file
    channels - list - channel numbers to include (starting from 0).
               Single channel can be given as a single list element or int.
               Channels in the list must be in sorted (ascending) order.
    """
    # Generate path to raw continuous data
    root_path = '/acquisition/timeseries/' + get_recordingKey(filename) \
                + '/continuous/' + get_processorKey(filename)
    data_path = root_path + '/data'
    timestamps_path = root_path + '/timestamps'
    # Check that data is available, otherwise return None
    if not check_if_path_exists(filename, data_path):
        return None
    if not check_if_path_exists(filename, timestamps_path):
        return None
    # Load continuous data
    continuous = load_data_as_array(filename, data_path, channels)
    # Load timestamps for data
    with h5py.File(filename, 'r') as h5file:
        timestamps = np.array(h5file[timestamps_path])
    # Arrange output into a dictionary
    data = {'continuous': continuous, 'timestamps': timestamps}

    return data


def remove_surrounding_binary_markers(text):
    if text.startswith("b'"):
        text = text[2:]
    if text.endswith("'"):
        text = text[:-1]
    return text


def get_downsampling_info_old(filename):
    # Generate path to downsampling data info
    root_path = '/acquisition/timeseries/' + get_recordingKey(filename) \
                + '/continuous/' + get_processorKey(filename)
    data_path = root_path + '/downsampling_info'
    # Load info from file
    with h5py.File(filename, 'r') as h5file:
        data = h5file[data_path]
        data = [str(i) for i in data]
    # Remove b'x' markers from strings if present. Python 3 change.
    data = list(map(remove_surrounding_binary_markers, data))
    # Parse elements in loaded data
    info_dict = {}
    for x in data:
        key, value = x.split(' ')
        if key == 'original_sampling_rate':
            info_dict[key] = np.int64(value)
        elif key == 'downsampled_sampling_rate':
            info_dict[key] = np.int64(value)
        elif key == 'downsampled_channels':
            info_dict[key] = np.array(list(map(int, value.split(',')))).astype(np.int64)

    return info_dict


def get_downsampling_info(filename):
    root_path = '/acquisition/timeseries/' + get_recordingKey(filename) \
                + '/continuous/' + get_processorKey(filename)
    data_path = root_path + '/downsampling_info/'
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, data_path)


def load_downsampled_tetrode_data_as_array(filename, tetrode_nrs):
    """
    Returns a dict with downsampled continuous data for requested tetrodes
    
    filename                    - str - full path to file
    tetrode_nrs                 - list - tetrode numbers to include (starting from 0).
                                Single tetrode can be given as a single list element or int.
                                Tetrode numbers in the list must be in sorted (ascending) order.
                                If data is not available for a given tetrode number, error is raised.
    """
    # Generate path to raw continuous data
    root_path = '/acquisition/timeseries/' + get_recordingKey(filename) \
                + '/continuous/' + get_processorKey(filename)
    data_path = root_path + '/downsampled_tetrode_data'
    timestamps_path = root_path + '/downsampled_timestamps'
    # Check that data is available, otherwise return None
    if not check_if_path_exists(filename, data_path):
        return None
    if not check_if_path_exists(filename, timestamps_path):
        return None
    # Get info on downsampled data
    info = get_downsampling_info(filename)
    sampling_rate = int(info['downsampled_sampling_rate'])
    downsampled_channels = list(info['downsampled_channels'])
    # Map tetrode_nrs elements to columns in downsampled_tetrode_data
    columns = []
    channels_used = []
    tetrode_nrs_remaining = copy(tetrode_nrs)
    for tetrode_nr in tetrode_nrs:
        for chan in tetrode_channels(tetrode_nr):
            if chan in downsampled_channels:
                columns.append(downsampled_channels.index(chan))
                channels_used.append(chan)
                tetrode_nrs_remaining.pop(tetrode_nrs_remaining.index(tetrode_nr))
                break
    # Check that all tetrode numbers were mapped
    if len(tetrode_nrs_remaining) > 0:
        raise Exception('The following tetrodes were not represented in downsampled data\n' \
                        + ','.join(list(map(str, tetrode_nrs_remaining))))
    # Load continuous data
    continuous = load_data_as_array(filename, data_path, columns)
    # Load timestamps for data
    with h5py.File(filename, 'r') as h5file:
        timestamps = np.array(h5file[timestamps_path])
    # Arrange output into a dictionary
    data = {'continuous': continuous, 'timestamps': timestamps,
            'tetrode_nrs': tetrode_nrs, 'channels': channels_used,
            'sampling_rate': sampling_rate}

    return data


def empty_spike_data():
    """
    Creates a fake waveforms of 0 values and at timepoint 0
    """
    waveforms = np.zeros((1,4,40), dtype=np.int16)
    timestamps = np.array([0], dtype=np.float64)

    return {'waveforms': waveforms, 'timestamps': timestamps}


def get_tetrode_nrs_if_spikes_available(filename, spike_name='spikes'):
    """
    Returns a list of tetrode numbers if spikes available in NWB file.
    """
    spikes_path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/' + spike_name + '/'
    # Get tetrode keys if available
    with h5py.File(filename, 'r') as h5file:
        if not (spikes_path in h5file):
            # Return empty list if spikes data not available
            return []
        tetrode_keys = list(h5file[spikes_path].keys())
    # Return empty list if spikes not available on any tetrode
    if len(tetrode_keys) == 0:
        return []
    # Extract tetrode numbers
    tetrode_nrs = []
    for tetrode_key in tetrode_keys:
        tetrode_nrs.append(int(tetrode_key[9:]) - 1)
    # Sort tetrode numbers in ascending order
    tetrode_nrs.sort()

    return tetrode_nrs


def construct_paths_to_tetrode_spike_data(filename, tetrode_nrs, spike_name='spikes'):
    spikes_path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/' + spike_name + '/'
    return [(spikes_path + 'electrode' + str(tetrode_nr + 1) + '/') for tetrode_nr in tetrode_nrs]


def count_spikes(filename, tetrode_nrs, spike_name='spikes', use_idx_keep=False):
    """
    :param filename: full path to NWB file
    :type filename: str
    :param tetrode_nrs: tetrode numbers to count spikes for
    :type tetrode_nrs: list
    :param spike_name: type of spikes to look for (field in NWB file)
    :type spike_name: str
    :param use_idx_keep: If False (default) all spikes are counted, otherwise only filtered spikes are counted
    :type use_idx_keep: bool
    :return: total number of spikes on each tetrode
    :rtype: list
    """
    tetrode_paths = construct_paths_to_tetrode_spike_data(filename, tetrode_nrs, spike_name=spike_name)
    count = []
    with h5py.File(filename, 'r') as h5file:
        for tetrode_path in tetrode_paths:
            if use_idx_keep:
                count.append(sum(np.array(h5file[tetrode_path + 'idx_keep'][()]).squeeze()))
            else:
                count.append(h5file[tetrode_path + 'timestamps/'].shape[0])

    return count


def load_spikes(filename, spike_name='spikes', tetrode_nrs=None, use_idx_keep=False,
                use_badChan=False, no_waveforms=False, clustering_name=None, verbose=True):
    """
    Inputs:
        filename - pointer to NWB file to load
        tetrode_nrs [list]    - can be a list of tetrodes to load (from 0)
        use_idx_keep [bool]   - if True, only outputs spikes according to idx_keep of tetrode, if available
        use_badChan [bool]    - if True, sets all spikes on badChannels to 0
        no_waveforms [bool]   - if True, waveforms are not loaded
        clustering_name [str] - if specified, clusterID will be loaded from:
                              electrode[nr]/clustering/clustering_name
        verbose [bool]        - prints out loading progress bar if True (default)
    Output:
        List of dictionaries for each tetrode in correct order where:
        List is empty, if no spike data detected
        'waveforms' is a list of tetrode waveforms in the order of channels
        'timestamps' is a list of spike detection timestamps corresponding to 'waveforms'
        If available, two more variables will be in the dictionary
        'idx_keep' is boolan index for 'waveforms' and 'timestamps' indicating the spikes
            that are to be used for further processing (based on filtering for artifacts etc)
        'clusterIDs' is the cluster identities of spikes in 'waveforms'['idx_keep',:,:]
    """
    # If not provided, get tetrode_nrs
    if tetrode_nrs is None:
        tetrode_nrs = get_tetrode_nrs_if_spikes_available(filename, spike_name=spike_name)
    tetrode_paths = construct_paths_to_tetrode_spike_data(filename, tetrode_nrs, spike_name=spike_name)
    with h5py.File(filename, 'r') as h5file:
        # Put waveforms and timestamps into a list of dictionaries in correct order
        data = []
        if verbose:
            print('Loading tetrodes from {}'.format(filename))
        iterable = zip(tetrode_nrs, tetrode_paths)
        for nr_tetrode, tetrode_path in (tqdm(iterable, total=len(tetrode_nrs)) if verbose else iterable):
            # Load waveforms and timestamps
            if no_waveforms:
                waveforms = empty_spike_data()['waveforms']
            else:
                waveforms = h5file[tetrode_path + 'data/'][()]
            timestamps = h5file[tetrode_path + 'timestamps/'][()]
            if not isinstance(timestamps, np.ndarray):
                timestamps = np.array([timestamps])
            if waveforms.shape[0] == 0:
                # If no waveforms are available, enter one waveform of zeros at timepoint zero
                waveforms = empty_spike_data()['waveforms']
                timestamps = empty_spike_data()['timestamps']
            # Arrange waveforms, timestamps and nr_tetrode into a dictionary
            tet_data = {'waveforms': waveforms,
                        'timestamps': timestamps,
                        'nr_tetrode': nr_tetrode}
            # Include idx_keep if available
            idx_keep_path = tetrode_path + 'idx_keep'
            if idx_keep_path in h5file:
                tet_data['idx_keep'] = np.array(h5file[idx_keep_path][()])
                if use_idx_keep:
                    # If requested, filter wavefoms and timestamps based on idx_keep
                    if np.sum(tet_data['idx_keep']) == 0:
                        tet_data['waveforms'] = empty_spike_data()['waveforms']
                        tet_data['timestamps'] = empty_spike_data()['timestamps']
                    else:
                        if not no_waveforms:
                            tet_data['waveforms'] = tet_data['waveforms'][tet_data['idx_keep'], :, :]
                        tet_data['timestamps'] = tet_data['timestamps'][tet_data['idx_keep']]
            # Include clusterIDs if available
            if clustering_name is None:
                clusterIDs_path = tetrode_path + 'clusterIDs'
            else:
                clusterIDs_path = tetrode_path + '/clustering/' + clustering_name
            if clusterIDs_path in h5file:
                tet_data['clusterIDs'] = np.int16(h5file[clusterIDs_path][()]).squeeze()
            # Set spikes to zeros for channels in badChan list if requested
            if use_badChan and not no_waveforms:
                badChan = listBadChannels(filename)
                if len(badChan) > 0:
                    for nchan in tetrode_channels(nr_tetrode):
                        if nchan in badChan:
                            tet_data['waveforms'][:, np.mod(nchan, 4), :] = 0
            data.append(tet_data)
        
        return data

def save_spikes(filename, tetrode_nr, data, timestamps, spike_name='spikes', overwrite=False):
    """
    Stores spike data in NWB file in the same format as with OpenEphysGUI.
    tetrode_nr=0 for first tetrode.
    """
    if data.dtype != np.int16:
        raise ValueError('Waveforms are not int16.')
    if timestamps.dtype != np.float64:
        raise ValueError('Timestamps are not float64.')
    recordingKey = get_recordingKey(filename)
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/' + spike_name + '/' + \
           'electrode' + str(tetrode_nr + 1) + '/'
    if check_if_path_exists(filename, path):
        if overwrite:
            # If overwrite is true, path is first cleared
            with h5py.File(filename, 'r+') as h5file:
                del h5file[path]
        else:
            raise Exception('Spikes already in file and overwrite not requested.\n' \
                            + 'File: ' + filename + '\n' \
                            + 'path: ' + path)
    with h5py.File(filename, 'r+') as h5file:
        h5file[path + 'data'] = data
        h5file[path + 'timestamps'] = np.float64(timestamps).squeeze()

def processing_method_and_spike_name_combinations():
    """
    Outputs a list of potential processing_method and spike_name combinations
    """
    processing_methods = ['klustakwik', 'klustakwik_raw', 'kilosort']
    spike_names = ['spikes', 'spikes_raw', 'spikes_kilosort']

    return processing_methods, spike_names

def get_spike_name_for_processing_method(processing_method):
    processing_methods, spike_names = processing_method_and_spike_name_combinations()
    spike_name = spike_names[processing_methods.index(processing_method)]

    return spike_name

def load_events(filename):
    # Outputs a dictionary timestamps and eventIDs for TTL signals received
    # timestamps are in seconds, aligned to timestamps of continuous recording
    # eventIDs indicate TTL channel number (starting from 1) and are positive for rising signals

    # Load data file
    recordingKey = get_recordingKey(filename)
    with h5py.File(filename, 'r') as h5file:
        # Load timestamps and TLL signal info
        timestamps = h5file['acquisition']['timeseries'][recordingKey]['events']['ttl1']['timestamps'][()]
        eventID = h5file['acquisition']['timeseries'][recordingKey]['events']['ttl1']['data'][()]
        data = {'eventID': eventID, 'timestamps': timestamps}

    return data

def load_GlobalClock_timestamps(filename, GlobalClock_TTL_channel=1):
    """
    Returns timestamps of GlobalClock TTL pulses.
    """
    data = load_events(filename)
    return data['timestamps'][data['eventID'] == GlobalClock_TTL_channel]


def load_network_events(filename):
    """returns network_events_data

    Extracts the list of network messages from NWB file 
    and returns it along with corresponding timestamps
    in dictionary with keys ['messages', 'timestamps']
    
    'messages' - list of str
    
    'timestamps' - list of float

    :param filename: full path to NWB file
    :type filename: str
    :return: network_events_data
    :rtype: dict
    """
    # Load data file
    recordingKey = get_recordingKey(filename)
    with h5py.File(filename, 'r') as h5file:
        # Load timestamps and messages
        timestamps = h5file['acquisition']['timeseries'][recordingKey]['events']['text1']['timestamps'][()]
        messages = h5file['acquisition']['timeseries'][recordingKey]['events']['text1']['data'][()]
    messages = [x.decode('utf-8') for x in messages]
    timestamps = [float(x) for x in timestamps]

    data = {'messages': messages, 'timestamps': timestamps}

    return data


def check_if_path_exists(filename, path):
    with h5py.File(filename, 'r') as h5file:
        return path in h5file


def save_list_of_dicts_to_group(h5file, path, dlist, overwrite=False, list_suffix='_NWBLIST'):
    # Check that all elements are dictionaries
    for dic in dlist:
        if not isinstance(dic, dict):
            raise Exception('List elements must be dictionaries')
    # Write elements to file
    for i, dic in enumerate(dlist):
        recursively_save_dict_contents_to_group(h5file, (path + str(i) + '/'), dic,
                                                overwrite=overwrite, list_suffix=list_suffix)


def recursively_save_dict_contents_to_group(h5file, path, dic, overwrite=False, list_suffix='_NWBLIST', verbose=False):
    """
    h5file - h5py.File
    path   - str       - path to group in h5file. Must end with '/'
    overwrite - bool   - any dictionary elements or lists that already exist are overwritten.
                         Default is False, if elements already exist in NWB file, error is raised.
    list_suffix - str  - suffix used to highlight paths created from lists of dictionaries.
                         Must be consistent when saving and loading data.
    verbose - bool     - If True (default is False), h5file path used is printed for each recursion

    Only works with: numpy arrays, numpy int64 or float64, strings, bytes, lists of strings and dictionaries these are contained in.
    Also works with lists dictionaries as part of the hierachy.
    Long lists of dictionaries are discouraged, as individual groups are created for each element.
    """
    if verbose:
        print(path)
    if len(dic) == 0:
        if path in h5file:
            del h5file[path]
        h5file.create_group(path)
    for key, item in dic.items():
        if isinstance(item, (int, float)):
            item = np.array(item)
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            if overwrite:
                if path + key in h5file:
                    del h5file[path + key]
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item,
                                                    overwrite=overwrite, list_suffix=list_suffix,
                                                    verbose=verbose)
        elif isinstance(item, list):
            if all(isinstance(i, str) for i in item):
                if overwrite:
                    if path + key in h5file:
                        del h5file[path + key]
                asciiList = [n.encode("ascii", "ignore") for n in item]
                h5file[path + key] = h5file.create_dataset(None, (len(asciiList),),'S100', asciiList)
            else:
                if overwrite:
                    if path + key + list_suffix in h5file:
                        del h5file[path + key + list_suffix]
                save_list_of_dicts_to_group(h5file, path + key + list_suffix + '/', item, 
                                            overwrite=overwrite, list_suffix=list_suffix)
        elif item is None:
            h5file.create_group(path + key)
        else:
            raise ValueError('Cannot save %s type'%type(item) + ' from ' + path + key)


def convert_bytes_to_string(b):
    """
    If input is bytes, returns str decoded with utf-8

    :param b:
    :type b: bytes
    :return: string decoded with utf-8 if input is bytes object, otherwise returns unchanged input
    :rtype: str
    """
    if isinstance(b, bytes):
        if sys.version_info >= (3, 0):
            return str(b, 'utf-8')
        else:
            return str(b.decode('utf-8'))
    else:
        return b


def load_list_of_dicts_from_group(h5file, path, list_suffix='_NWBLIST', ignore=()):
    # Load all elements on this path
    items = []
    for key in list(h5file[path].keys()):
        items.append(
            (int(key), recursively_load_dict_contents_from_group(h5file, path + key + '/', 
                                                                 list_suffix=list_suffix,
                                                                 ignore=ignore))
        )
    # Create a list from items sorted by group keys
    ans = [item for _, item in sorted(items)]

    return ans


def recursively_load_dict_contents_from_group(h5file, path, list_suffix='_NWBLIST', ignore=()):
    """
    Returns value at path if it has no further items

    h5file - h5py.File
    path   - str       - path to group in h5file. Must end with '/'
    list_suffix - str  - suffix used to highlight paths created from lists of dictionaries.
                         Must be consistent when saving and loading data.
    ignore - tuple     - paths including elements matching any element in this tuple return None
    """
    if not path.endswith('/'):
        raise ValueError('Input path must end with "/"')

    if path.split('/')[-2] in ignore or path.split('/')[-2][:-len(list_suffix)] in ignore:
        ans = None

    elif path[:-1].endswith(list_suffix):
        ans = load_list_of_dicts_from_group(h5file, path, list_suffix=list_suffix,
                                            ignore=ignore)
    elif hasattr(h5file[path], 'items'):

        ans = {}
        for key, item in h5file[path].items():

            if key.endswith(list_suffix):
                ans[str(key)[:-len(list_suffix)]] = load_list_of_dicts_from_group(
                    h5file, path + key + '/', list_suffix=list_suffix,
                    ignore=ignore
                )

            elif isinstance(item, h5py._hl.dataset.Dataset):
                if 'S100' == item.dtype:
                    tmp = list(item[()])
                    ans[str(key)] = [convert_bytes_to_string(i) for i in tmp]
                elif item.dtype == 'bool' and item.ndim == 0:
                    ans[str(key)] = np.array(bool(item[()]))
                else:
                    ans[str(key)] = convert_bytes_to_string(item[()])

            elif isinstance(item, h5py._hl.group.Group):
                ans[str(key)] = recursively_load_dict_contents_from_group(h5file, path + key + '/',
                                                                          ignore=ignore)

    else:
        ans = convert_bytes_to_string(h5file[path][()])

    return ans


def save_settings(filename, Settings, path='/'):
    """
    Writes into an existing file if path is not yet used.
    Creates a new file if filename does not exist.
    Only works with: numpy arrays, numpy int64 or float64, strings, bytes, lists of strings and dictionaries these are contained in.
    To save specific subsetting, e.g. TaskSettings, use:
        Settings=TaskSetttings, path='/TaskSettings/'
    """
    full_path = '/general/data_collection/Settings' + path
    if os.path.isfile(filename):
        write_method = 'r+'
    else:
        write_method = 'w'
    with h5py.File(filename, write_method) as h5file:
        recursively_save_dict_contents_to_group(h5file, full_path, Settings)

def load_settings(filename, path='/', ignore=()):
    """
    By default loads all settings from path
        '/general/data_collection/Settings/'
    or for example to load animal ID, use:
        path='/General/animal/'

    ignore - tuple - any paths including any element of ignore are returned as None
    """
    full_path = '/general/data_collection/Settings' + path
    with h5py.File(filename, 'r') as h5file:
        data = recursively_load_dict_contents_from_group(h5file, full_path, ignore=ignore)

    return data


def check_if_settings_available(filename, path='/'):
    """
    Returns whether settings information exists in NWB file
    Specify path='/General/badChan/' to check for specific settings
    """
    full_path = '/general/data_collection/Settings' + path
    with h5py.File(filename, 'r') as h5file:
        return full_path in h5file


def save_analysis(filename, data, overwrite=False, complete_overwrite=False, verbose=False):
    """Stores analysis results from nested dictionary to /analysis path in NWB file.

    See :py:func:`NWBio.recursively_save_dict_contents_to_group` for details on supported data structures.

    :param str filename: path to NWB file
    :param dict data: analysis data to be stored in NWB file
    :param bool overwrite: if True, any existing data at same dictionary keys
                           as in previously saved data is overwritten.
                           Default is False.
    :param bool complete_overwrite: if True, all previous analysis data is discarded before writing.
                                    Default is False.
    :param bool verbose: if True (default is False), the path in file for each element is printed.
    """
    with h5py.File(filename, 'r+') as h5file:
        if complete_overwrite:
            del h5file['/analysis']
        recursively_save_dict_contents_to_group(h5file, '/analysis/', data, overwrite=overwrite, verbose=verbose)


def load_analysis(filename, ignore=()):
    """Loads analysis results from /analysis path in NWB file into a dictionary.

    :param str filename: path to NWB file
    :param tuple ignore: paths containing any element of ignore are terminated with None.
        In the output dictionary any elements downstream of a key matching any element of ignore
        is not loaded and dictionary tree is terminated at that point with value None.
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/analysis/', ignore=ignore)


def listBadChannels(filename):
    if check_if_settings_available(filename,'/General/badChan/'):
        badChanString = load_settings(filename,'/General/badChan/')
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
        badChan = list(np.array(list(map(int, badChanStringList))) - 1)
    else:
        badChan = []

    return badChan

def save_tracking_data(filename, TrackingData, ProcessedPos=False, overwrite=False):
    """
    TrackingData is expected as dictionary with keys for each source ID
    If saving processed data, TrackingData is expected to be numpy array
        Use ProcessedPos=True to store processed data
        Use overwrite=True to force overwriting existing processed data
    """
    if os.path.isfile(filename):
        write_method = 'r+'
    else:
        write_method = 'w'
    recordingKey = get_recordingKey(filename)
    with h5py.File(filename, write_method) as h5file:
        full_path = '/acquisition/timeseries/' + recordingKey + '/tracking/'
        if not ProcessedPos:
            recursively_save_dict_contents_to_group(h5file, full_path, TrackingData)
        elif ProcessedPos:
            processed_pos_path = full_path + 'ProcessedPos/'
            # If overwrite is true, path is first cleared
            if overwrite:
                if full_path in h5file and 'ProcessedPos' in list(h5file[full_path].keys()):
                    del h5file[processed_pos_path]
            h5file[processed_pos_path] = TrackingData


def get_recording_cameraIDs(filename):
    path = '/general/data_collection/Settings/CameraSettings/CameraSpecific'
    with h5py.File(filename, 'r') as h5file:
        if path in h5file:
            return list(h5file[path].keys())


def load_raw_tracking_data(filename, cameraID, specific_path=None):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/tracking/' + cameraID + '/'
    if not (specific_path is None):
        path = path + '/' + specific_path + '/'
    with h5py.File(filename, 'r') as h5file:
        if path in h5file:
            return recursively_load_dict_contents_from_group(h5file, path)

def load_processed_tracking_data(filename, subset='ProcessedPos'):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/tracking/'
    path = path + subset
    with h5py.File(filename, 'r') as h5file:
        return np.array(h5file[path][()])

def get_processed_tracking_data_timestamp_edges(filename, subset='ProcessedPos'):
    if check_if_processed_position_data_available(filename):
        data = load_processed_tracking_data(filename, subset=subset)
        edges = [data[0, 0], data[-1, 0]]
    else:
        print('Warning! ProcessedPos not available. Using continuous data timestamps')
        h5file = h5py.File(filename, 'r')
        recordingKey = get_recordingKey(filename)
        processorKey = get_processorKey(filename)
        path = '/acquisition/timeseries/' + recordingKey + '/continuous/' + processorKey + '/timestamps'
        edges = [h5file[path][0], h5file[path][-1]]
        h5file.close()

    return edges

def check_if_tracking_data_available(filename):
    if check_if_settings_available(filename, path='/General/Tracking/'):
        return load_settings(filename, path='/General/Tracking/')
    else:
        return False

def check_if_processed_position_data_available(filename, subset='ProcessedPos'):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/tracking/'
    path = path + subset
    return check_if_path_exists(filename, path)

def check_if_binary_pos(filename):
    # Checks if binary position data exists in NWB file
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/events/binary1/'
    return check_if_path_exists(filename, path)


def save_tetrode_idx_keep(filename, ntet, idx_keep, spike_name='spikes', overwrite=False):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/' + spike_name + '/' + \
           'electrode' + str(ntet + 1) + '/idx_keep'
    with h5py.File(filename, 'r+') as h5file:
        if path in h5file:
            if overwrite:
                del h5file[path]
            else:
                raise ValueError('Tetrode ' + str(ntet + 1) + ' idx_keep already exists in ' + filename)
        h5file[path] = idx_keep

def save_tetrode_clusterIDs(filename, ntet, clusterIDs, spike_name='spikes', overwrite=False):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/' + spike_name + '/' + \
           'electrode' + str(ntet + 1) + '/clusterIDs'
    with h5py.File(filename, 'r+') as h5file:
        if path in h5file:
            if overwrite:
                del h5file[path]
            else:
                raise ValueError('Tetrode ' + str(ntet + 1) + ' clusterIDs already exists in ' + filename)
        h5file[path] = np.int16(clusterIDs).squeeze()

def fill_empty_dictionary_from_source(selection, src_dict):
    """
    Populates a dictionary with None values with values from a source
    dictionary with identical structure.
    """
    dst_dict = copy(selection)
    for key, item in dst_dict.items():
        if isinstance(item, dict):
            dst_dict[key] = fill_empty_dictionary_from_source(item, src_dict[key])
        elif item is None:
            dst_dict[key] = src_dict[key]
        else:
            raise ValueError('Destination dictionary has incorrect.')

    return dst_dict


def get_recording_start_timestamp_offset(filename):
    """Returns the first timestamp of raw or downsampled continuous data.

    :param str filename: path to NWB file
    :return: first timestamp of continuous data
    :rtype: float
    """
    if check_if_raw_data_available(filename):
        path = get_raw_data_paths(filename)['timestamps']
    elif check_if_downsampled_data_available(filename):
        path = get_downsampled_data_paths(filename)['timestamps']
    else:
        raise Exception('NWB file does not contain raw or downsampled data ' + filename)
    with h5py.File(filename, 'r') as h5file:
        return float(h5file[path][0:1])


def get_recording_full_duration(filename):
    """Returns the total duration from first to last timestamp of
    raw or downsampled continuous data.

    :param str filename: path to NWB file
    :return: total duration from first to last timestamp of continuous data
    :rtype: float
    """
    if check_if_raw_data_available(filename):
        path = get_raw_data_paths(filename)['timestamps']
    elif check_if_downsampled_data_available(filename):
        path = get_downsampled_data_paths(filename)['timestamps']
    else:
        raise Exception('NWB file does not contain raw or downsampled data ' + filename)
    with h5py.File(filename, 'r') as h5file:
        return float(h5file[path][-1]) - float(h5file[path][0:1])


def import_task_specific_log_parser(task_name):
    """
    Returns LogParser module for the specific task.

    :param task_name: name of the task
    :type task_name: str
    :return: TaskLogParser
    :rtype: module
    """
    if task_name == 'Pellets_and_Rep_Milk_Task':  # Temporary workaround to function with older files
        task_name = 'Pellets_and_Rep_Milk'
    try:
        return importlib.import_module('.Tasks.' + task_name + '.LogParser', package='openEPhys_DACQ')
    except ModuleNotFoundError:
        print('Task {} LogParser not found. Returning None.'.format(task_name))
        return None


def load_task_name(filename):
    """
    Returns the name of the task active in the recording.

    :param filename: absolute path to NWB recording file
    :type filename: str
    :return: task_name
    :rtype: str
    """
    return load_settings(filename, path='/TaskSettings/name/')


def get_recording_log_parser(filename, final_timestamp=None):
    """Finds task specific LogParser class and returns it initialized
    with network events from that recording.

    :param str filename:
    :return: Task specific log parser initialized with network events
    :rtype: LogParser class
    """
    task_log_parser = import_task_specific_log_parser(load_task_name(filename))
    if task_log_parser is None:
        return None
    else:
        return task_log_parser.LogParser(task_settings=load_settings(filename, path='/TaskSettings/'),
                                         final_timestamp=final_timestamp,
                                         **load_network_events(filename))


def get_channel_map(filename):
    return load_settings(filename, '/General/channel_map/')


def list_tetrode_nrs_for_area_channel_map(area_channel_map):
    return list(set([channels_tetrode(chan) for chan in list(area_channel_map['list'])]))


def get_channel_map_with_tetrode_nrs(filename):
    channel_map = get_channel_map(filename)
    for area in channel_map:
        channel_map[area]['tetrode_nrs'] = list_tetrode_nrs_for_area_channel_map(channel_map[area])

    return channel_map


def check_if_channel_maps_are_same(channel_map_1, channel_map_2):
    """
    Determines if two channel maps are identical
    """
    # Check that there are same number of areas in the dictionary
    if len(channel_map_1) != len(channel_map_2):
        return False
    # Sort the area names because dictionary is not ordered
    channel_map_1_keys = sorted(list(channel_map_1.keys()))
    channel_map_2_keys = sorted(list(channel_map_2.keys()))
    # Check that the areas have the same name
    for n_area in range(len(channel_map_1_keys)):
        if channel_map_1_keys[n_area] != channel_map_2_keys[n_area]:
            return False
    # Check that the channel lists are the same
    for area in channel_map_1_keys:
        if not all(channel_map_1[area]['list'] == channel_map_2[area]['list']):
            return False

    return True


def estimate_open_ephys_timestamps_from_other_timestamps(open_ephys_global_clock_times, other_global_clock_times,
                                                         other_times, other_times_divider=None):
    """Returns Open Ephys timestamps for each timestamp from another device by synchronising with global clock.

    Note, other times must be in same units as open_ephys_global_clock_times. Most likely seconds.
    For example, Raspberry Pi camera timestamps would need to be divided by 10 ** 6

    :param numpy.ndarray open_ephys_global_clock_times: shape (N,)
    :param numpy.ndarray other_global_clock_times: shape (M,)
    :param numpy.ndarray other_times: shape (K,)
    :param int other_times_divider: if provided, timestamps from the other devices are divided by this value
        before matching to Open Ephys time. This allows inputting timestamps from other device in original units.
        In case of Raspberry Pi camera timestamps, this value should be 10 ** 6.
        If this value is not provided, all provided timestamps must be in same units.
    :return: open_ephys_times
    :rtype: numpy.ndarray
    """

    # Crop data if more timestamps recorded on either system.
    if open_ephys_global_clock_times.size > other_global_clock_times.size:
        open_ephys_global_clock_times = open_ephys_global_clock_times[:other_global_clock_times.size]
        print('[ Warning ] OpenEphys recorded more GlobalClock TTL pulses than other system.\n' +
              'Dumping extra OpenEphys timestamps from the end.')
    elif open_ephys_global_clock_times.size < other_global_clock_times.size:
        other_global_clock_times = other_global_clock_times[:open_ephys_global_clock_times.size]
        print('[ Warning ] Other system recorded more GlobalClock TTL pulses than Open Ephys.\n' +
              'Dumping extra other system timestamps from the end.')

    # Find closest other_global_clock_times indices to each other_times
    other_times_gc_indices = closest_argmin(other_times, other_global_clock_times)

    # Compute difference from the other_global_clock_times for each value in other_times
    other_times_nearest_global_clock_times = other_global_clock_times[other_times_gc_indices]
    other_times_global_clock_delta = other_times - other_times_nearest_global_clock_times

    # Convert difference values to Open Ephys timestamp units
    if not (other_times_divider is None):
        other_times_global_clock_delta = other_times_global_clock_delta / float(other_times_divider)

    # Use other_times_global_clock_delta to estimate timestamps in OpenEphys time
    other_times_nearest_open_ephys_global_clock_times = open_ephys_global_clock_times[other_times_gc_indices]
    open_ephys_times = other_times_nearest_open_ephys_global_clock_times + other_times_global_clock_delta

    return open_ephys_times


def extract_recording_info(filename, selection='default'):
    """
    Returns recording info for the recording file.

    selection - allows specifying which data return
        'default' - some hard-coded selection of data
        'all' - all of the recording settings
        dict - a dictionary with the same exact keys and structure
               as the recording settings, with None for item values
               and missing keys for unwanted elements. The dictionary
               will be returned with None values populated by values
               from recording settings.
    """
    recording_info = {}
    if isinstance(selection, str) and selection == 'default':
        recording_info.update(load_settings(filename, '/General/'))
        del recording_info['experimenter']
        del recording_info['rec_file_path']
        del recording_info['root_folder']
        if recording_info['TaskActive']:
            recording_info.update({'TaskName': load_settings(filename, '/TaskSettings/name/')})
        for key in list(recording_info['channel_map'].keys()):
            del recording_info['channel_map'][key]['list']
        pos_edges = get_processed_tracking_data_timestamp_edges(filename)
        recording_info['duration'] = pos_edges[1] - pos_edges[0]
        recording_info['duration (min)'] = int(round((pos_edges[1] - pos_edges[0]) / 60))
        recording_info['time'] = load_settings(filename, '/Time/')
    elif isinstance(selection, str) and selection == 'all':
        recording_info = load_settings(filename)
    elif isinstance(selection, dict):
        full_recording_info = load_settings(filename)
        recording_info = fill_empty_dictionary_from_source(selection, full_recording_info)

    return recording_info


def display_recording_data(root_path, selection='default'):
    """
    Prints recording info for the whole directory tree.
    """
    for dirName, subdirList, fileList in os.walk(root_path):
        for fname in fileList:
            if fname == 'experiment_1.nwb':
                filename = os.path.join(dirName, fname)
                recording_info = extract_recording_info(filename, selection)
                print('Data on path: ' + dirName)
                pprint(recording_info)


if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Extract info from Open Ephys.')
    parser.add_argument('root_path', type=str, nargs=1, 
                        help='Root directory for recording(s)')
    args = parser.parse_args()
    # Get paths to recording files
    display_recording_data(args.root_path[0])
