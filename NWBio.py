# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
from HelperFunctions import tetrode_channels, time_string, tetrode_channels
from pprint import pprint
from copy import copy
import argparse
from TrackingDataProcessing import remove_tracking_data_jumps
from TrackingDataProcessing import iteratively_combine_multicamera_data_for_recording


def OpenEphys_SamplingRate():
    return 30000


def get_filename(folder_path):
    if not os.path.isfile(folder_path):
        return os.path.join(folder_path, 'experiment_1.nwb')

def get_recordingKey(filename):
    with h5py.File(filename, 'r') as h5file:
        return list(h5file['acquisition']['timeseries'].keys())[0]

def get_processorKey(filename):
    with h5py.File(filename, 'r') as h5file:
        return list(h5file['acquisition']['timeseries'][get_recordingKey(filename)]['continuous'].keys())[0]

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


def load_data_columns_as_array(filename, data_path, first_column, last_column):
    '''
    Loads a contiguous columns of dataset efficiently from HDF5 dataset.
    '''
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


def get_downsampling_info(filename):
    # Generate path to downsampling data info
    root_path = '/acquisition/timeseries/' + get_recordingKey(filename) \
                + '/continuous/' + get_processorKey(filename)
    data_path = root_path + '/downsampling_info'
    # Load info from file
    with h5py.File(filename, 'r') as h5file:
        data = h5file[data_path]
        data = [str(i) for i in data]
    # Parse elements in loaded data
    info_dict = {}
    for x in data:
        key, value = x.split(' ')
        info_dict[key] = value

    return info_dict


def load_downsampled_tetrode_data_as_array(filename, tetrode_nrs):
    """
    Returns a dict with downsampled continuous data for requested tetrodes
    
    filename    - str - full path to file
    tetrode_nrs - list - tetrode numbers to include (starting from 0).
                  Single tetrode can be given as a single list element or int.
                  Tetrode numbers in the list must be in sorted (ascending) order.
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
    # Get list of channels in downsampled data
    downsampling_info = get_downsampling_info(filename)
    downsampled_channels = map(int, downsampling_info['downsampled_channels'].split(','))
    # Map tetrode_nrs elements to columns in downsampled_tetrode_data
    columns = []
    tetrode_nrs_remaining = copy(tetrode_nrs)
    for tetrode_nr in tetrode_nrs:
        for chan in tetrode_channels(tetrode_nr):
            if chan in downsampled_channels:
                columns.append(downsampled_channels.index(chan))
                tetrode_nrs_remaining.pop(tetrode_nrs_remaining.index(tetrode_nr))
                break
    # Check that all tetrode numbers were mapped
    if len(tetrode_nrs_remaining) > 0:
        raise Exception('The following tetrodes were not represented in downsampled data\n' \
                        + ','.join(tetrode_nrs_remaining))
    # Load continuous data
    continuous = load_data_as_array(filename, data_path, columns)
    # Load timestamps for data
    with h5py.File(filename, 'r') as h5file:
        timestamps = np.array(h5file[timestamps_path])
    # Arrange output into a dictionary
    data = {'continuous': continuous, 'timestamps': timestamps}

    return data


def empty_spike_data():
    '''
    Creates a fake waveforms of 0 values and at timepoint 0
    '''
    waveforms = np.zeros((1,4,40), dtype=np.int16)
    timestamps = np.array([0], dtype=np.float64)

    return {'waveforms': waveforms, 'timestamps': timestamps}

def load_spikes(filename, spike_name='spikes', tetrode_nrs=None, use_idx_keep=False, use_badChan=False):
    '''
    Inputs:
        filename - pointer to NWB file to load
        tetrode_nrs [list] - can be a list of tetrodes to load (from 0)
        use_idx_keep [bool] - if True, only outputs spikes according to idx_keep of tetrode, if available
        use_badChan [bool] - if True, sets all spikes on badChannels to 0
    Output:
        List of dictionaries for each tetrode in correct order where:
        List is empty, if no spike data detected
        'waveforms' is a list of tetrode waveforms in the order of channels
        'timestamps' is a list of spike detection timestamps corresponding to 'waveforms'
        If available, two more variables will be in the dictionary
        'idx_keep' is boolan index for 'waveforms' and 'timestamps' indicating the spikes
            that are to be used for further processing (based on filtering for artifacts etc)
        'clusterIDs' is the cluster identities of spikes in 'waveforms'['idx_keep',:,:]
    '''

    recordingKey = get_recordingKey(filename)
    with h5py.File(filename, 'r') as h5file:
        spike_name_path = '/acquisition/timeseries/' + recordingKey + '/' + spike_name
        if spike_name_path in h5file:
            # Get data file spikes folder keys and sort them into ascending order by tetrode number
            tetrode_keys = list(h5file[spike_name_path].keys())
        else:
            return []
        if len(tetrode_keys) > 0:
            # Sort tetrode keys into ascending order
            tetrode_keys_int = []
            for tetrode_key in tetrode_keys:
                tetrode_keys_int.append(int(tetrode_key[9:]) - 1)
            keyorder = list(np.argsort(np.array(tetrode_keys_int)))
            tetrode_keys = [tetrode_keys[i] for i in keyorder]
            tetrode_keys_int = [tetrode_keys_int[i] for i in keyorder]
            if tetrode_nrs is None:
                tetrode_nrs = tetrode_keys_int
            # Put waveforms and timestamps into a list of dictionaries in correct order
            data = []
            for nr_tetrode in tetrode_nrs:
                if nr_tetrode in tetrode_keys_int:
                    print([time_string(), 'DEBUG: loading tetrode ', nr_tetrode])
                    # If data is available for this tetrode
                    ntet = tetrode_keys_int.index(nr_tetrode)
                    waveforms = h5file['/acquisition/timeseries/' + recordingKey + '/' + spike_name + '/' + \
                                       tetrode_keys[ntet] + '/data/'][()]
                    timestamps = h5file['/acquisition/timeseries/' + recordingKey + '/' + spike_name + '/' + \
                                        tetrode_keys[ntet] + '/timestamps/'][()]
                    if waveforms.shape[0] == 0:
                        # If no waveforms are available, enter one waveform of zeros at timepoint zero
                        waveforms = empty_spike_data()['waveforms']
                        timestamps = empty_spike_data()['timestamps']
                    tet_data = {'waveforms': np.int16(waveforms), 
                                'timestamps': np.float64(timestamps).squeeze(), 
                                'nr_tetrode': nr_tetrode}
                    # Include idx_keep if available
                    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/' + spike_name + '/' + \
                           tetrode_keys[ntet] + '/idx_keep'
                    if check_if_path_exists(filename, path):
                        tet_data['idx_keep'] = np.array(h5file[path][()]).squeeze()
                        if use_idx_keep:
                            if np.sum(tet_data['idx_keep']) == 0:
                                tet_data['waveforms'] = empty_spike_data()['waveforms']
                                tet_data['timestamps'] = empty_spike_data()['timestamps']
                            else:
                                tet_data['waveforms'] = tet_data['waveforms'][tet_data['idx_keep'],:,:]
                                tet_data['timestamps'] = tet_data['timestamps'][tet_data['idx_keep']]
                    # Include clusterIDs if available
                    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/' + spike_name + '/' + \
                           tetrode_keys[ntet] + '/clusterIDs'
                    if check_if_path_exists(filename, path):
                        tet_data['clusterIDs'] = np.int16(h5file[path][()]).squeeze()
                    # Set spikes to zeros for channels in badChan list
                    if use_badChan:
                        badChan = listBadChannels(filename)
                        if len(badChan) > 0:
                            for nchan in tetrode_channels(nr_tetrode):
                                if nchan in badChan:
                                    tet_data['waveforms'][:,np.mod(nchan,4),:] = 0
                    data.append(tet_data)
                else:
                    data.append({'nr_tetrode': nr_tetrode})
        else:
            data = []
        
        return data

def save_spikes(filename, tetrode_nr, data, timestamps, spike_name='spikes', overwrite=False):
    '''
    Stores spike data in NWB file in the same format as with OpenEphysGUI.
    tetrode_nr=0 for first tetrode.
    '''
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
    '''
    Outputs a list of potential processing_method and spike_name combinations
    '''
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
    '''
    Returns timestamps of GlobalClock TTL pulses.
    '''
    data = load_events(filename)
    return data['timestamps'][data['eventID'] == GlobalClock_TTL_channel]


def load_network_events(filename):
    '''returns network_events_data

    Extracts the list of network messages from NWB file 
    and returns it along with corresponding timestamps
    in dictionary with keys ['messages', 'timestamps']
    
    'messages' - list of str
    
    'timestamps' - list of float

    :param filename: full path to NWB file
    :type filename: str
    :return: network_events_data
    :rtype: dict
    '''
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
    with h5py.File(filename,'r') as h5file:
        return path in h5file

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Only works with: numpy arrays, numpy int64 or float64, strings, bytes, lists of strings and dictionaries these are contained in.
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        elif isinstance(item, list):
            if all(isinstance(i, str) for i in item):
                asciiList = [n.encode("ascii", "ignore") for n in item]
                h5file[path + key] = h5file.create_dataset(None, (len(asciiList),),'S100', asciiList)
            else:
                raise ValueError('Cannot save %s type'%type(item) + ' from ' + path + key)
        else:
            raise ValueError('Cannot save %s type'%type(item) + ' from ' + path + key)

def recursively_load_dict_contents_from_group(h5file, path):
    """
    Returns value at path if it has no further items
    """
    if hasattr(h5file[path], 'items'):
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                if 'S100' == item.dtype:
                    tmp = list(item[()])
                    ans[str(key)] = [str(i) for i in tmp]
                elif item.dtype == 'bool':
                    ans[str(key)] = np.array(bool(item[()]))
                else:
                    ans[str(key)] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
                ans[str(key)] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    else:
        ans = h5file[path][()]
        if isinstance(ans, bytes):
            ans = ans.decode('utf-8')

    return ans

def save_settings(filename, Settings, path='/'):
    '''
    Writes into an existing file if path is not yet used.
    Creates a new file if filename does not exist.
    Only works with: numpy arrays, numpy int64 or float64, strings, bytes, lists of strings and dictionaries these are contained in.
    To save specific subsetting, e.g. TaskSettings, use:
        Settings=TaskSetttings, path='/TaskSettings/'
    '''
    full_path = '/general/data_collection/Settings' + path
    if os.path.isfile(filename):
        write_method = 'r+'
    else:
        write_method = 'w'
    with h5py.File(filename, write_method) as h5file:
        recursively_save_dict_contents_to_group(h5file, full_path, Settings)

def load_settings(filename, path='/'):
    '''
    By default loads all settings from path
        '/general/data_collection/Settings/'
    or for example to load animal ID, use:
        path='/General/animal/'
    '''
    full_path = '/general/data_collection/Settings' + path
    with h5py.File(filename, 'r') as h5file:
        data = recursively_load_dict_contents_from_group(h5file, full_path)

    return data

def check_if_settings_available(filename, path='/'):
    '''
    Returns whether settings information exists in NWB file
    Specify path='/General/badChan/' to check for specific settings
    '''
    full_path = '/general/data_collection/Settings' + path
    with h5py.File(filename,'r') as h5file:
        return full_path in h5file

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
    '''
    TrackingData is expected as dictionary with keys for each source ID
    If saving processed data, TrackingData is expected to be numpy array
        Use ProcessedPos=True to store processed data
        Use overwrite=True to force overwriting existing processed data
    '''
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

def load_raw_tracking_data(filename, cameraID, specific_path=None):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/tracking/' + cameraID
    if not (specific_path is None):
        path = path + '/' + specific_path
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

def use_binary_pos(filename, postprocess=False, maxjump=25):
    '''
    Copies binary position data into tracking data
    Apply postprocessing with postprocess=True
    '''
    recordingKey = get_recordingKey(filename)
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
    save_tracking_data(filename, posdata, ProcessedPos=True, overwrite=True)

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
    '''
    Populates a dictionary with None values with values from a source
    dictionary with identical structure.
    '''
    dst_dict = copy(selection)
    for key, item in dst_dict.items():
        print(['key', key])
        print(['item', item])
        if isinstance(item, dict):
            dst_dict[key] = fill_empty_dictionary_from_source(item, src_dict[key])
        elif item is None:
            dst_dict[key] = src_dict[key]
        else:
            raise ValueError('Destination dictionary has incorrect.')

    return dst_dict

def extract_recording_info(filename, selection='default'):
    '''
    Returns recording info for the recording file.

    selection - allows specifying which data return
        'default' - some hard-coded selection of data
        'all' - all of the recording settings
        dict - a dictionary with the same exact keys and structure
               as the recording settings, with None for item values
               and missing keys for unwanted elements. The dictionary
               will be returned with None values populated by values
               from recording settings.
    '''
    if isinstance(selection, str) and selection == 'default':
        recording_info = {}
        recording_info.update(load_settings(filename, '/General/'))
        del recording_info['experimenter']
        del recording_info['rec_file_path']
        del recording_info['root_folder']
        if recording_info['TaskActive']:
            recording_info.update({'TaskName': load_settings(filename, '/TaskSettings/name/')})
        for key in list(recording_info['channel_map'].keys()):
            del recording_info['channel_map'][key]['list']
        pos_edges = get_processed_tracking_data_timestamp_edges(filename)
        recording_info['duration (min)'] = int(round((pos_edges[1] - pos_edges[0]) / 60))
    elif isinstance(selection, str) and selection == 'all':
        recording_info = load_settings(filename)
    elif isinstance(selection, dict):
        full_recording_info = load_settings(filename)
        recording_info = fill_empty_dictionary_from_source(selection, full_recording_info)

    return recording_info


def process_tracking_data(filename, save_to_file=False):
    # Get CameraSettings
    CameraSettings = load_settings(filename, '/CameraSettings/')
    cameraIDs = sorted(CameraSettings['CameraSpecific'].keys())
    # Get global clock timestamps
    OE_GC_times = load_GlobalClock_timestamps(filename)
    # Get arena_size
    arena_size = load_settings(filename, '/General/arena_size')
    # Load position data for all cameras
    posdatas = []
    for cameraID in cameraIDs:
        posdatas.append(load_raw_tracking_data(filename, cameraID))
    ProcessedPos = iteratively_combine_multicamera_data_for_recording(
        CameraSettings, arena_size, posdatas, OE_GC_times)
    if save_to_file:
        # Save corrected data to file
        save_tracking_data(filename, ProcessedPos, ProcessedPos=True, ReProcess=False)

    return ProcessedPos


def display_recording_data(root_path, selection='default'):
    '''
    Prints recording info for the whole directory tree.
    '''
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
