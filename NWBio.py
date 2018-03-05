# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os

def get_recordingKey(filename):
    with h5py.File(filename, 'r') as h5file:
        return h5file['acquisition']['timeseries'].keys()[0]

def get_processorKey(filename):
    with h5py.File(filename, 'r') as h5file:
        return h5file['acquisition']['timeseries'][get_recordingKey()]['continuous'].keys()[0]

def load_continuous(filename):
    # Load data file
    f = h5py.File(filename, 'r')
    # Load timestamps and continuous data
    recordingKey = get_recordingKey(filename)
    processorKey = get_processorKey(filename)
    continuous = f['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['data'] # not converted to microvolts!!!! need to multiply by 0.195
    timestamps = f['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['timestamps'] # not converted to microvolts!!!! need to multiply by 0.195
    data = {'continuous': continuous, 'timestamps': timestamps} 

    return data

def load_spikes(filename, tetrode_nrs=None):
    # Outputs a list of dictionaries for each tetrode in correct order where:
    # 'waveforms' is a list of tetrode waveforms in the order of channels
    # Waveforms are passed as HDF5 file objects (handles to memory maps).
    # 'timestamps' is a list of spike detection timestamps corresponding to 'waveforms'
    # Timestampsare passed as HDF5 file objects (handles to memory maps).

    # Load data file
    f = h5py.File(filename, 'r')
    recordingKey = get_recordingKey(filename)
    # Get data file spikes folder keys and sort them into ascending order by tetrode number
    tetrode_keys = f['acquisition']['timeseries'][recordingKey]['spikes'].keys()
    if len(tetrode_keys) > 0:
        tetrode_keys_int = []
        for tetrode_key in tetrode_keys:
            tetrode_keys_int.append(int(tetrode_key[9:]))
        keyorder = list(np.argsort(np.array(tetrode_keys_int)))
        # Put waveforms and timestamps into a list of dictionaries in correct order
        data = []
        for ntet in keyorder:
            if not tetrode_nrs or ntet in tetrode_nrs:
                waveforms = f['acquisition']['timeseries'][recordingKey]['spikes'][tetrode_keys[ntet]]['data']
                timestamps = f['acquisition']['timeseries'][recordingKey]['spikes'][tetrode_keys[ntet]]['timestamps']
                data.append({'waveforms': waveforms, 'timestamps': timestamps})
        # Check if any tetrodes had spikes. If not, set data to be empty.
        tetrodes_with_spikes = 0
        for ntet in range(len(data)):
            if len(data[ntet]['timestamps']) > 0:
                tetrodes_with_spikes += 1
        if tetrodes_with_spikes == 0:
            data = []
    else:
        data = []

    return data

def load_events(filename):
    # Outputs a dictionary timestamps and eventIDs for TTL signals received
    # timestamps are in seconds, aligned to timestamps of continuous recording
    # eventIDs indicate TTL channel number (starting from 1) and are positive for rising signals

    # Load data file
    f = h5py.File(filename, 'r')
    recordingKey = get_recordingKey(filename)
    # Load timestamps and TLL signal info
    timestamps = f['acquisition']['timeseries'][recordingKey]['events']['ttl1']['timestamps'].value
    eventID = f['acquisition']['timeseries'][recordingKey]['events']['ttl1']['data'].value
    data = {'eventID': eventID, 'timestamps': timestamps}

    return data

def check_if_path_exists(filename, path):
    with h5py.File(filename,'r') as h5file:
        return path in h5file

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Only works with: numpy arrays, numpy int64 or float64, strings, bytes, lists of strings and dictionaries.
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
                    tmp = list(item.value)
                    ans[str(key)] = [str(i) for i in tmp]
                elif item.dtype == 'bool':
                    ans[str(key)] = np.array(bool(item.value))
                else:
                    ans[str(key)] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                ans[str(key)] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    else:
        ans = h5file[path].value
    return ans

def save_settings(filename, Settings, path='/'):
    '''
    Writes into an existing file if path is not yet used.
    Creates a new file if filename does not exist.
    Only works with: numpy arrays, numpy int64 or float64, strings, bytes, lists of strings and dictionaries.
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
    To load specific settings, e.g. RPiSettings, use:
        path='/RPiSettings/'
    or to load animal ID, use:
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

def save_position_data(filename, PosData, ProcessedPos=False, ReProcess=False):
    '''
    PosData is expected as dictionary with keys for each source ID
    If saving processed data, PosData is expected to be numpy array
        Use ProcessedPos=True to store processed data
        Use ReProcess=True to force overwriting existing processed data
    '''
    if os.path.isfile(filename):
        write_method = 'r+'
    else:
        write_method = 'w'
    recordingKey = get_recordingKey(filename)
    with h5py.File(filename, write_method) as h5file:
        full_path = '/acquisition/timeseries/' + recordingKey + '/tracking/'
        if not ProcessedPos:
            recursively_save_dict_contents_to_group(h5file, full_path, PosData)
        elif ProcessedPos:
            # If ReProcess is true, path is first cleared
            processed_pos_path = full_path + 'ProcessedPos/'
            if ReProcess and 'ProcessedPos' in h5file[full_path].keys():
                del h5file[processed_pos_path]
            h5file[processed_pos_path] = PosData

def check_if_tracking_data_available(filename):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/tracking/'
    return check_if_path_exists(filename, path)

def check_if_processed_position_data_available(filename):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/tracking/ProcessedPos/'
    return check_if_path_exists(filename, path)

def check_if_binary_pos(filename):
    # Checks if binary position data exists in NWB file
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/events/binary1/'
    return check_if_path_exists(filename, path)

def use_binary_pos(filename, postprocess=False):
    '''
    Copies binary position data into tracking data
    Apply postprocessing with postprocess=True
    '''
    recordingKey = get_recordingKey(filename)
    # Load timestamps and position data
    with h5py.File(filename, write_method) as h5file:
        timestamps = np.array(h5file['acquisition']['timeseries'][recordingKey]['events']['binary1']['timestamps'])
        xy = np.array(h5file['acquisition']['timeseries'][recordingKey]['events']['binary1']['data'][:,:2])
    data = {'xy': xy, 'timestamps': timestamps}
    # Postprocess the data if requested
    if postprocess:
        maxjump = 25
        keepPos = []
        lastPos = data['xy'][0,:]
        for npos in range(data['xy'].shape[0]):
            currpos = data['xy'][npos,:]
            if np.max(np.abs(lastPos - currpos)) < maxjump:
                keepPos.append(npos)
                lastPos = currpos
        keepPos = np.array(keepPos)
        print(str(data['xy'].shape[0] - keepPos.size) + ' of ' + 
              str(data['xy'].shape[0]) + ' removed in postprocessing')
        data['xy'] = data['xy'][keepPos,:]
        data['timestamps'] = data['timestamps'][keepPos]
    # Save data to ProcessedPos position with correct format
    PosData = np.append(data['timestamps'][:,None], data['xy'].astype(np.float64), axis=1)
    if PosData.shape[1] < 5:
        # Add NaNs for second LED if missing
        nanarray = np.zeros(data['xy'].shape, dtype=np.float64)
        nanarray[:] = np.nan
        PosData = np.append(PosData, nanarray, axis=1)
    save_position_data(filename, PosData, ProcessedPos=True, ReProcess=False)
