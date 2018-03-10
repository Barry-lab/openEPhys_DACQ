# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os
from HelperFunctions import tetrode_channels

def get_recordingKey(filename):
    with h5py.File(filename, 'r') as h5file:
        return h5file['acquisition']['timeseries'].keys()[0]

def get_processorKey(filename):
    with h5py.File(filename, 'r') as h5file:
        return h5file['acquisition']['timeseries'][get_recordingKey(filename)]['continuous'].keys()[0]

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

def empty_spike_data():
    '''
    Creates a fake waveforms of 0 values and at timepoint 0
    '''
    waveforms = np.zeros((1,40,4), dtype=np.int16)
    timestamps = np.array([0], dtype=np.float64)

    return {'waveforms': waveforms, 'timestamps': timestamps}

def load_spikes(filename, tetrode_nrs=None, use_idx_keep=False, use_badChan=False):
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
        # Get data file spikes folder keys and sort them into ascending order by tetrode number
        tetrode_keys = h5file['acquisition']['timeseries'][recordingKey]['spikes'].keys()
        if len(tetrode_keys) > 0:
            # Sort tetrode keys into ascending order
            tetrode_keys_int = []
            for tetrode_key in tetrode_keys:
                tetrode_keys_int.append(int(tetrode_key[9:]) - 1)
            keyorder = list(np.argsort(np.array(tetrode_keys_int)))
            tetrode_keys = [tetrode_keys[i] for i in keyorder]
            tetrode_keys_int = [tetrode_keys_int[i] for i in keyorder]
            # Put waveforms and timestamps into a list of dictionaries in correct order
            data = []
            for ntet in range(len(tetrode_keys)):
                if tetrode_nrs is None or tetrode_keys_int[ntet] in tetrode_nrs:
                    waveforms = h5file['/acquisition/timeseries/' + recordingKey + '/spikes/' + \
                                       tetrode_keys[ntet] + '/data/'].value
                    timestamps = h5file['/acquisition/timeseries/' + recordingKey + '/spikes/' + \
                                        tetrode_keys[ntet] + '/timestamps/'].value
                    if waveforms.shape[0] == 0:
                        # If no waveforms are available, enter one waveform of zeros at timepoint zero
                        waveforms = empty_spike_data()['waveforms']
                        timestamps = empty_spike_data()['timestamps']
                    tet_data = {'waveforms': waveforms, 
                                'timestamps': timestamps, 
                                'nr_tetrode': tetrode_keys_int[ntet]}
                    # Include idx_keep if available
                    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/spikes/' + \
                           'electrode' + str(ntet + 1) + '/idx_keep'
                    if check_if_path_exists(filename, path):
                        tet_data['idx_keep'] = h5file[path]
                        if use_idx_keep:
                            if np.sum(tet_data['idx_keep']) == 0:
                                tet_data['waveforms'] = empty_spike_data()['waveforms']
                                tet_data['timestamps'] = empty_spike_data()['timestamps']
                            else:
                                tet_data['waveforms'] = tet_data['waveforms'][tet_data['idx_keep'],:,:]
                                tet_data['timestamps'] = tet_data['timestamps'][tet_data['idx_keep']]
                    # Include clusterIDs if available
                    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/spikes/' + \
                           'electrode' + str(ntet + 1) + '/clusterIDs'
                    if check_if_path_exists(filename, path):
                        tet_data['clusterIDs'] = h5file[path]
                    # Set spikes to zeros for channels in badChan list
                    if use_badChan:
                        badChan = listBadChannels(filename)
                        if len(badChan) > 0:
                            for nchan in tetrode_channels(tetrode_keys_int[ntet]):
                                if nchan in badChan:
                                    tet_data['waveforms'][:,:,np.mod(nchan,4)] = 0
                    data.append(tet_data)
        else:
            data = []
        
        return data

def save_spikes(filename, ntet, data, timestamps):
    '''
    Stores spike data in NWB file in the same format as with OpenEphysGUI.
    ntet=0 for first tetrode.
    '''
    if data.dtype != np.int16:
        raise ValueError('Waveforms are not int16.')
    if timestamps.dtype != np.float64:
        raise ValueError('Timestamps are not float64.')
    recordingKey = get_recordingKey(filename)
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/spikes/' + \
           'electrode' + str(ntet + 1) + '/'
    if not check_if_path_exists(filename, path):
        with h5py.File(filename, 'r+') as h5file:
            h5file[path + 'data'] = data
            h5file[path + 'timestamps'] = timestamps

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
        badChan = list(np.array(map(int, badChanStringList)) - 1)
    else:
        badChan = []

    return badChan

def save_tracking_data(filename, TrackingData, ProcessedPos=False, ReProcess=False):
    '''
    TrackingData is expected as dictionary with keys for each source ID
    If saving processed data, TrackingData is expected to be numpy array
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
            recursively_save_dict_contents_to_group(h5file, full_path, TrackingData)
        elif ProcessedPos:
            # If ReProcess is true, path is first cleared
            processed_pos_path = full_path + 'ProcessedPos/'
            if ReProcess and 'ProcessedPos' in h5file[full_path].keys():
                del h5file[processed_pos_path]
            h5file[processed_pos_path] = TrackingData

def load_tracking_data(filename, subset='ProcessedPos'):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/tracking/'
    path = path + subset
    with h5py.File(filename, 'r') as h5file:
        return np.array(h5file[path].value)

def get_processed_tracking_data_timestamp_edges(filename):
    data = load_tracking_data(filename, subset='ProcessedPos')
    edges = [data[0, 0], data[-1, 0]]
    return edges

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

def save_tetrode_idx_keep(filename, ntet, idx_keep, overwrite=False):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/spikes/' + \
           'electrode' + str(ntet + 1) + '/idx_keep'
    with h5py.File(filename, 'r+') as h5file:
        if path in h5file:
            if overwrite:
                del h5file[path]
            else:
                raise ValueError('Tetrode ' + str(ntet + 1) + ' idx_keep already exists in ' + filename)
        h5file[path] = idx_keep

def save_tetrode_clusterIDs(filename, ntet, clusterIDs, overwrite=False):
    path = '/acquisition/timeseries/' + get_recordingKey(filename) + '/spikes/' + \
           'electrode' + str(ntet + 1) + '/clusterIDs'
    with h5py.File(filename, 'r+') as h5file:
        if path in h5file:
            if overwrite:
                del h5file[path]
            else:
                raise ValueError('Tetrode ' + str(ntet + 1) + ' clusterIDs already exists in ' + filename)
        h5file[path] = clusterIDs
