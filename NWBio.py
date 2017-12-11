# -*- coding: utf-8 -*-

import h5py
import numpy as np
import os

def load_continuous(filename):
    # Load data file
    f = h5py.File(filename, 'r')
    # Load timestamps and continuous data
    recordingKey = f['acquisition']['timeseries'].keys()[0]
    processorKey = f['acquisition']['timeseries'][recordingKey]['continuous'].keys()[0]
    continuous = f['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['data'] # not converted to microvolts!!!! need to multiply by 0.195
    timestamps = f['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['timestamps'] # not converted to microvolts!!!! need to multiply by 0.195
    data = {'continuous': continuous, 'timestamps': timestamps} 

    return data

def load_spikes(filename):
    # Outputs a list of dictionaries for each tetrode in correct order where:
    # 'waveforms' is a list of tetrode waveforms in the order of channels
    # Waveforms are passed as HDF5 file objects (handles to memory maps).
    # 'timestamps' is a list of spike detection timestamps corresponding to 'waveforms'
    # Timestampsare passed as HDF5 file objects (handles to memory maps).

    # Load data file
    f = h5py.File(filename, 'r')
    recordingKey = f['acquisition']['timeseries'].keys()[0]
    # Get data file spikes folder keys and sort them into ascending order by tetrode number
    tetrode_nrs = f['acquisition']['timeseries'][recordingKey]['spikes'].keys()
    tetrode_nrs_int = []
    for tetrode_nr in tetrode_nrs:
        tetrode_nrs_int.append(int(tetrode_nr[9:]))
    keyorder = np.argsort(np.array(tetrode_nrs_int))
    # Put waveforms and timestamps into a list of dictionaries in correct order
    data = []
    for ntet in keyorder:
        waveforms = f['acquisition']['timeseries'][recordingKey]['spikes'][tetrode_nrs[ntet]]['data']
        timestamps = f['acquisition']['timeseries'][recordingKey]['spikes'][tetrode_nrs[ntet]]['timestamps']
        data.append({'waveforms': waveforms, 'timestamps': timestamps})

    return data

def load_events(filename):
    # Outputs a dictionary timestamps and eventIDs for TTL signals received
    # timestamps are in seconds, aligned to timestamps of continuous recording
    # eventIDs indicate TTL channel number (starting from 1) and are positive for rising signals

    # Load data file
    f = h5py.File(filename, 'r')
    recordingKey = f['acquisition']['timeseries'].keys()[0]
    # Load timestamps and TLL signal info
    timestamps = f['acquisition']['timeseries'][recordingKey]['events']['ttl1']['timestamps']
    eventID = f['acquisition']['timeseries'][recordingKey]['events']['ttl1']['data']
    data = {'eventID': eventID, 'timestamps': timestamps}

    return data

def load_pos(filename, savecsv=False, postprocess=False):
    # Loads position data from NWB file
    # Optionally saves data into a csv file.

    # Load data file
    f = h5py.File(filename, 'r')
    recordingKey = f['acquisition']['timeseries'].keys()[0]
    # Load timestamps and position data
    timestamps = np.array(f['acquisition']['timeseries'][recordingKey]['events']['binary1']['timestamps'])
    xy = np.array(f['acquisition']['timeseries'][recordingKey]['events']['binary1']['data'][:,:2])
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
    # Save the data as csv file in the same folder as NWB file
    if savecsv:
        posdata = np.append(timestamps[:,None], xy.astype(np.float32), axis=1)
        nanarray = np.zeros(xy.shape, dtype=np.float32)
        nanarray[:] = np.nan
        posdata = np.append(posdata, nanarray, axis=1)
        rootfolder = os.path.dirname(filename)
        CombFileName = os.path.join(rootfolder,'PosLogComb.csv')
        with open(CombFileName, 'wb') as f:
            np.savetxt(f, posdata, delimiter=',')

    return data

def check_if_binary_pos(filename):
    # Checks if binary position data exists in NWB file
    # Load data file
    f = h5py.File(filename, 'r')
    recordingKey = f['acquisition']['timeseries'].keys()[0]
    # Check if 'binary1' is among event keys
    event_data_keys = f['acquisition']['timeseries'][recordingKey]['events'].keys()
    binaryPosData = 'binary1' in event_data_keys

    return binaryPosData
