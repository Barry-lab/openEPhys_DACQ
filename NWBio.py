# -*- coding: utf-8 -*-

import h5py
import numpy as np

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
