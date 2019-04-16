import os
import sys
import h5py
import HelperFunctions as hfunct
import NWBio
import numpy as np
from time import sleep

def lowpass_and_downsample_channel(
        fpath, path_processor, chan, original_sampling_rate, target_sampling_rate):
    data = NWBio.load_continuous_as_array(fpath, chan)['continuous'].squeeze()
    data = hfunct.lowpass_and_downsample(data, original_sampling_rate, target_sampling_rate)

    return data

def lowpass_and_downsample_channels(
        fpath, path_processor, channels, original_sampling_rate, target_sampling_rate):
    # Load, lowpass filter and downsample each channel in a separate process
    multiprocessor = hfunct.multiprocess()
    for chan in channels:
        hfunct.proceed_when_enough_memory_available(percent=0.50)
        print(hfunct.time_string() + ' Starting lowpass_and_downsample_channel for chan ' + str(chan))
        multiprocessor.run(lowpass_and_downsample_channel, 
                           args=(fpath, path_processor, chan, 
                                 original_sampling_rate, target_sampling_rate))
        sleep(4)
    # Collect processed data from multiprocess class
    out = multiprocessor.results()
    print(hfunct.time_string() + ' Completed lowpass_and_downsample_channel for all channels')
    
    return out

def lowpass_and_downsample_channel_on_each_tetrode(
        fpath, path_processor, original_sampling_rate, 
        target_sampling_rate, n_tetrodes, badChans):
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
        fpath, path_processor, processed_chans, original_sampling_rate, target_sampling_rate)
    new_data_size = processed_data_list[0].size
    processed_data_array = np.zeros((new_data_size, n_tetrodes), dtype=np.int16)
    for n_tet, processed_data in zip(processed_tets, processed_data_list):
        processed_data_array[:, n_tet] = np.int16(processed_data)

    return processed_data_array, processed_chans

def downsample_and_repack(fpath, path_processor, downsampling, n_tetrodes, auxChanStart, nwb_raw_deleted, nwb_repacked):
    # Get original sampling rate and compute target rate based on downsampling factor
    original_sampling_rate = NWBio.OpenEphys_SamplingRate()
    target_sampling_rate = int(NWBio.OpenEphys_SamplingRate() / downsampling)
    # Get list of bad channels
    badChans = NWBio.listBadChannels(fpath)
    # Get downsampled data
    downsampled_data, used_chans = lowpass_and_downsample_channel_on_each_tetrode(
        fpath, path_processor, original_sampling_rate, target_sampling_rate, n_tetrodes, badChans)
    downsampling_settings_list = ['original_sampling_rate ' + str(original_sampling_rate), 
                                  'downsampled_sampling_rate ' + str(target_sampling_rate), 
                                  'downsampled_channels ' + str(','.join(map(str, used_chans)))]
    # Get timestamps for downsampled data
    with h5py.File(fpath,'r') as h5file:
        timestamps = np.array(h5file[path_processor + '/timestamps']).squeeze()
    timestamps = timestamps[::downsampling]
    # Ensure timestamps and downsampled data have same number of samples
    assert timestamps.size == downsampled_data.shape[0], \
        'Downsampled timestamps does not match number of samples in downsampled continuous data.'
    # Write downsampled data to file
    with h5py.File(fpath,'r+') as h5file:
        h5file[path_processor + '/downsampled_tetrode_data'] = downsampled_data
        h5file[path_processor + '/downsampled_timestamps'] = timestamps
        asciiList = [n.encode("ascii", "ignore") for n in downsampling_settings_list]
        h5file[path_processor + '/downsampling_info'] = h5file.create_dataset(None, (len(asciiList),),'S100', asciiList)
    # Clear up memory
    del downsampled_data
    del timestamps
    # List AUX channels
    with h5py.File(fpath, 'r') as h5file:
        n_raw_channels = h5file[path_processor + '/data'].shape[1]
    aux_chan_list = range(auxChanStart, n_raw_channels)
    # Get downsampled AUX channels
    processed_data_list = lowpass_and_downsample_channels(
                              fpath, path_processor, aux_chan_list, 
                              original_sampling_rate, target_sampling_rate)
    downsampled_AUX = np.concatenate([x[:, None] for x in processed_data_list], axis=1)
    downsampled_AUX = downsampled_AUX.astype(np.int16)
    # Write downsampled_AUX to file
    with h5py.File(fpath,'r+') as h5file:
        h5file[path_processor + '/downsampled_AUX_data'] = downsampled_AUX
    # Clear up memory
    del downsampled_AUX
    # Delete raw data and original timestamps from file
    with h5py.File(fpath,'r+') as h5file:
        del h5file[path_processor + '/data']
        del h5file[path_processor + '/timestamps']
    nwb_raw_deleted += 1
    # Repack NWB file to recover space
    if NWBio.check_if_path_exists(fpath, path_processor + '/downsampled_tetrode_data'):
        # Create a repacked copy of the file
        os.system('h5repack ' + fpath + ' ' + (fpath + '.repacked'))
        # Check that the new file is not corrupted
        with h5py.File(fpath + '.repacked','r') as h5file:
            _ = h5file[path_processor + '/downsampled_tetrode_data'].shape
        # Replace original file with repacked file
        os.system('mv ' + (fpath + '.repacked') + ' ' + fpath)
        # Check that the new file is not corrupted
        with h5py.File(fpath,'r') as h5file:
            _ = h5file[path_processor + '/downsampled_tetrode_data'].shape
        nwb_repacked += 1

    return nwb_raw_deleted, nwb_repacked

if __name__ == '__main__':
    try:
        # Parse input arguments into variables
        fpath = sys.argv[1]
        path_processor = sys.argv[2]
        downsampling = int(sys.argv[3])
        n_tetrodes = int(sys.argv[4])
        auxChanStart = int(sys.argv[5])
        nwb_raw_deleted = int(sys.argv[6])
        nwb_repacked = int(sys.argv[7])
        # Complete processing
        input_args = (fpath, path_processor, downsampling, 
                      n_tetrodes, auxChanStart, nwb_raw_deleted, nwb_repacked)
        nwb_raw_deleted, nwb_repacked = downsample_and_repack(*input_args)
        print(','.join(['successful 1 int', 
                        'nwb_raw_deleted ' + str(nwb_raw_deleted) + ' int', 
                        'nwb_repacked ' + str(nwb_repacked) + ' int']))
    except:
        print('successful 0 int')
