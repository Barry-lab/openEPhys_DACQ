import os
import sys
import h5py
import HelperFunctions as hfunct
from NWBio import listBadChannels, check_if_path_exists
import numpy as np
from time import sleep

def lowpass_and_downsample_channel(fpath, path_processor, chan, lowpass_freq, downsampling):
    with h5py.File(fpath,'r') as h5file:
        data = h5file[path_processor + '/data']
        data = data[:, chan:chan + 1]
    data_array = np.array(data).squeeze()
    del data
    signal = np.float32(data_array)
    del data_array
    processed_signal = hfunct.butter_lowpass_filter(signal, lowpass_freq, sampling_rate=30000.0, filt_order=4)
    del signal
    processed_signal = np.int16(processed_signal[::downsampling])
    # NOTE! The following line roughly corrects the phase shift due to filtering
    # given that filtering is done lowpass at 500 Hz on a signal with 30 kHz samling rate,
    # which is then downsampled to 1 kHz.
    # The phase shift is unavoidable, unless filtering is done both ways, using the future of the signal.
    processed_signal = np.append(processed_signal[1:], processed_signal[-1])

    return processed_signal

def lowpass_and_downsample_channel_on_each_tetrode(fpath, path_processor, downsampling, lowpass_freq, n_tetrodes, badChans):
    with h5py.File(fpath,'r') as h5file:
        data = h5file[path_processor + '/data']
        tmp = range(data.shape[0])
        new_data_size = len(tmp[::downsampling])
        del tmp
    processed_data_array = np.zeros((new_data_size, n_tetrodes), dtype=np.int16)
    processed_chans = []
    processed_tets = []
    processed_data_list = []
    for n_tet in range(n_tetrodes):
        chans = hfunct.tetrode_channels(n_tet)
        chan = []
        for c in chans:
            if c not in badChans:
                chan.append(c)
        if len(chan) > 0:
            processed_chans.append(chan[0])
            processed_tets.append(n_tet)
    multiprocessor = hfunct.multiprocess()
    for chan in processed_chans:
        if hfunct.proceed_when_enough_memory_available(percent=0.66):
            multiprocessor.run(lowpass_and_downsample_channel, (fpath, path_processor, chan, lowpass_freq, downsampling))
            sleep(4)
    processed_data_list = multiprocessor.results()
    for n_tet, processed_data in zip(processed_tets, processed_data_list):
        processed_data_array[:, n_tet] = np.int16(processed_data)
    del processed_data_list

    return processed_data_array

def downsample_and_repack(fpath, path_processor, lowpass_freq, downsampling, n_tetrodes, auxChanStart, nwb_raw_deleted, nwb_repacked):
    badChans = listBadChannels(fpath)
    # Get timestamps for downsampled data
    with h5py.File(fpath,'r') as h5file:
        timestamps = h5file[path_processor + '/timestamps'].value
    downsampled_data_timestamps = timestamps[::downsampling]
    del timestamps
    # Get downsampled data
    downsampled_data = lowpass_and_downsample_channel_on_each_tetrode(fpath, path_processor, downsampling, lowpass_freq, n_tetrodes, badChans)
    downsampling_settings_list = ['downsampling ' + str(downsampling), 
                                  'lowpass_frequency ' + str(lowpass_freq)]
    # Write downsampled data to file
    with h5py.File(fpath,'r+') as h5file:
        h5file[path_processor + '/tetrode_lowpass'] = downsampled_data
        h5file[path_processor + '/tetrode_lowpass_timestamps'] = downsampled_data_timestamps
        asciiList = [n.encode("ascii", "ignore") for n in downsampling_settings_list]
        h5file[path_processor + '/tetrode_lowpass_info'] = h5file.create_dataset(None, (len(asciiList),),'S100', asciiList)
        # Copy AUX channels to separate array in file
        data = h5file[path_processor + '/data']
        data_AUX = np.array(data[:,auxChanStart:])
        h5file[path_processor + '/data_AUX'] = data_AUX
    # Clear up memory
    del data
    del downsampled_data
    del downsampled_data_timestamps
    del data_AUX
    # Delete raw data from file
    with h5py.File(fpath,'r+') as h5file:
        del h5file[path_processor + '/data']
    nwb_raw_deleted += 1
    # Repack NWB file to recover space
    if check_if_path_exists(fpath, path_processor + '/tetrode_lowpass'):
        os.system('h5repack ' + fpath + ' ' + (fpath + '.repacked'))
        os.system('mv ' + (fpath + '.repacked') + ' ' + fpath)
        nwb_repacked += 1

    return nwb_raw_deleted, nwb_repacked

if __name__ == '__main__':
    try:
        # Parse input arguments into variables
        fpath = sys.argv[1]
        path_processor = sys.argv[2]
        lowpass_freq = int(sys.argv[3])
        downsampling = int(sys.argv[4])
        n_tetrodes = int(sys.argv[5])
        auxChanStart = int(sys.argv[6])
        nwb_raw_deleted = int(sys.argv[7])
        nwb_repacked = int(sys.argv[8])
        # Complete processing
        input_args = (fpath, path_processor, lowpass_freq, downsampling, 
                      n_tetrodes, auxChanStart, nwb_raw_deleted, nwb_repacked)
        nwb_raw_deleted, nwb_repacked = downsample_and_repack(*input_args)
        print('successful 1 int')
        print('nwb_raw_deleted ' + str(nwb_raw_deleted) + ' int')
        print('nwb_repacked ' + str(nwb_repacked) + ' int')
    except:
        print('successful 0 int')
