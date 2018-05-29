import os
import sys
import h5py
import HelperFunctions as hfunct
from NWBio import listBadChannels
import numpy as np
from time import sleep

def get_recordingKey(fpath):
    with h5py.File(fpath, 'r') as h5file:
        return h5file['acquisition']['timeseries'].keys()[0]

def get_processorKey(fpath):
    with h5py.File(fpath, 'r') as h5file:
        return h5file['acquisition']['timeseries'][get_recordingKey(fpath)]['continuous'].keys()[0]

def check_if_path_exists(fpath, path):
    with h5py.File(fpath,'r') as h5file:
        return path in h5file

if len(sys.argv) < 2:
    raise ValueError('Enter path to process as the first argument!')

root_path = sys.argv[1]

nwb_noRawData = 0
nwb_raw_deleted = 0
nwb_repacked = 0
nwb_noSpikes = 0
ioError = 0
kwd_deleted = 0
continuous_deleted = 0

def lowpass_and_downsample_channel(fpath, raw_data_path, chan, lowpass_freq, downsampling):
    print('Loading chan ' + str(chan) + '...')
    with h5py.File(fpath,'r') as h5file:
        data = h5file[raw_data_path]
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
    print('Processed chan ' + str(chan) + '.')

    return processed_signal

def lowpass_and_downsample_channel_on_each_tetrode(fpath, raw_data_path, downsampling, lowpass_freq, n_tetrodes, badChans):
    with h5py.File(fpath,'r') as h5file:
        data = h5file[raw_data_path]
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
            multiprocessor.run(lowpass_and_downsample_channel, (fpath, raw_data_path, chan, lowpass_freq, downsampling))
            sleep(4)
    processed_data_list = multiprocessor.results()
    for n_tet, processed_data in zip(processed_tets, processed_data_list):
        processed_data_array[:, n_tet] = np.int16(processed_data)

    return processed_data_array

downsampling = 30
lowpass_freq = 500
for dirName, subdirList, fileList in os.walk(root_path):
    for fname in fileList:
        if not ('Experiment' in dirName):
            fpath = os.path.join(dirName, fname)
            if fname == 'experiment_1.nwb':
                # Check if spikes recorded
                try:
                    SpikesAvailable = False
                    recordingKey = get_recordingKey(fpath)
                    processorKey = get_processorKey(fpath)
                    raw_data_path = '/acquisition/timeseries/' + recordingKey + '/continuous/' + processorKey + '/data'
                    with h5py.File(fpath,'r') as h5file:
                        raw_data_available = raw_data_path in h5file
                    if raw_data_available:
                        with h5py.File(fpath,'r') as h5file:
                            nchan = h5file[raw_data_path].shape[1]
                        spikes_path = '/acquisition/timeseries/' + get_recordingKey(fpath) + '/spikes/'
                        if check_if_path_exists(fpath, spikes_path):
                            with h5py.File(fpath, 'r') as h5file:
                                n_tetrodes = len(h5file[spikes_path].keys())
                            if nchan >= 128 and n_tetrodes == 32:
                                SpikesAvailable = True
                                auxChanStart = 128
                            if nchan < 128 and n_tetrodes == 16:
                                SpikesAvailable = True
                                auxChanStart = 64
                        if SpikesAvailable:
                            print('Downsample and save: ' + fpath)
                            with h5py.File(fpath,'r') as h5file:
                                timestamps = h5file['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['timestamps'].value
                                downsampled_data_timestamps = timestamps[::downsampling]
                            badChans = listBadChannels(fpath)
                            downsampled_data = lowpass_and_downsample_channel_on_each_tetrode(fpath, raw_data_path, downsampling, lowpass_freq, n_tetrodes, badChans)
                            downsampling_settings_list = ['downsampling ' + str(downsampling), 
                                                          'lowpass_frequency ' + str(lowpass_freq)]
                            with h5py.File(fpath,'r+') as h5file:
                                h5file['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['tetrode_lowpass'] = downsampled_data
                                h5file['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['tetrode_lowpass_timestamps'] = downsampled_data_timestamps
                                asciiList = [n.encode("ascii", "ignore") for n in downsampling_settings_list]
                                h5file['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['tetrode_lowpass_info'] = h5file.create_dataset(None, (len(asciiList),),'S100', asciiList)
                                data = h5file[raw_data_path]
                                data_AUX = np.array(data[:,auxChanStart:])
                                h5file['acquisition']['timeseries'][recordingKey]['continuous'][processorKey]['data_AUX'] = data_AUX
                            print('DEL RAW: ' + fpath)
                            with h5py.File(fpath,'r+') as h5file:
                                del h5file[raw_data_path]
                            nwb_raw_deleted += 1
                            # Repack NWB file to recover space
                            path = '/acquisition/timeseries/' + recordingKey + '/continuous/' + processorKey + '/tetrode_lowpass_timestamps'
                            if check_if_path_exists(fpath, path):
                                print('Repacking file: ' + fpath)
                                os.system('h5repack ' + fpath + ' ' + (fpath + '.repacked'))
                                os.system('mv ' + (fpath + '.repacked') + ' ' + fpath)
                                nwb_repacked += 1
                        else:
                            print('No Spikes: ' + fpath)
                            nwb_noSpikes += 1
                    else:
                        print('No RAW data: ' + fpath)
                        nwb_noRawData += 1
                except IOError:
                    print('IOError in file: ' + fpath)
                    ioError += 1
            elif fname == 'experiment1_101.raw.kwd':
                print('DEL: ' + fpath)
                os.remove(fpath)
                kwd_deleted += 1
            elif fname.endswith('.continuous'):
                print('DEL: ' + fpath)
                os.remove(fpath)
                continuous_deleted += 1

print('\nProcessing output:')
print('nwb_noRawData ' + str(nwb_noRawData))
print('nwb_raw_deleted ' + str(nwb_raw_deleted))
print('nwb_repacked ' + str(nwb_repacked))
print('nwb_noSpikes ' + str(nwb_noSpikes))
print('ioError ' + str(ioError))
print('kwd_deleted ' + str(kwd_deleted))
print('continuous_deleted ' + str(continuous_deleted))
