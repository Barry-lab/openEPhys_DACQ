import matlab.engine
# To install matlab engine, go to folder /usr/local/MATLAB/R2017a/extern/engines/python
# and run terminal command: sudo python setup.py install
import sys
import numpy as np
import os
import NWBio
import CombineTrackingData
import createAxonaData
import argparse
import tempfile
import shutil
import HelperFunctions as hfunct

def createWaveformDict(OpenEphysDataPath,KiloSortOutputPath,UseChans=False,badChan=[]):
    # This function uses KiloSort output to extract waveforms from NWB file
    # UseChans = [0,64] will limit finding waveforms to channels 1 to 64
    # UseChans = [64,128] will limit finding waveforms to channels 65 to 128
    # Set waveform size in datapoints
    winsize_before = 10
    winsize_after = 20
    wintotal = winsize_before + winsize_after + 1 # This should be 31 for all downstream functions to work
    NWBfilePath = os.path.join(OpenEphysDataPath,'experiment_1.nwb')
    # Load data
    clusters = np.load(os.path.join(KiloSortOutputPath, 'spike_clusters.npy'))[:,0]
    spiketimes = np.load(os.path.join(KiloSortOutputPath, 'spike_times.npy'))[:,0]
    print('Loading NWB data for extracting spikes')
    data = NWBio.load_continuous(NWBfilePath)
    timestamps = np.array(data['timestamps'])
    continuous = -np.transpose(np.array(data['continuous']))
    if UseChans:
        continuous = continuous[UseChans[0]:UseChans[1],:]
    # Create and edit good channels list according to bad channels list
    goodChan = np.arange(continuous.shape[0])
    if badChan:
        for nchan in badChan:
            goodChan = goodChan[goodChan != nchan]
    # Remove the mean of the signal from all channels
    print('Common average referencing all channels')
    continuous_mean = np.mean(continuous[goodChan,:], axis=0, keepdims=True)
    continuous = continuous - np.repeat(continuous_mean, continuous.shape[0], axis=0)
    # Set bad channels to 0
    if badChan:
        continuous[badChan,:] = np.int16(0)
    # Filter each channel
    hfunct.print_progress(0, goodChan.size, prefix = 'Filtering raw data:', initiation=True)
    for nchan in range(goodChan.size):
        signal_in = np.float64(continuous[goodChan[nchan],:])
        signal_out = hfunct.Filter(signal_in)
        continuous[goodChan[nchan],:] = np.int16(signal_out)
        hfunct.print_progress(nchan + 1, goodChan.size, prefix = 'Filtering raw data:')
    n_tetrodes = continuous.shape[0] / 4
    cluster_nrs = np.unique(clusters)
    waveform_data = []
    for ntet in range(n_tetrodes):
        tmp = {'waveforms':np.zeros((1,wintotal,4),dtype=np.int16), 
               'spiketimes':np.array([100],dtype=np.int64), 
               'clusterIDs':np.array([1],dtype=np.int16)}
        waveform_data.append(tmp)
    hfunct.print_progress(0, len(cluster_nrs), prefix = 'Loading waveforms from NWB:', initiation=True)
    progress_count = 0
    for nclu in list(cluster_nrs):
        stimes = spiketimes[clusters == nclu]
        clu_timestamps = timestamps[stimes]
        # Create windows for indexing all samples for a waveform
        stimes = np.int32(np.expand_dims(stimes, 1))
        windows = np.arange(winsize_before + winsize_after + 1, dtype=np.int32) - winsize_before
        windows = np.tile(windows, (stimes.size,1))
        windows = windows + np.tile(stimes, (1,windows.shape[1]))
        # Skip windows that are too close to edge of signal
        tooearly = windows < 0
        toolate = windows > (continuous.shape[1] - 1)
        idx_delete = np.any(np.logical_or(tooearly, toolate), axis=1)
        windows = np.delete(windows, np.where(idx_delete)[0], axis=0)
        clu_timestamps = np.delete(clu_timestamps, np.where(idx_delete)[0], axis=0)
        # Create indexing for all channels and spikes
        # windows and windows_channels shape is nchan x windowsize x nspikes
        windows = np.repeat(windows[:,:,np.newaxis], n_tetrodes * 4, axis=2)
        windows = np.swapaxes(windows,0,2)
        windows_channels = np.tile(np.arange(windows.shape[0]), (windows.shape[1],1))
        windows_channels = np.transpose(windows_channels)
        windows_channels = np.repeat(windows_channels[:,:,np.newaxis], windows.shape[2], axis=2)
        # Compute mean spike for all channels
        tmp_waveforms = continuous[windows_channels,windows]
        # # Zero all tmp_waveforms
        mean_waveforms = np.mean(tmp_waveforms, axis=2)
        # Find channel and tetrode with highest amplitude
        wave_peaks = np.ptp(mean_waveforms, axis=1)
        nchan = np.argmax(wave_peaks)
        ntet = hfunct.channels_tetrode(nchan)
        # Extract waveforms of all spikes on all channels of this tetrode
        waveforms = continuous[windows_channels[hfunct.tetrode_channels(ntet),:,:], 
                               windows[hfunct.tetrode_channels(ntet),:,:]]
        waveforms = np.swapaxes(waveforms,0,2)
        # Remove spikes outside position data
        pos_edges = hfunct.get_position_data_edges(OpenEphysDataPath)
        idx_outside_pos_data = clu_timestamps < pos_edges[0]
        idx_outside_pos_data = np.logical_or(idx_outside_pos_data, clu_timestamps > pos_edges[1])
        idx_outside_pos_data = np.where(idx_outside_pos_data)[0]
        waveforms = np.delete(waveforms, idx_outside_pos_data, 0)
        clu_timestamps = np.delete(clu_timestamps, idx_outside_pos_data)
        # Store data for the corresponding tetrode
        waveform_data[ntet]['waveforms'] = np.append(waveform_data[ntet]['waveforms'],waveforms,axis=0)
        waveform_data[ntet]['spiketimes'] = np.append(waveform_data[ntet]['spiketimes'],clu_timestamps)
        waveform_data[ntet]['clusterIDs'] = np.append(waveform_data[ntet]['clusterIDs'],nclu * np.ones(clu_timestamps.size,dtype=np.int16))
        progress_count += 1
        hfunct.print_progress(progress_count, len(cluster_nrs), prefix = 'Loading waveforms from NWB:')
    # Correct spike order for each tetrode and remove placeholder if spikes detected
    for ntet in range(n_tetrodes):
        if waveform_data[ntet]['spiketimes'].size > 1:
            waveform_data[ntet]['waveforms'] = waveform_data[ntet]['waveforms'][1:,:,:]
            waveform_data[ntet]['spiketimes'] = waveform_data[ntet]['spiketimes'][1:]
            waveform_data[ntet]['clusterIDs'] = waveform_data[ntet]['clusterIDs'][1:]
            idx = waveform_data[ntet]['spiketimes'].argsort()
            waveform_data[ntet]['waveforms'] = waveform_data[ntet]['waveforms'][idx,:,:]
            waveform_data[ntet]['spiketimes'] = waveform_data[ntet]['spiketimes'][idx]
            waveform_data[ntet]['clusterIDs'] = waveform_data[ntet]['clusterIDs'][idx]
    # Set cluster values on each tetrode to start from 1
    for ntet in range(n_tetrodes):
        if waveform_data[ntet]['spiketimes'].size > 1:
            originalClus = np.unique(waveform_data[ntet]['clusterIDs'])
            for nclu in range(originalClus.size):
                idx = waveform_data[ntet]['clusterIDs'] == originalClus[nclu]
                waveform_data[ntet]['clusterIDs'][idx] = nclu + 1
    # Put additional variables into the dictionary for each tetrode
    for ntet in range(n_tetrodes):
        nchan = hfunct.tetrode_channels(ntet)[0]
        if UseChans:
            nchan = nchan + UseChans[0]
        waveform_data[ntet]['nr_tetrode'] = hfunct.channels_tetrode(nchan)
        waveform_data[ntet]['tetrode_channels'] = hfunct.tetrode_channels(hfunct.channels_tetrode(nchan))
        waveform_data[ntet]['badChan'] = hfunct.listBadChannels(OpenEphysDataPath)
        waveform_data[ntet]['bitVolts'] = float(0.195)

    return waveform_data


def main(OpenEphysDataPath, UseChans=False, keepKiloSortOutput=False):
    # Assume NWB file has name experiment_1.nwb
    NWBfilePath = os.path.join(OpenEphysDataPath,'experiment_1.nwb')
    if keepKiloSortOutput:
        # Set default location for KiloSortProcessing Folder based on specific channels if requested
        if UseChans:
            KiloSortProcessingFolder = 'KiloSortProcess_' + str(UseChans[0] + 1) + '-' + str(UseChans[1])
        else:
            KiloSortProcessingFolder = 'KiloSortProcess'
        KiloSortProcessingFolder = os.path.join(OpenEphysDataPath,KiloSortProcessingFolder)
        if not os.path.isdir(KiloSortProcessingFolder):
            os.mkdir(KiloSortProcessingFolder)
    else:
        KiloSortProcessingFolder = tempfile.mkdtemp('KiloSortProcessing')
    KiloSortBinaryFileName = 'experiment_1.dat'
    # Load file, if promted only specific channel range
    print('Loading NWB data for feeding into KiloSort')
    if UseChans:
        data = np.array(NWBio.load_continuous(NWBfilePath)['continuous'][:,UseChans[0]:UseChans[1]])
    else:
        data = np.array(NWBio.load_continuous(NWBfilePath)['continuous'])
    # Get bad channels and renumber according to used channels
    badChan = hfunct.listBadChannels(OpenEphysDataPath)
    if badChan:
        print('Ignoring bad channels: ' + str(list(np.array(badChan) + 1)))
        badChan = np.array(badChan, dtype=np.int16)
        if UseChans:
            # Correct bad channel number depending on used channels
            badChan = badChan[badChan >= np.array(UseChans[0], dtype=np.int16)]
            badChan = badChan - np.array(UseChans[0], dtype=np.int16)
            badChan = badChan[badChan < data.shape[1]]
        badChan = list(badChan)
    # Remove bad channels from data sent to KiloSort
    if badChan:
        data = np.delete(data, badChan, axis=1)
    # Write binary file for KiloSort
    print('Writing NWB into binary')
    data.tofile(os.path.join(KiloSortProcessingFolder,KiloSortBinaryFileName))
    # Run KiloSort
    eng = matlab.engine.start_matlab()
    eng.cd('KiloSortScripts')
    matlab_badChan = matlab.double(badChan)
    eng.master_file(float(data.shape[1]), matlab_badChan, 
                    KiloSortProcessingFolder, KiloSortBinaryFileName, nargout=0)
    # Make sure position data is available
    if not os.path.exists(os.path.join(OpenEphysDataPath,'PosLogComb.csv')):
        if NWBio.check_if_binary_pos(NWBfilePath):
            _ = NWBio.load_pos(NWBfilePath, savecsv=True, postprocess=True)
        else:
            CombineTrackingData.combdata(NWBfilePath)
    # Extract waveforms from NWB data based on KiloSort spiketimes and clusters
    waveform_data = createWaveformDict(OpenEphysDataPath, KiloSortProcessingFolder, 
                                       UseChans=UseChans, badChan=badChan)
    # Define Axona data subfolder name based on specific channels if requested
    if UseChans:
        subfolder = 'AxonaData_' + str(UseChans[0] + 1) + '-' + str(UseChans[1])
    else:
        subfolder = 'AxonaData'
    # Create Axona data
    createAxonaData.createAxonaData(OpenEphysDataPath, waveform_data, 
                                    subfolder=subfolder, eegChan=1)
    # Delete KiloSort data unless asked to keep it
    if not keepKiloSortOutput:
        shutil.rmtree(KiloSortProcessingFolder)


if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Apply KiloSort and export into Axona format.')
    parser.add_argument('path', type=str,
                        help='recording data folder')
    parser.add_argument('--chan', type=int, nargs = 2, 
                        help='list the first and last channel to process (counting starts from 1)')
    parser.add_argument('--keep', action='store_true',
                        help='store KiloSort output to recording data folder')
    args = parser.parse_args()
    # Assign input arguments to variables
    OpenEphysDataPath = args.path
    keepKiloSortOutput = args.keep
    if args.chan:
        UseChans = [args.chan[0] - 1, args.chan[1]]
        if np.mod(UseChans[1] - UseChans[0], 32) != 0:
            raise ValueError('Total number of channels must be a multiple of 32')
    else:
        UseChans = False
    # Launch script
    main(OpenEphysDataPath, UseChans, keepKiloSortOutput)