from TrackingDataProcessing import process_tracking_data
import argparse
import numpy as np
import os
import NWBio
import createAxonaData
import HelperFunctions as hfunct
from KlustaKwikWrapper import klustakwik
import tempfile
import shutil
import copy

def extract_spikes_from_raw_data(NWBfilePath, UseChans=False, threshold=50):
    # Load data
    print('Loading NWB data for extracting spikes')
    data = NWBio.load_continuous(NWBfilePath)
    timestamps = np.array(data['timestamps'])
    if UseChans:
        continuous = np.array(data['continuous'][:, UseChans[0]:UseChans[1]])
    else:
        continuous = np.array(data['continuous'])
    continuous = -np.transpose(continuous)
    # Create and edit good channels list according to bad channels list
    goodChan = np.arange(continuous.shape[0])
    badChan = get_badChan(NWBfilePath, UseChans=UseChans)
    if badChan:
        for nchan in badChan:
            goodChan = goodChan[goodChan != nchan]
    # Remove the mean of the signal from all channels
    print('Common average referencing all channels.\n' + \
          'Make sure only data on a single drive is selected.')
    continuous_mean = np.mean(continuous[goodChan,:], axis=0, keepdims=True)
    continuous_mean = np.round(continuous_mean).astype(np.int16)
    continuous = continuous - np.repeat(continuous_mean, continuous.shape[0], axis=0)
    # Set bad channels to 0
    if badChan:
        continuous[badChan,:] = np.int16(0)
    # Filter each channel
    hfunct.print_progress(0, goodChan.size, prefix = 'Filtering raw data:', initiation=True)
    for nchan in range(goodChan.size):
        signal_in = np.float64(continuous[goodChan[nchan],:])
        signal_out = hfunct.butter_bandpass_filter(signal_in)
        continuous[goodChan[nchan],:] = np.int16(signal_out)
        hfunct.print_progress(nchan + 1, goodChan.size, prefix = 'Filtering raw data:')
    n_tetrodes = continuous.shape[0] / 4
    # Find threshold crossings on each tetrode
    threshold_int16 = np.int16(np.round(threshold / 0.195))
    spike_data = []
    for ntet in range(n_tetrodes):
        # Get tetrode number right
        if UseChans:
            nr_tetrode = ntet + hfunct.channels_tetrode(UseChans[0])
        else:
            nr_tetrode = ntet
        # Find threshold crossings for each channel
        spiketimes = np.array([], dtype=np.int64)
        for nchan in hfunct.tetrode_channels(ntet):
            tmp = continuous[nchan,:] > threshold_int16
            spiketimes = np.append(spiketimes, np.where(tmp)[0])
        if len(spiketimes) > 0: 
            spiketimes = np.sort(spiketimes)
        # Remove duplicates based on temporal proximity
        if len(spiketimes) > 0:
            tooclose = np.int64(np.round(30000 * 0.001))
            spike_diff = np.concatenate((np.array([0]),np.diff(spiketimes)))
            tooclose_idx = spike_diff < tooclose
            spiketimes = np.delete(spiketimes, np.where(tooclose_idx)[0])
        if len(spiketimes) > 0:
            # Using spiketimes create an array of indices (windows) to extract waveforms from LFP trace
            winsize_before = 10
            winsize_after = 20
            wintotal = winsize_before + winsize_after + 1 # This should be 31 for all downstream functions to work
            spiketimes = np.expand_dims(spiketimes, 1)
            # Create windows for indexing all samples for a waveform
            windows = np.arange(winsize_before + winsize_after + 1, dtype=np.int32) - winsize_before
            windows = np.tile(windows, (spiketimes.size,1))
            windows = windows + np.tile(spiketimes, (1,windows.shape[1]))
            # Skip windows that are too close to edge of signal
            tooearly = windows < 0
            toolate = windows > (continuous.shape[1] - 1)
            idx_delete = np.any(np.logical_or(tooearly, toolate), axis=1)
            windows = np.delete(windows, np.where(idx_delete)[0], axis=0)
            spiketimes = np.delete(spiketimes, np.where(idx_delete)[0], axis=0)
            # Create indexing for channels and spikes
            # windows and windows_channels shape is nchan x windowsize x nspikes
            windows = np.repeat(windows[:,:,np.newaxis], 4, axis=2)
            windows = np.swapaxes(windows,0,2)
            windows_channels = np.tile(np.arange(windows.shape[0]) + hfunct.tetrode_channels(ntet)[0], 
                                       (windows.shape[1],1))
            windows_channels = np.transpose(windows_channels)
            windows_channels = np.repeat(windows_channels[:,:,np.newaxis], windows.shape[2], axis=2)
            waveforms = continuous[windows_channels,windows]
            waveforms = np.swapaxes(waveforms,0,2)
            # Append data as dictionary to spike_data list
            spike_data.append({'waveforms': waveforms, 
                               'timestamps': timestamps[spiketimes.squeeze()],
                               'nr_tetrode': nr_tetrode})
        else:
            spike_data.append({'waveforms': np.zeros((3, 40, 4), dtype=np.int16), 
                               'timestamps': np.array([0],dtype=np.float64), 
                               'nr_tetrode': nr_tetrode})

    return spike_data

def applyKlustaKwik(waveform_data):
    hfunct.print_progress(0, len(waveform_data), prefix='Applying KlustaKwik:', suffix=' T: 0/' + str(len(waveform_data)), initiation=True)
    for ntet in range(len(waveform_data)):
        if len(waveform_data[ntet]['spiketimes']) > 1:
            # Create temporary processing folder
            KlustaKwikProcessingFolder = tempfile.mkdtemp('KlustaKwikProcessing')
            # Prepare input to KlustaKwik
            waves = np.swapaxes(waveform_data[ntet]['waveforms'],1,2)
            features2use = ['PC1', 'PC2', 'PC3', 'Amp', 'Vt']
            d = {0: features2use}
            klustakwik(waves, d, os.path.join(KlustaKwikProcessingFolder, 'KlustaKwikTemp'))
            # Read in cluster IDs
            cluFileName = os.path.join(KlustaKwikProcessingFolder, 'KlustaKwikTemp.clu.0')
            with open(cluFileName, 'rb') as file:
                lines = file.readlines()
            clusterIDs = []
            for line in lines:
                clusterIDs.append(int(line.rstrip()))
            clusterIDs = clusterIDs[1:] # Drop the first value which is number of spikes
            waveform_data[ntet]['clusterIDs'] = np.array(clusterIDs, dtype=np.int16)
            # Delete KlustaKwik temporary processing folder
            shutil.rmtree(KlustaKwikProcessingFolder)
        else:
            print('No spikes on tetrode ' + str(ntet + 1))
            waveform_data[ntet]['clusterIDs'] = np.ones(waveform_data[ntet]['spiketimes'].shape, dtype=np.int16)
        hfunct.print_progress(ntet + 1, len(waveform_data), prefix='Applying KlustaKwik:', suffix=' T: ' + str(ntet + 1) + '/' + str(len(waveform_data)))

    return waveform_data

def filter_spike_data(spike_data, pos_edges, threshold, noise_cut_off):
    '''
    Filters data on all tetrodes according to multiple criteria
    Inputs:
        spike_data [list] - list of dictionaries with 'waveforms' and 'timestamps' for each tetrode
        pos_edges [list] - first and last position data timestamp
        threshold - threshold value in microvolts
        noise_cut_off - value for removing spikes above this cut off in microvolts
    '''
    # Create idx_keep all True for all tetrodes. Then turn values to False with each filter
    idx_keep = []
    for ntet in range(len(spike_data)):
        idx_keep.append(np.ones(spike_data[ntet]['timestamps'].shape, dtype=np.bool))
    # Include spikes occured during position data
    for ntet in range(len(spike_data)):
        idx = np.logical_and(spike_data[ntet]['timestamps'] > pos_edges[0], 
                             spike_data[ntet]['timestamps'] < pos_edges[1])
        idx_keep[ntet] = np.logical_and(idx_keep[ntet], idx)
    # Include spikes above threshold
    threshold_int16 = np.int16(np.round(threshold / 0.195))
    for ntet in range(len(spike_data)):
        idx = np.any(np.any(np.abs(spike_data[ntet]['waveforms']) > threshold_int16, axis=2), axis=1)
        idx_keep[ntet] = np.logical_and(idx_keep[ntet], idx)
        if np.sum(idx) < idx.size:
            percentage_above_threshold = np.sum(idx) / float(idx.size) * 100
            print('{:.1f}% of spikes on tetrode {} were above threshold'.format(percentage_above_threshold, ntet + 1))
    # Include spikes below noise cut off
    if noise_cut_off and (noise_cut_off != 0):
        noise_cut_off_int16 = np.int16(np.round(noise_cut_off / 0.195))
        for ntet in range(len(spike_data)):
            idx = np.all(np.all(np.abs(spike_data[ntet]['waveforms']) < noise_cut_off_int16, axis=2), axis=1)
            idx_keep[ntet] = np.logical_and(idx_keep[ntet], idx)
            if np.sum(idx) < idx.size:
                percentage_too_big = (1 - np.sum(idx) / float(idx.size)) * 100
                print('{:.1f}% of spikes removed on tetrode {}'.format(percentage_too_big,ntet+1))

    return idx_keep

def createWaveformDict(OpenEphysDataPath, UseChans=False, UseRaw=False, noise_cut_off=500, threshold=50):
    '''
    This function organises NWB data into dictonaries
    '''
    # Get thresholded spike data for each tetrode
    print('Loading spikes')
    spike_data = NWBio.load_spikes(OpenEphysDataPath, use_badChan=True)
    useTet = np.arange(len(spike_data), dtype=np.int16)
    if len(spike_data) > 0 and not UseRaw:
        # Limit analysis to specific tetrodes if channels are specified
        if UseChans:
            firstTet = hfunct.channels_tetrode(UseChans[0])
            lastTet = hfunct.channels_tetrode(UseChans[1] - 1)
            spike_data = spike_data[firstTet:lastTet + 1]
            useTet = np.arange(lastTet - firstTet, dtype=np.int16) + int(firstTet)
        # Fully load into memory
        for ntet in range(len(spike_data)):
            if spike_data[ntet]['waveforms'].shape[0] > 0:
                spike_data[ntet]['waveforms'] = np.swapaxes(np.array(spike_data[ntet]['waveforms']),1,2)
            else:
                # If no spikes detected, create one flat spike waveform at timempoint 1
                spike_data[ntet]['waveforms'] = np.zeros((3, 40, 4), dtype=np.int16)
                spike_data[ntet]['timestamps'] = np.array([0],dtype=np.float64)
        # Invert waveforms
        for ntet in range(len(spike_data)):
            spike_data[ntet]['waveforms'] = -spike_data[ntet]['waveforms']
    else:
        print('Extracting spikes from raw data.')
        spike_data = extract_spikes_from_raw_data(OpenEphysDataPath, UseChans, threshold)
        # Save extracted spikes in recording file
        for ntet in range(len(spike_data)):
            NWBio.save_spikes(OpenEphysDataPath, spike_data[ntet]['nr_tetrode'], 
                              spike_data[ntet]['waveforms'], spike_data[ntet]['timestamps'])
        print('Spikes saved to ' + OpenEphysDataPath)
    # Find eligible spikes on all tetrodes
    pos_edges = NWBio.get_processed_tracking_data_timestamp_edges(OpenEphysDataPath)
    idx_keep = filter_spike_data(spike_data, pos_edges, threshold, noise_cut_off)
    # Arrange data into a list of dictionaries for each tetrode
    waveform_data = []
    for ntet in range(len(spike_data)):
        if np.sum(idx_keep[ntet]) == 0:
            # If there are no spikes on a tetrode, create one zero spike 1 second after first position sample
            waveforms = np.zeros((1,40,4), dtype=np.int16)
            spiketimes = np.array([pos_edges[0] + 1], dtype=np.float64)
        else:
            waveforms = spike_data[ntet]['waveforms'][idx_keep[ntet],:40,:]
            spiketimes = spike_data[ntet]['timestamps'][idx_keep[ntet]]
        # Combine all information into dictionary
        waveform_data.append({'idx_keep': idx_keep[ntet], 
                              'waveforms': waveforms, 
                              'spiketimes': spiketimes, 
                              'nr_tetrode': spike_data[ntet]['nr_tetrode']})

    return waveform_data

def make_UseChans_from_channel_map(channel_map):
    channels = []
    for key in channel_map.keys():
        channels.append(channel_map[key]['list'])
    channels = np.sort(np.concatenate(channels))
    # Make sure channel map is as expected
    if int(channels[-1] + 1) - int(channels[0]) != np.unique(channels).size:
        raise ValueError('Channel Map is discontinuous or overlapping between areas.')
    # Create UseChans that covers the whole range of channels
    UseChans = [channels[0], channels[-1] + 1]

    return UseChans

def get_badChan(OpenEphysDataPaths, UseChans=False):
    if isinstance(OpenEphysDataPaths, basestring):
        OpenEphysDataPaths = [OpenEphysDataPaths]
    # Get bad channels for all datasets
    badChans = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        badChans.append(NWBio.listBadChannels(OpenEphysDataPath))
    # Ensure that the badChan values are the same in all datasets
    if not badChans[1:] == badChans[:-1]:
        raise ValueError('badChan is not the same in all datasets.')
    # Then use the common badChan list
    badChan = badChans[0]
    print('Ignoring bad channels: ' + str(list(np.array(badChan) + 1)))
    if UseChans:
        # Convert bad channel number for later use depending on used channels
        badChan = np.array(badChan)
        badChan = badChan[badChan >= np.array(UseChans[0], dtype=np.int16)]
        badChan = badChan - np.array(UseChans[0], dtype=np.int16)
        badChan = badChan[badChan < UseChans[1] - UseChans[0]]
        badChan = list(badChan)

    return badChan

def main(OpenEphysDataPaths, UseChans=False, UseRaw=False, noise_cut_off=500, threshold=50):
    if isinstance(OpenEphysDataPaths, basestring):
        OpenEphysDataPaths = [OpenEphysDataPaths]
    # If use chans not specified, get channel list from Genera settgins Channel Map, if available
    if not UseChans and NWBio.check_if_settings_available(OpenEphysDataPaths[0],'/General/channel_map/'):
        if len(OpenEphysDataPaths) > 1:
            print('Using Channel Map from the first dataset on the list.')
        channel_map = NWBio.load_settings(OpenEphysDataPaths[0],'/General/channel_map/')
        UseChans = make_UseChans_from_channel_map(channel_map)
    # Create ProcessedPos if any raw tracking data available
    for OpenEphysDataPath in OpenEphysDataPaths:
        if not NWBio.check_if_processed_position_data_available(OpenEphysDataPath):
            if NWBio.check_if_tracking_data_available(OpenEphysDataPath):
                print('Processing tracking data...')
                ProcessedPos = process_tracking_data(OpenEphysDataPath)
                NWBio.save_tracking_data(OpenEphysDataPath, ProcessedPos, ProcessedPos=True, ReProcess=False)
                print('ProcessedPos saved to ' + OpenEphysDataPath)
            elif NWBio.check_if_binary_pos(OpenEphysDataPath):
                NWBio.use_binary_pos(OpenEphysDataPath, postprocess=False)
                print('Using binary position data')
    # Extract spikes for each tetrode in each recording into a dictonary
    waveform_datas = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        print('Extracting spikes from: ' + OpenEphysDataPath)
        waveform_data = createWaveformDict(OpenEphysDataPath, UseChans=UseChans, UseRaw=UseRaw, 
                                           noise_cut_off=noise_cut_off, threshold=threshold)
        # Add this dictonary to list of all dictionaries
        waveform_datas.append(waveform_data)
    # Overwrite idx_keep in recording files
    for OpenEphysDataPath, waveform_data in zip(OpenEphysDataPaths, waveform_datas):
        for ntet in range(len(waveform_data)):
            NWBio.save_tetrode_idx_keep(OpenEphysDataPath, waveform_data[ntet]['nr_tetrode'], 
                                        waveform_data[ntet]['idx_keep'], overwrite=True)
    # Apply Klustakwik on each tetrode
    if len(waveform_datas) == 1:
        waveform_datas[0] = applyKlustaKwik(waveform_datas[0])
    else:
        print('Combining recordings')
        # Combine all waveform_datas into a single waveform_data
        waveform_datas_comb = copy.deepcopy(waveform_datas[0])
        for tmp_waveform_data in waveform_datas[1:]:
            for ntet in range(len(tmp_waveform_data)):
                waveform_datas_comb[ntet]['waveforms'] = np.append(waveform_datas_comb[ntet]['waveforms'], 
                                                                   tmp_waveform_data[ntet]['waveforms'], axis=0)
                waveform_datas_comb[ntet]['spiketimes'] = np.append(waveform_datas_comb[ntet]['spiketimes'], 
                                                                    tmp_waveform_data[ntet]['spiketimes'], axis=0)   
        # Apply KlustaKwik on the combined waveform_data
        waveform_datas_comb = applyKlustaKwik(waveform_datas_comb)
        # Extract clusters for each original waveform_data
        for ndata in range(len(waveform_datas)):
            for ntet in range(len(waveform_datas[ndata])):
                # Get clusterIDs for this dataset and tetrode
                nspikes = len(waveform_datas[ndata][ntet]['spiketimes'])
                clusterIDs = waveform_datas_comb[ntet]['clusterIDs'][range(nspikes)]
                waveform_datas[ndata][ntet]['clusterIDs'] = clusterIDs
                # Remove these clusterIDs from the list
                waveform_datas_comb[ntet]['clusterIDs'] = np.delete(waveform_datas_comb[ntet]['clusterIDs'], range(nspikes), axis=0)
    # Overwrite clusterIDs in recording files
    for OpenEphysDataPath, waveform_data in zip(OpenEphysDataPaths, waveform_datas):
        for ntet in range(len(waveform_data)):
            NWBio.save_tetrode_clusterIDs(OpenEphysDataPath, waveform_data[ntet]['nr_tetrode'], 
                                          waveform_data[ntet]['clusterIDs'], overwrite=True)
    for OpenEphysDataPath, waveform_data in zip(OpenEphysDataPaths, waveform_datas):
        # Define Axona data subfolder name based on specific channels if requested
        if UseChans:
            subfolder = 'AxonaData_' + str(UseChans[0] + 1) + '-' + str(UseChans[1])
        else:
            subfolder = 'AxonaData'
        # Create Axona data
        createAxonaData.createAxonaData(OpenEphysDataPath, waveform_data, 
                                        subfolder=subfolder, eegChan=1)

if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Apply KlustaKwik and export into Axona format.')
    parser.add_argument('paths', type=str, nargs='*', 
                        help='recording data folder(s) (can enter multiple paths separated by spaces to KlustaKwik simultaneously)')
    parser.add_argument('--chan', type=int, nargs = 2, 
                        help='list the first and last channel to process (counting starts from 1)')
    parser.add_argument('--noisecut', type=int, nargs = 1, 
                        help='enter 0 to skip or value in microvolts for noise cutoff (default is 500)')
    parser.add_argument('--threshold', type=int, nargs = 1, 
                        help='enter spike threshold in microvolts (default is 50)')
    parser.add_argument('--useraw', action='store_true',
                        help='extract spikes from raw continuous data')
    args = parser.parse_args()
    # Get paths to recording files
    OpenEphysDataPaths = args.paths
    # Get UseChans input variable
    if args.chan:
        UseChans = [args.chan[0] - 1, args.chan[1]]
        if np.mod(UseChans[1] - UseChans[0], 4) != 0:
            raise ValueError('Channel range must cover full tetrodes')
    else:
        UseChans = False
    if args.noisecut:
        noise_cut_off = args.noisecut[0]
    else:
        noise_cut_off = 500
    if args.threshold:
        threshold = args.threshold[0]
    else:
        threshold = 50
    # Run the script
    main(OpenEphysDataPaths, UseChans, args.useraw, noise_cut_off, threshold)
