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

def extract_spikes_from_raw_data(NWBfilePath, UseChans, tetrode_nrs=None, threshold=50):
    '''
    This function mimicks the OpenEphysGUI spike extraction
    Common Average Referencing -> Bandpass Filter -> Thresholding
    Output:
        Dictonary with 'waveforms' - nspike x nchan x windowsize
                                     with same polarity as from OpenEphysGUI
                       'timestamps' - for each spike
                       'nr_tetrode' - the number of extracted tetrode from whole dataset
                                      this corrects for UseChans
    '''
    # Only extract data based on requested tetrodes
    if tetrode_nrs is None:
        tetrode_nrs = get_tetrode_nrs(UseChans)
    # Load data
    print('Loading NWB data for extracting spikes')
    data = NWBio.load_continuous(NWBfilePath)
    chan_nrs = list(np.arange(data['continuous'].shape[1])[UseChans[0]:UseChans[1]])
    timestamps = np.array(data['timestamps'])
    continuous = np.array(data['continuous'][:, UseChans[0]:UseChans[1]])
    continuous = np.transpose(continuous)
    # Create and edit good channels list according to bad channels list
    badChan = NWBio.listBadChannels(NWBfilePath)
    goodChan = range(len(chan_nrs))
    if badChan:
        for nchan in goodChan:
            if chan_nrs[nchan] in badChan:
                goodChan.delete(nchan)
    # Remove the mean of the signal from all channels
    print('Common average referencing all channels.\n' + \
          'Make sure only data on a single drive is selected.')
    continuous_mean = np.mean(continuous[goodChan,:], axis=0, keepdims=True)
    continuous_mean = np.round(continuous_mean).astype(np.int16)
    continuous = continuous - np.repeat(continuous_mean, continuous.shape[0], axis=0)
    # Set bad channels to 0
    if badChan:
        for nchanBad in badChan:
            nchan = chan_nrs.index(nchanBad)
        continuous[nchan,:] = np.int16(0)
    # Filter each channel
    chanToFilter = [nchan for nchan in goodChan if hfunct.channels_tetrode(chan_nrs[nchan]) in tetrode_nrs]
    hfunct.print_progress(0, len(chanToFilter), prefix = 'Filtering raw data:', initiation=True)
    for nchan in range(len(chanToFilter)):
        signal_in = np.float64(continuous[chanToFilter[nchan],:])
        signal_out = hfunct.butter_bandpass_filter(signal_in)
        continuous[chanToFilter[nchan],:] = np.int16(signal_out)
        hfunct.print_progress(nchan + 1, len(chanToFilter), prefix = 'Filtering raw data:')
    # Find threshold crossings on each tetrode
    threshold_int16 = np.int16(np.round(threshold / 0.195))
    spike_data = []
    for nr_tetrode in tetrode_nrs:
        # Find threshold crossings for each channel
        spike_indices = np.array([], dtype=np.int64)
        for chan_nr in hfunct.tetrode_channels(nr_tetrode):
            tmp = continuous[chan_nrs.index(chan_nr),:] < -threshold_int16
            spike_indices = np.append(spike_indices, np.where(tmp)[0])
        if len(spike_indices) > 0: 
            spike_indices = np.sort(spike_indices)
        # Remove duplicates based on temporal proximity
        if len(spike_indices) > 0:
            tooclose = np.int64(np.round(30000 * 0.001))
            spike_diff = np.concatenate((np.array([0]),np.diff(spike_indices)))
            tooclose_idx = spike_diff < tooclose
            spike_indices = np.delete(spike_indices, np.where(tooclose_idx)[0])
        if len(spike_indices) > 0:
            # Using spike_indices create an array of indices (windows) to extract waveforms from LFP trace
            # The following values are chosen to match OpenEphysGUI default window
            winsize_before = 6
            winsize_after = 34
            spike_indices = np.expand_dims(spike_indices, 1)
            # Create windows for indexing all samples for a waveform
            windows = np.arange(winsize_before + winsize_after, dtype=np.int32) - winsize_before
            windows = np.tile(windows, (spike_indices.size,1))
            windows = windows + np.tile(spike_indices, (1,windows.shape[1]))
            # Skip windows that are too close to edge of signal
            tooearly = windows < 0
            toolate = windows > (continuous.shape[1] - 1)
            idx_delete = np.any(np.logical_or(tooearly, toolate), axis=1)
            windows = np.delete(windows, np.where(idx_delete)[0], axis=0)
            spike_indices = np.delete(spike_indices, np.where(idx_delete)[0], axis=0)
            # Create indexing for channels and spikes
            # windows and windows_channels shape must be  nspikes x nchan x windowsize
            windows = np.repeat(windows[:,:,np.newaxis], 4, axis=2)
            windows = np.swapaxes(windows,1,2)
            windows_channels = [chan_nrs.index(nchan) for nchan in hfunct.tetrode_channels(nr_tetrode)]
            windows_channels = np.array(windows_channels)
            windows_channels = np.tile(windows_channels[np.newaxis,:,np.newaxis], 
                                       (windows.shape[0], 1, windows.shape[2]))
            waveforms = continuous[windows_channels,windows]
            # Append data as dictionary to spike_data list
            spike_data.append({'waveforms': waveforms, 
                               'timestamps': timestamps[spike_indices.squeeze()],
                               'nr_tetrode': nr_tetrode})
        else:
            spike_data.append({'waveforms': np.zeros((3, 40, 4), dtype=np.int16), 
                               'timestamps': np.array([0],dtype=np.float64), 
                               'nr_tetrode': nr_tetrode})

    return spike_data

def applyKlustaKwik(waveform_data):
    hfunct.print_progress(0, len(waveform_data), prefix='Applying KlustaKwik:', suffix=' T: 0/' + str(len(waveform_data)), initiation=True)
    for ntet in range(len(waveform_data)):
        if len(waveform_data[ntet]['timestamps']) > 1:
            # Create temporary processing folder
            KlustaKwikProcessingFolder = tempfile.mkdtemp('KlustaKwikProcessing')
            # Prepare input to KlustaKwik
            waves = waveform_data[ntet]['waveforms']
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
            waveform_data[ntet]['clusterIDs'] = np.ones(waveform_data[ntet]['timestamps'].shape, dtype=np.int16)
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
        idx = np.any(np.any(spike_data[ntet]['waveforms'] < -threshold_int16, axis=2), axis=1)
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

def get_tetrode_nrs(UseChans):
    firstTet = hfunct.channels_tetrode(UseChans[0])
    lastTet = hfunct.channels_tetrode(UseChans[1])
    tetrode_nrs = list(np.arange(lastTet - firstTet, dtype=np.int16) + int(firstTet))

    return tetrode_nrs

def createWaveformDict(OpenEphysDataPath, UseChans, UseRaw=False, noise_cut_off=500, threshold=50):
    '''
    This function organises NWB data into dictonaries
    '''
    # Limit analysis to specific tetrodes channels
    tetrode_nrs = get_tetrode_nrs(UseChans)
    # Get thresholded spike data for each tetrode
    if not UseRaw:
        print('Loading spikes')
        spike_data = NWBio.load_spikes(OpenEphysDataPath, tetrode_nrs=tetrode_nrs, use_badChan=True)
    if len(spike_data) > 0 and not UseRaw:
        # Check which tetrodes have data missing in the recording
        tetrodes_missing_in_spike_data = tetrode_nrs
        for data in spike_data:
            if 'waveforms' in data.keys():
                tetrodes_missing_in_spike_data.remove(data['nr_tetrode'])
    else:
        spike_data = [] * len(tetrode_nrs)
        tetrodes_missing_in_spike_data = tetrode_nrs
    if len(tetrodes_missing_in_spike_data) > 0:
        print('Extracting spikes from raw data for tetrodes: ' + str(tetrodes_missing_in_spike_data))
        extracted_spike_data = extract_spikes_from_raw_data(OpenEphysDataPath, UseChans, 
                                                            tetrode_nrs=tetrodes_missing_in_spike_data, 
                                                            threshold=threshold)
        # Combine extracted spike data to list of tetrode spike datas
        for data in extracted_spike_data:
            spike_data[tetrode_nrs.index(data['nr_tetrode'])] = data
        # Save extracted spikes in recording file
        for data in extracted_spike_data:
            NWBio.save_spikes(OpenEphysDataPath, data['nr_tetrode'], 
                              data['waveforms'], data['timestamps'])
        print('Spikes saved to ' + OpenEphysDataPath)
    # Find eligible spikes on all tetrodes
    pos_edges = NWBio.get_processed_tracking_data_timestamp_edges(OpenEphysDataPath)
    idx_keep = filter_spike_data(spike_data, pos_edges, threshold, noise_cut_off)
    # Overwrite idx_keep in recording files
    for ntet in range(len(spike_data)):
        NWBio.save_tetrode_idx_keep(OpenEphysDataPath, spike_data[ntet]['nr_tetrode'], 
                                    idx_keep[ntet], overwrite=True)
    # Arrange data into a list of dictionaries for each tetrode
    waveform_data = []
    for ntet in range(len(spike_data)):
        if np.sum(idx_keep[ntet]) == 0:
            # If there are no spikes on a tetrode, create one zero spike 1 second after first position sample
            waveforms = NWBio.empty_spike_data()['waveforms']
            timestamps = NWBio.empty_spike_data()['timestamps']
        else:
            waveforms = spike_data[ntet]['waveforms'][idx_keep[ntet],:40,:]
            timestamps = spike_data[ntet]['timestamps'][idx_keep[ntet]]
        # Combine all information into dictionary
        waveform_data.append({'idx_keep': idx_keep[ntet], 
                              'waveforms': waveforms, 
                              'timestamps': timestamps, 
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
    print(UseChans)
    error

    return UseChans

def get_badChan(OpenEphysDataPaths, UseChans):
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
    # If directories entered as paths, attempt creating path to file by appending experiment_1.nwb
    for ndata, OpenEphysDataPath in enumerate(OpenEphysDataPaths):
        if not os.path.isfile(OpenEphysDataPath):
            new_path = os.path.join(OpenEphysDataPath, 'experiment_1.nwb')
            if os.path.isfile(new_path):
                OpenEphysDataPaths[ndata] = new_path
            else:
                raise ValueError('The following path does not lead to a NWB data file:\n' + OpenEphysDataPath)
    # If use chans not specified, get channel list from Genera settgins Channel Map, if available
    if not UseChans:
        if NWBio.check_if_settings_available(OpenEphysDataPaths[0],'/General/channel_map/'):
            if len(OpenEphysDataPaths) > 1:
                print('Using Channel Map from the first dataset on the list.')
            channel_map = NWBio.load_settings(OpenEphysDataPaths[0],'/General/channel_map/')
            UseChans = make_UseChans_from_channel_map(channel_map)
        else:
            raise ValueError('Must specify channels to be used in channel map or function call.')
    # Create ProcessedPos if any raw tracking data available
    for OpenEphysDataPath in OpenEphysDataPaths:
        if not NWBio.check_if_processed_position_data_available(OpenEphysDataPath):
            if NWBio.check_if_tracking_data_available(OpenEphysDataPath):
                print('Processing tracking data...')
                ProcessedPos = process_tracking_data(OpenEphysDataPath)
                NWBio.save_tracking_data(OpenEphysDataPath, ProcessedPos, ProcessedPos=True, ReProcess=False)
                print('ProcessedPos saved to ' + OpenEphysDataPath)
            elif NWBio.check_if_binary_pos(OpenEphysDataPath):
                NWBio.use_binary_pos(OpenEphysDataPath, postprocess=True)
                print('Using binary position data')
    # Extract spikes for each tetrode in each recording into a dictonary
    waveform_datas = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        print('Extracting spikes from: ' + OpenEphysDataPath)
        waveform_data = createWaveformDict(OpenEphysDataPath, UseChans, UseRaw=UseRaw, 
                                           noise_cut_off=noise_cut_off, threshold=threshold)
        # Add this dictonary to list of all dictionaries
        waveform_datas.append(waveform_data)
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
                waveform_datas_comb[ntet]['timestamps'] = np.append(waveform_datas_comb[ntet]['timestamps'], 
                                                                    tmp_waveform_data[ntet]['timestamps'], axis=0)   
        # Apply KlustaKwik on the combined waveform_data
        waveform_datas_comb = applyKlustaKwik(waveform_datas_comb)
        # Extract clusters for each original waveform_data
        for ndata in range(len(waveform_datas)):
            for ntet in range(len(waveform_datas[ndata])):
                # Get clusterIDs for this dataset and tetrode
                nspikes = len(waveform_datas[ndata][ntet]['timestamps'])
                clusterIDs = waveform_datas_comb[ntet]['clusterIDs'][range(nspikes)]
                waveform_datas[ndata][ntet]['clusterIDs'] = clusterIDs
                # Remove these clusterIDs from the list
                waveform_datas_comb[ntet]['clusterIDs'] = np.delete(waveform_datas_comb[ntet]['clusterIDs'], range(nspikes), axis=0)
    # Overwrite clusterIDs in recording files
    for OpenEphysDataPath, waveform_data in zip(OpenEphysDataPaths, waveform_datas):
        for ntet in range(len(waveform_data)):
            NWBio.save_tetrode_clusterIDs(OpenEphysDataPath, waveform_data[ntet]['nr_tetrode'], 
                                          waveform_data[ntet]['clusterIDs'], overwrite=True)
    # Save data in Axona Format
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
