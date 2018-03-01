import CombineTrackingData
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

def extract_spikes_from_raw_data(NWBfilePath, UseChans=False, badChan=[], threshold=50):
    # Load data
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
    print('Make sure only data on a single drive is selected')
    continuous_mean = np.mean(continuous[goodChan,:], axis=0, keepdims=True)
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
            spike_data.append({'waveforms': waveforms, 'timestamps': timestamps[spiketimes.squeeze()]})
        else:
            spike_data.append({'waveforms': np.zeros((0,31,4), dtype=np.int16), 'timestamps': np.zeros((0), dtype=np.float64)})

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

def createWaveformDict(OpenEphysDataPath, UseChans=False, badChan=[], UseRaw=False, noise_cut_off=500, threshold=50):
    # This function processes NWB data captured spikes with KlustaKwik and organises data into dictonaries
    NWBfilePath = os.path.join(OpenEphysDataPath,'experiment_1.nwb')
    # Get thresholded spike data for each tetrode
    print('Loading spikes')
    spike_data = NWBio.load_spikes(NWBfilePath)
    if spike_data and not UseRaw:
        # Limit analysis to specific tetrodes if channels are specified
        if UseChans:
            firstTet = hfunct.channels_tetrode(UseChans[0])
            lastTet = hfunct.channels_tetrode(UseChans[1] - 1)
            spike_data = spike_data[firstTet:lastTet + 1]
        # Fully load into memory
        for ntet in range(len(spike_data)):
            spike_data[ntet]['waveforms'] = np.swapaxes(np.array(spike_data[ntet]['waveforms']),1,2)
        # Invert waveforms
        for ntet in range(len(spike_data)):
            spike_data[ntet]['waveforms'] = -spike_data[ntet]['waveforms']
        # Remove spikes below threshold
        threshold_int16 = np.int16(np.round(threshold / 0.195))
        for ntet in range(len(spike_data)):
            idx_keep = np.any(np.any(np.abs(spike_data[ntet]['waveforms']) > threshold_int16, axis=2), axis=1)
            spike_data[ntet]['timestamps'] = spike_data[ntet]['timestamps'][idx_keep]
            spike_data[ntet]['waveforms'] = spike_data[ntet]['waveforms'][idx_keep,:,:]
            if np.sum(idx_keep) < idx_keep.size:
                percentage_above_threshold = np.sum(idx_keep) / float(idx_keep.size) * 100
                print('{:.1f}% of spikes on tetrode {} were above threshold'.format(percentage_above_threshold, ntet + 1))
    else:
        print('Extracting spikes from raw data.')
        spike_data = extract_spikes_from_raw_data(NWBfilePath, UseChans, badChan, 
                                                  threshold)
    # Set bad channel waveforms to 0
    if badChan:
        for nchan in badChan:
            ntet = hfunct.channels_tetrode(nchan)
            spike_data[ntet]['waveforms'][:,:,np.mod(nchan,4)] = 0
    # Remove spikes outside position data
    pos_edges = hfunct.get_position_data_edges(OpenEphysDataPath)
    for ntet in range(len(spike_data)):
        idx_delete = np.logical_or(spike_data[ntet]['timestamps'] < pos_edges[0], spike_data[ntet]['timestamps'] > pos_edges[1])
        spike_data[ntet]['timestamps'] = spike_data[ntet]['timestamps'][np.logical_not(idx_delete)]
        spike_data[ntet]['waveforms'] = spike_data[ntet]['waveforms'][np.logical_not(idx_delete),:,:]
    # Remove spikes where maximum amplitude exceedes limit
    if noise_cut_off and (noise_cut_off != 0):
        noise_cut_off = np.int16(np.round(noise_cut_off / 0.195))
        for ntet in range(len(spike_data)):
            idx_delete = np.any(np.any(np.abs(spike_data[ntet]['waveforms']) > noise_cut_off, axis=2), axis=1)
            spike_data[ntet]['timestamps'] = spike_data[ntet]['timestamps'][np.logical_not(idx_delete)]
            spike_data[ntet]['waveforms'] = spike_data[ntet]['waveforms'][np.logical_not(idx_delete),:,:]
            if np.sum(idx_delete) > 0:
                percentage_too_big = np.sum(idx_delete) / float(idx_delete.size) * 100
                print('{:.1f}% of spikes removed on tetrode {}'.format(percentage_too_big,ntet+1))
    # Arrange data into a list of dictionaries
    waveform_data = []
    for ntet in range(len(spike_data)):
        # If there are no spikes on a tetrode, create one zero spike 1 second after first position sample
        if len(spike_data[ntet]['timestamps']) == 0:
            spike_data[ntet]['timestamps'] = np.array([pos_edges[0] + 1], dtype=np.float64)
            spike_data[ntet]['waveforms'] = np.zeros((1,31,4), dtype=np.int16)
        nchan = hfunct.tetrode_channels(ntet)[0]
        if UseChans:
            nchan = nchan + UseChans[0]
        tmp = {'waveforms': spike_data[ntet]['waveforms'][:,:31,:], 
               'spiketimes': spike_data[ntet]['timestamps'], 
               'nr_tetrode': hfunct.channels_tetrode(nchan), 
               'tetrode_channels': hfunct.tetrode_channels(hfunct.channels_tetrode(nchan)), 
               'badChan': badChan, 
               'bitVolts': float(0.195)}
        waveform_data.append(tmp)

    return waveform_data


def main(OpenEphysDataPaths, UseChans=False, UseRaw=False, noise_cut_off=500, threshold=50):
    if isinstance(OpenEphysDataPaths, basestring):
        OpenEphysDataPaths = [OpenEphysDataPaths]
    badChans = []
    for OpenEphysDataPath in OpenEphysDataPaths:
        # Assume NWB file has name experiment_1.nwb
        NWBfilePath = os.path.join(OpenEphysDataPath,'experiment_1.nwb')
        # Get bad channels and renumber according to used channels
        badChan = hfunct.listBadChannels(OpenEphysDataPath)
        if badChan:
            print('Ignoring bad channels: ' + str(list(np.array(badChan) + 1)))
            badChan = np.array(badChan, dtype=np.int16)
            if UseChans:
                # Correct bad channel number depending on used channels
                badChan = badChan[badChan >= np.array(UseChans[0], dtype=np.int16)]
                badChan = badChan - np.array(UseChans[0], dtype=np.int16)
                badChan = badChan[badChan < UseChans[1] - UseChans[0]]
            badChan = list(badChan)
        badChans.append(badChan)
        # Make sure position data is available
        if not os.path.exists(os.path.join(os.path.dirname(NWBfilePath),'PosLogComb.csv')):
            print('Creating PosLogComb.csv')
            if NWBio.check_if_binary_pos(NWBfilePath):
                _ = NWBio.load_pos(NWBfilePath, savecsv=True, postprocess=True)
            else:
                CombineTrackingData.combdata(NWBfilePath)
    # Extract spikes for each tetrode in each recording into a dictonary
    waveform_datas = []
    for OpenEphysDataPath, badChan in zip(OpenEphysDataPaths, badChans):
        waveform_data = createWaveformDict(OpenEphysDataPath, UseChans=UseChans, 
                                           badChan=badChan, UseRaw=UseRaw, 
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
