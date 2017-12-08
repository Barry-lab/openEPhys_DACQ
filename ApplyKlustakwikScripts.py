# -*- coding: utf-8 -*-
'''
Created on Mon Jun 18 18:31:31 2012

@author: robin
'''
import numpy as np
import os
import re
from subprocess import Popen, PIPE

class Kluster():
    '''
    Runs KlustaKwik (KK) against data recorded on the Axona dacqUSB recording
    system
    
    Inherits from axonaIO.Tetrode to allow for mask construction (for the
    newer version of KK) and easy access to some relevant information (e.g
    number of spikes recorded)
    '''
    def __init__(self, filename, tet_num, feature_array):
        '''
        Inherits from dacq2py.IO so as to be able to load .set file for fmask
        construction (ie remove grounded channels, eeg channels etc)
        
        Parameters
        ---------------
        filename: fully qualified, absolute root filename (i.e. without the
        .fet.n)
        
        tet_num: the tetrode number
        
        feature_array: array containing the features to be put into the fet file
        
        '''
        self.filename = filename
        self.tet_num = tet_num
        self.feature_array = feature_array
        self.n_features = feature_array.shape[1] / 4
        self.distribution = 1
        self.feature_mask = None

    def make_fet(self):
        '''
        Creates and writes a .fet.n file for reading in  to KlustaKwik given
        the array input a
        '''
        fet_filename = self.filename + '.fet.' + str(self.tet_num)
        with open(fet_filename, 'w') as f:
            f.write(str(self.feature_array.shape[1]))
            f.write('\n')
            np.savetxt(f, self.feature_array, fmt='%1.5f')

    def get_mask(self):
        '''
        Returns a feature mask based on unused channels, eeg recordings etc
        Loads the set file associated with the trial and creates two dicts
        containing the mode on each channel and the channels which contain the
        eeg recordings
        keys and values for both dicts are in the form of ints NB mode is
        numbered from 0 upwards but eeg_filter is from 1 upwards
        The mode key/ value pair of the Axona .set file correspond to the
        following values:
        2 - eeg
        5 - ref-sig
        6 - grounded
        The collectMask key in the Axona .set file corresponds to whether or
        not a tetrode was kept in the recording - use this also to construct
        the feature mask
        '''
        #  use the feature array a to calculate which channels to include etc
        sums = np.sum(self.feature_array, 0)
        feature_mask = np.repeat(np.ones(4, dtype=np.int), self.n_features)
        #  if there are "missing" channels use the older version of KK
        zero_sums = sums == 0
        if np.any(zero_sums):
            self.distribution = 1
            feature_mask[zero_sums] = 0
        self.feature_mask = feature_mask
        return feature_mask

    def make_fmask(self, feature_mask):
        '''
        Create a .fmask.n file for use in the new (01/09/14) KlustaKwik program
        where n denotes tetrode id
        From the github site:
        "The .fmask file is a text file, every line of which is a vector of
        length the number of features, in which 1 denotes unmasked and 0
        denotes masked, and values between 0 and 1 indicate partial masking"
        Inputs:
            filename: fully qualified, absolute root filename (i.e. without the
            .fmask.n)
            a: array containing the features to be put into the fet file
            n: the tetrode number
            feature_mask: array of numbers between 0 and 1 (see above
            description from github site)
        '''
        fmask_filename = self.filename + '.fmask.' + str(self.tet_num)
        mask = np.tile(feature_mask, (self.feature_array.shape[0], 1))
        with open(fmask_filename, 'w') as f:
            f.write(str(self.feature_array.shape[1]))
            f.write('\n')
            np.savetxt(f, mask, fmt='%1d')

    def kluster(self):
        '''
        Using a .fet.n file this makes a system call to KlustaKwik (KK) which
        clusters data and saves istributional ' + str(self.distribution) + 
            ' -MinClusters 5'
            ' -MaxPossibleClusters 31'
            ' -MaskStarts 30'
            ' -FullStepEvery 1'
            ' -SplitEvery 40'
            ' -UseMaskedInitialConditions 1'
            ' -AssignToFirstClosestMask 1'
            ' -DropLastNFeatures 1'
            ' -RandomSeed 123'
            ' -PriorPoint 1'
            ' -MaxIter 10000'
            ' -PenaltyK 1'
            ' -PenaltyKLogN 0'
            ' -Log 0'
            ' -DistThthe result in a cut file that can be read into
        Axona's Tint cluster cutting app
        Inputs:
            fname - the root name of the file (i.e. without the .fet.n)
        Outputs:
            None but saves a Tint-friendly cut file in the same directory as
            the spike data
        '''
        # specify path to KlustaKwik exe
        kk_path = os.path.expanduser('~') + '/Programs/klustakwik/KlustaKwik'
        if not os.path.exists(kk_path):
            print kk_path
            raise IOError()
        kk_proc = Popen(
            kk_path + ' ' +
            self.filename + ' ' +
            str(self.tet_num) +
            ' -UseDistributional ' + str(self.distribution) + 
            ' -MinClusters 5'
            ' -MaxPossibleClusters 31'
            ' -MaskStarts 30'
            ' -FullStepEvery 1'
            ' -SplitEvery 40'
            ' -UseMaskedInitialConditions 1'
            ' -AssignToFirstClosestMask 1'
            ' -DropLastNFeatures 1'
            ' -RandomSeed 123'
            ' -PriorPoint 1'
            ' -MaxIter 10000'
            ' -PenaltyK 1'
            ' -PenaltyKLogN 0'
            ' -Log 0'
            ' -DistThresh 9.6'
            ' -UseFeatures ' + ''.join(map(str, self.feature_mask))
        , shell=True, stdout=PIPE)
        # Print the output of the KlustaKwik algo
        for line in kk_proc.stdout:
            print line.replace('\n', '')
            
        '''
        now read in the .clu.n file that has been created as a result of this
        process and create the Tint-friendly cut file
        '''
        clu_filename = self.filename + '.clu.' + str(self.tet_num)
        clu_data = np.loadtxt(clu_filename)
        n_clusters = clu_data[0]
        clu_data = clu_data[1:] - 1  # -1 so cluster 0 is junk
        n_chan = 4
        n_spikes = int(clu_data.shape[0])
        cut_filename = self.filename.split('.')[0] + '_' + str(self.tet_num) + '.cut'
        with open(cut_filename, 'w') as f:
            f.write('n_clusters: {nClusters}\n'.format(nClusters=n_clusters.astype(int)))
            f.write('n_channels: {nChan}\n'.format(nChan=n_chan))
            f.write('n_params: {nParam}\n'.format(nParam=2))
            f.write('times_used_in_Vt:    {Vt}    {Vt}    {Vt}    {Vt}\n'.format(Vt=0))
            for i in range(0, n_clusters.astype(int)):
                f.write(' cluster: {i} center:{zeros}\n'.format(i=i, zeros='    0    0    0    0    0    0    0    0'))
                f.write('                min:{zeros}\n'.format(i=i, zeros='    0    0    0    0    0    0    0    0'))
                f.write('                max:{zeros}\n'.format(i=i, zeros='    0    0    0    0    0    0    0    0'))
            f.write('Exact_cut_for: {fname} spikes: {nSpikes}\n'.format(fname=os.path.basename(self.filename), nSpikes=str(n_spikes)))
            for spk in clu_data:
                f.write('{spk}  '.format(spk=spk.astype(int)))

#   def cleanup(self):
#       '''
#       Removes any extraneous files following the call to KlustaKwik
#       '''
#       files_to_remove = ['.klg', '.clu', '.fet', '.fmask', '.initialclusters.2.clu']
#
#       for f in files_to_remove:
#           try:
#               os.remove(self.filename + f Parameters
#       ---------------+ '.' + str(self.tet_num))
#           except IOError:
#               pass

def getParam(waveforms=None, param='Amp', t=200, fet=1):
    '''
    Returns the requested parameter from a spike train as a numpy array
    
    Parameters
    -------------------
    
    waveforms - numpy array     
        Shape of array can be an nSpikes x nSamples
        OR
        a nSpikes x nElectrodes x nSamples
    
    param - str
        Valid values are:
            'Amp' - peak-to-trough amplitude (default)
            'P' - height of peak
            'T' - depth of trough
            'Vt' height at time t
            'tP' - time of peak (in seconds)
            'tT' - time of trough (in seconds)
            'PCA' - first n fet principal components (defaults to 1)
            
    t - int
        The time used for Vt
        
    fet - int
        The number of principal components (used with param 'PCA')
    '''
    from scipy import interpolate
    from sklearn.decomposition import PCA
            
    if param == 'Amp':
        return np.ptp(waveforms, axis=-1)
    elif param == 'P':
        return np.max(waveforms, axis=-1)
    elif param == 'T':
        return np.min(waveforms, axis=-1)
    elif param == 'Vt':
        times = np.arange(0,1000,20)
        f = interpolate.interp1d(times, range(50), 'nearest')
        if waveforms.ndim == 2:
            return waveforms[:, int(f(t))]
        elif waveforms.ndim == 3:
            return waveforms[:, :, int(f(t))]
    elif param == 'tP':
        idx = np.argmax(waveforms, axis=-1)
        m = interpolate.interp1d([0, waveforms.shape[-1]-1], [0, 1/1000.])
        return m(idx)
    elif param == 'tT':
        idx = np.argmin(waveforms, axis=-1)
        m = interpolate.interp1d([0, waveforms.shape[-1]-1], [0, 1/1000.])
        return m(idx)
    elif param == 'PCA':
        pca = PCA(n_components=fet)
        if waveforms.ndim == 2:
            return pca.fit(waveforms).transform(waveforms).squeeze()
        elif waveforms.ndim == 3:
            out = np.zeros((waveforms.shape[0], waveforms.shape[1] * fet))
            st = np.arange(0, waveforms.shape[1] * fet, fet)
            en = np.arange(fet, fet + (waveforms.shape[1] * fet), fet)
            rng = np.vstack((st, en))
            for i in range(waveforms.shape[1]):
                if ~np.any(np.isnan(waveforms[:,i,:])):
                    A = np.squeeze(pca.fit(waveforms[:,i,:].squeeze()).transform(waveforms[:,i,:].squeeze()))
                    if A.ndim < 2:
                        out[:,rng[0,i]:rng[1,i]] = np.atleast_2d(A).T
                    else:
                        out[:,rng[0,i]:rng[1,i]] = A
            return out

def klustakwik(waveforms, d, filename_root):
    """ 
    Calls two methods below (kluster and getPC) to run klustakwik on
    a given tetrode with nFet number of features (for the PCA)

    Parameters
    ---------------
        d : dict
            Specifies the vector of features to be used in
            clustering. Each key is the identity of a tetrode (i.e. 1, 2 etc)
             and the values are the features used to do the clustering for that tetrode (i.e.
            'PC1', 'PC2', 'Amp' (amplitude) etc
    """
    legal_values = ['PC1', 'PC2', 'PC3', 'PC4', 'Amp',
                    'Vt', 'P', 'T', 'tP', 'tT', 'En', 'Ar']
    reg = re.compile(".*(PC).*")  # check for number of principal comps
    # check for any input errors in whole dictionary first
    for i_tetrode in d.keys():
        for v in d[i_tetrode]:
            if v not in legal_values:
                raise ValueError('Could not find %s in %s' % (v, legal_values))
    # iterate through features and see what the max principal component is
    for i_tetrode in d.keys():
        pcs = [m.group(0) for l in d[i_tetrode] for m in [reg.search(l)] if m]
        waves = waveforms
        princomp = None
        if pcs:
            max_pc = []
            for pc in pcs:
                max_pc.append(int(pc[2]))
            num_pcs = np.max(max_pc)  # get max number of prin comps
            princomp = getParam(waves, param='PCA', fet=num_pcs)
            # Rearrange the output from PCA calc to match the 
            # number of requested principal components
            inds2keep = []
            for m in max_pc:
                inds2keep.append(np.arange((m-1)*4, (m)*4))
            inds2keep = np.hstack(inds2keep)
            princomp = np.take(princomp, inds2keep, axis=1)
        out = []
        for value in d[i_tetrode]:
            if 'PC' not in value:
                out.append(getParam(waves, param=value))
        if princomp is not None:
            out.append(princomp)
        out = np.hstack(out)
        
        c = Kluster(filename_root, i_tetrode, out)
        c.make_fet()
        mask = c.get_mask()
        c.make_fmask(mask)
        c.kluster()

# Below stuff is written by Sander, UCL, 31/10/2017
import NWBio
import pickle
import createAxonaData

def get_position_data_edges(filename):
    fpath = filename[:filename.rfind('/')]
    # Get position data first and last timestamps
    posLog_fileName = 'PosLogComb.csv'
    pos_csv = np.genfromtxt(fpath + '/' + posLog_fileName, delimiter=',')
    pos_timestamps = np.array(pos_csv[:,0], dtype=np.float64)
    pos_edges = [pos_timestamps[0], pos_timestamps[-1]]

    return pos_edges

def listBadChannels(fpath):
    # Find file BadChan in the directory and extract numbers from each row
    badChanFile = os.path.join(fpath,'BadChan')
    if os.path.exists(badChanFile):
        with open(badChanFile) as file:
            content = file.readlines()
        content = [x.strip() for x in content]
        badChan = list(np.array(map(int, content)) - 1)
    else:
        badChan = []

    return badChan

def cluster_all_spikes_NWB(filename):
    # Loads whole NWB spike data, cuts off spikes outside position data, clusters all tetrodes
    # filename - the full path to the raw data file
    fpath = filename[:filename.rfind('/')]

    spike_data = NWBio.load_spikes(filename)# Get raw data file

    pos_edges = get_position_data_edges(filename)

    badChan = listBadChannels(fpath)

    files_created = []
    for ntet in range(len(spike_data)):
        waveforms = -np.array(spike_data[ntet]['waveforms']) # Waveforms are inverted
        waveforms = np.swapaxes(waveforms,1,2)
        timestamps = np.array(spike_data[ntet]['timestamps'])
        # Set bad channel waveforms to 0
        channels = np.arange(4) + 4 * ntet
        if len(badChan) > 0:
            for bc in badChan:
                badChanOnTetrode = channels == bc
                if np.any(badChanOnTetrode):
                    waveforms[:,:,badChanOnTetrode] = 0
        # Remove spikes outside position data range
        idx_delete = np.logical_or(timestamps < pos_edges[0], timestamps > pos_edges[1])
        timestamps = timestamps[np.logical_not(idx_delete)]
        waveforms = waveforms[np.logical_not(idx_delete),:,:]
        # Remove spikes where maximum amplitude exceedes limit
        noise_cut_off = 500
        noise_cut_off = np.int16(np.round(noise_cut_off / 0.195))
        idx_delete = np.any(np.any(np.abs(waveforms) > noise_cut_off, axis=2), axis=1)
        timestamps = timestamps[np.logical_not(idx_delete)]
        waveforms = waveforms[np.logical_not(idx_delete),:,:]
        # Create filename base for tetrode
        tet_file_basename = 'Tet_' + str(ntet + 1) + '_CH' + '_'.join(map(str, list(channels + 1)))
        # Save logical array of spikes used
        idx_keep = np.logical_not(idx_delete)
        np.savetxt(os.path.join(fpath,tet_file_basename + '.SpikesUsed'),idx_keep,fmt='%i')
        # Set up dictionary to be saved for this tetrode
        waveform_data = {'waveforms': waveforms[:,:31,:], 
                         'spiketimes': timestamps, 
                         'nr_tetrode': ntet, 
                         'tetrode_channels': channels, 
                         'badChan': badChan, 
                         'bitVolts': float(0.195)}
        # Save as a pickle file
        wave_filename = tet_file_basename + '.spikes.p'
        with open(os.path.join(fpath,wave_filename),'wb') as file:
            print('Saving waveforms for tetrode ' + str(ntet + 1))
            pickle.dump(waveform_data, file)
        files_created.append(tet_file_basename)
        # Applying Klustkwik on tetrode
        print('Applying KlustaKwik on tetrode ' + str(ntet + 1))
        waveforms = np.swapaxes(waveforms,1,2)
        features2use = ['PC1', 'PC2', 'PC3', 'Amp', 'Vt']
        d = {0: features2use}
        klustakwik(waveforms, d, os.path.join(fpath,tet_file_basename))
        # Delete all files aside from .clu.0 and .SpikesUsed
        extensions = ['.fet.0','.fmask.0','.initialclusters.2.clu.0','.temp.clu.0','_0.cut']
        for extension in extensions:
            os.remove(os.path.join(fpath,tet_file_basename + extension))
    # Load up createWaveformGUIdata
    fileNames = createAxonaData.getAllFiles(fpath, files_created)
    createAxonaData.createAxonaData(fpath, fileNames)
