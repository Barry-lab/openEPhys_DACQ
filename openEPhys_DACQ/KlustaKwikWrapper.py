# -*- coding: utf-8 -*-
'''
Created on Mon Jun 18 18:31:31 2012

@author: robin
'''
import numpy as np
import os
import re
from subprocess import Popen, PIPE, STDOUT
import tempfile
import shutil

from openEPhys_DACQ.package_configuration import package_config


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

    def kluster(self, max_possible_clusters=31, cpu_core_nr=None):
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
        if cpu_core_nr is None:
            taskset_cpu_affinity = ''
        else:
            taskset_cpu_affinity = 'taskset -c ' + str(int(cpu_core_nr)) + ' '
        # specify path to KlustaKwik exe
        kk_path = package_config()['klustakwik_path']
        if not os.path.exists(kk_path):
            print(kk_path)
            raise IOError()
        # This FNULL stops klustakwik from printing progress reports
        FNULL = open(os.devnull, 'w')
        kk_proc = Popen(
            taskset_cpu_affinity +
            kk_path + ' ' +
            self.filename + ' ' +
            str(self.tet_num) +
            ' -UseDistributional ' + str(self.distribution) + 
            ' -MinClusters 5'
            ' -MaxPossibleClusters ' + str(max_possible_clusters) + 
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
            ' -Verbose 0'
            ' -UseFeatures ' + ''.join(map(str, self.feature_mask))
        , shell=True, stdout=FNULL, stderr=STDOUT)
        # Print the output of the KlustaKwik algo
        kk_proc.communicate()
        if kk_proc.stderr:
            for line in kk_proc.stderr:
                print(line.replace('\n', ''))
            
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

def klustakwik(waveforms, d, filename_root, max_possible_clusters=31, cpu_core_nr=None):
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
        c.kluster(max_possible_clusters=max_possible_clusters, 
                  cpu_core_nr=cpu_core_nr)

def applyKlustaKwik_on_spike_data_tet(spike_data_tet, max_possible_clusters=31, 
                                      cpu_core_nr=None):
    '''
    Returns the input dictionary with added field 'clusterIDs'
    Input dictionary required fields:
        'waveforms' - nspikes x waveformLength x n_channels - spike waveforms for one tetrode
        'idx_keep' - logical indexing array of length nspikes to specify which spikes to use
    Returns a vector clusterIDs of length equal to sum(idx_keep)

        Added by Sander Tanni 04/06/2018
    '''
    if spike_data_tet['waveforms'].shape[0] == 0:
        return np.array([], dtype=np.int16)
    if spike_data_tet['waveforms'].shape[0] < 4:
        return np.ones(spike_data_tet['waveforms'].shape[0], dtype=np.int16)

    # Create spike waveform array and filter using idx_keep
    waves = spike_data_tet['waveforms'][spike_data_tet['idx_keep'],:,:]
    if waves.shape[0] == 0:
        return np.array([], dtype=np.int16)
    if waves.shape[0] < 4:
        return np.ones(waves.shape[0], dtype=np.int16)

    # Create temporary processing folder
    KlustaKwikProcessingFolder = tempfile.mkdtemp('KlustaKwikProcessing')
    # Prepare input to KlustaKwik
    features2use = ['PC1', 'PC2', 'PC3', 'Amp', 'Vt']
    d = {0: features2use}
    klustakwik(waves, d, os.path.join(KlustaKwikProcessingFolder, 'KlustaKwikTemp'), 
               max_possible_clusters=max_possible_clusters, 
               cpu_core_nr=cpu_core_nr)
    # Read in cluster IDs
    cluFileName = os.path.join(KlustaKwikProcessingFolder, 'KlustaKwikTemp.clu.0')
    with open(cluFileName, 'rb') as file:
        lines = file.readlines()
    # Delete KlustaKwik temporary processing folder
    shutil.rmtree(KlustaKwikProcessingFolder)
    # Parse lines into an array and return it
    clusterIDs = []
    for line in lines:
        clusterIDs.append(int(line.rstrip()))
    clusterIDs = clusterIDs[1:] # Drop the first value which is number of spikes

    return np.array(clusterIDs, dtype=np.int16)
