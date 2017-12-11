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
if args.chan:
    UseChans = [args.chan[0] - 1, args.chan[1]]
else:
    UseChans = False
# Get file path from script call input
OpenEphysDataPath = args.path
NWBfilePath = os.path.join(OpenEphysDataPath,'experiment_1.nwb')
if args.keep:
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

# Load file, if promted only specific channels
print('Loading NWB data')
if UseChans:
    data = np.array(NWBio.load_continuous(NWBfilePath)['continuous'][:,UseChans[0]:UseChans[1]])
else:
    data = np.array(NWBio.load_continuous(NWBfilePath)['continuous'])
# If BadChan file exists, zero values for those channels
if os.path.exists(os.path.join(OpenEphysDataPath,'BadChan')):
    badChan = np.array(NWBio.listBadChannels(OpenEphysDataPath), dtype=np.int16)
    if UseChans:
        badChan = badChan[badChan >= np.array(UseChans[0], dtype=np.int16)]
        badChan = badChan - np.array(UseChans[0], dtype=np.int16)
        badChan = badChan[badChan < data.shape[1]]
    data[:,badChan] = np.int16(0)
# Write binary file for KiloSort
print('Writing NWB into binary')
data.tofile(os.path.join(KiloSortProcessingFolder,KiloSortBinaryFileName))
# Run KiloSort
eng = matlab.engine.start_matlab()
eng.cd('KiloSortScripts')
eng.master_file(float(data.shape[1]), KiloSortProcessingFolder, KiloSortBinaryFileName, nargout=0)

# Make sure position data is available
if not os.path.exists(os.path.join(OpenEphysDataPath,'PosLogComb.csv')):
    if NWBio.check_if_binary_pos(NWBfilePath):
        _ = NWBio.load_pos(NWBfilePath, savecsv=True, postprocess=True)
    else:
        CombineTrackingData.combdata(NWBfilePath)

# Define Axona data subfolder name based on specific channels if requested
if UseChans:
    subfolder = 'AxonaData_' + str(UseChans[0] + 1) + '-' + str(UseChans[1])
else:
    subfolder = 'AxonaData'
# Create Axona data based on KiloSort output
createAxonaData.createAxonaData(OpenEphysDataPath, 
                                KiloSortProcessingFolder, speedcut=0, 
                                subfolder=subfolder, UseChans=UseChans, 
                                eegChan=1)
if not args.keep:
    shutil.rmtree(KiloSortProcessingFolder)
