import matlab.engine
import sys
import numpy as np
import os
import NWBio
import CombineTrackingData
import createAxonaData

# To install matlab engine, go to folder /usr/local/MATLAB/R2017a/extern/engines/python
# and run terminal command: sudo python setup.py install

# Set default location for KiloSortProcessing Folder
KiloSortProcessingFolder = '/media/DataDrive/sander/Documents/KiloSortProcess/'
KiloSortBinaryFileName = 'experiment_1.dat'

# Get file path from script call input
NWBfilePath = str(sys.argv[1])
# Load file, if promted only specific channels
print('Writing NWB into binary')
if len(sys.argv) > 2:
    # UseChans = [0,32] will limit finding waveforms to channels 1 to 32
    # UseChans = [32,64] will limit finding waveforms to channels 33 to 64
    # UseChans = [0,64] will limit finding waveforms to channels 1 to 64
    # UseChans = [64,128] will limit finding waveforms to channels 65 to 128
    UseChans = [int(sys.argv[2]), int(sys.argv[3])]
    data = np.array(NWBio.load_continuous(NWBfilePath)['continuous'][:,UseChans[0]:UseChans[1]])
else:
    UseChans = False
    data = np.array(NWBio.load_continuous(NWBfilePath)['continuous'])
# Write binary file for KiloSort
data.tofile(os.path.join(KiloSortProcessingFolder,KiloSortBinaryFileName))
# Run KiloSort
eng = matlab.engine.start_matlab()
eng.cd('KiloSortScripts')
eng.master_file(float(data.shape[1]), KiloSortProcessingFolder, KiloSortBinaryFileName, nargout=0)

# Make sure position data is available
if not os.path.exists(os.path.join(os.path.dirname(NWBfilePath),'PosLogComb.csv')):
    if NWBio.check_if_binary_pos(NWBfilePath):
        _ = NWBio.load_pos(NWBfilePath,savecsv=True)
    else:
        CombineTrackingData.combdata(NWBfilePath)

# Define Axona data subfolder name based on specific channels if requested
if UseChans:
    subfolder = 'AxonaData_' + str(UseChans[0]) + '-' + str(UseChans[1])
else:
    subfolder = 'AxonaData'
# Create Axona data based on KiloSort output
createAxonaData.createAxonaData(os.path.dirname(NWBfilePath), 
                                KiloSortProcessingFolder, speedcut=0, 
                                subfolder=subfolder, UseChans=UseChans, 
                                eegChan=1)
