import NWBio
import sys
import numpy as np

dataFileName = str(sys.argv[1])
if len(sys.argv) > 2:
    # UseChans = [0,32] will limit finding waveforms to channels 1 to 32
    # UseChans = [32,64] will limit finding waveforms to channels 33 to 64
    # UseChans = [0,64] will limit finding waveforms to channels 1 to 64
    # UseChans = [64,128] will limit finding waveforms to channels 65 to 128
    UseChans = [int(sys.argv[2]), int(sys.argv[3])]
    data = np.array(NWBio.load_continuous(dataFileName)['continuous'][:,UseChans[0]:UseChans[1]])
else:
    data = np.array(NWBio.load_continuous(dataFileName)['continuous'])

data.tofile('/media/DataDrive/sander/Documents/KiloSortProcess/experiment_1.dat')