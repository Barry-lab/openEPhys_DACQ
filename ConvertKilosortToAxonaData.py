import sys
import os
import createAxonaData
import CombineTrackingData as combPos
import NWBio

# fpath = str(sys.argv[1])

# UseChans = [0,32] will limit finding waveforms to channels 1 to 32
# UseChans = [32,64] will limit finding waveforms to channels 33 to 64
# UseChans = [0,64] will limit finding waveforms to channels 1 to 64
# UseChans = [64,128] will limit finding waveforms to channels 65 to 128
UseChans = [0,64]
fpath = '/media/DataDrive/sander/Documents/RobinData'
KiloSortOutputPath = '/media/DataDrive/sander/Documents/KiloSortProcess'

NWBfilePath = os.path.join(fpath,'experiment_1.nwb')
data = NWBio.load_pos(NWBfilePath,savecsv=True)
# combPos.combdata(NWBfilePath)


createAxonaData.createAxonaData(fpath, KiloSortOutputPath, speedcut=0, subfolder='AxonaData',UseChans=UseChans,eegChan=1)
