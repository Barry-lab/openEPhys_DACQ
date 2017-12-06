import NWBio
import sys
import numpy as np

dataFileName = str(sys.argv[1])
driveID = int(sys.argv[2])

if driveID == 1:
	channels = np.arange(64)
elif driveID == 2:
	channels = np.arange(64) + 64
data = np.array(NWBio.load_continuous(dataFileName)['continuous'][:,channels])

data.tofile('/media/DataDrive/sander/Documents/KiloSortProcess/experiment_1.dat')