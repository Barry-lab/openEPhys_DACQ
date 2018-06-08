import os
import sys
import h5py
from NWBio import get_recordingKey, get_processorKey, check_if_path_exists
import subprocess
import Processing_KlustaKwik

if __name__ == '__main__':
    # Get directory walk root path
    if len(sys.argv) < 2:
        raise ValueError('Enter path to process as the first argument!')
    else:
        root_path = sys.argv[1]

    # Commence directory walk
    for dirName, subdirList, fileList in os.walk(root_path):
        for fname in fileList:
            if not ('Experiment' in dirName):
                fpath = os.path.join(dirName, fname)
                if fname == 'experiment_1.nwb':
                    AxonaDataExists = any(['AxonaData' in subdir for subdir in subdirList])
                    if not AxonaDataExists:
                        Processing_KlustaKwik.main(fpath, [0, 128], False, 1000, 50, False)
