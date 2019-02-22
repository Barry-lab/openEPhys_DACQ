import os
import sys
import h5py
from NWBio import get_recordingKey, get_processorKey, check_if_path_exists
import subprocess

def parse_subprocess_output(subprocess_output):
    '''
    Parses subprocess print output into dictionary.
    Each row is considered to be a variable with the following format
    name value type_parser
    e.g. 'successful 1 bool' or 'duration 3.6 float'
    '''
    output = {}
    for part in subprocess_output.split(','):
        if len(part) > 0:
            datapoints = part.split(' ')
            if len(datapoints) == 2:
                output[datapoints[0]] = datapoints[1]
            elif len(datapoints) == 3:
                output[datapoints[0]] = eval(datapoints[2])(datapoints[1])
            else:
                raise ValueError('Incorrect input format')

    return output

def subprocess_with_stdout(args):
    """
    Calls subprocess.Popen with args tuple.
    Prints stdout output while the call is in progress.

    Returns the last output as string.
    """
    process = subprocess.Popen(args, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            last_output = output.strip()
            print(last_output)

    return last_output

def main(root_path):
    # Prepare counters
    nwb_noRawData = 0
    nwb_raw_deleted = 0
    nwb_repacked = 0
    nwb_processing_failed = 0
    nwb_noSpikes = 0
    ioError = 0
    # Set downsampling parameters
    downsampling = 30
    lowpass_freq = 500
    # Commence directory walk
    for dirName, subdirList, fileList in os.walk(root_path):
        for fname in fileList:
            if not ('Experiment' in dirName):
                fpath = os.path.join(dirName, fname)
                if fname == 'experiment_1.nwb':
                    try:
                        recordingKey = get_recordingKey(fpath)
                        processorKey = get_processorKey(fpath)
                        path_processor = '/acquisition/timeseries/' + recordingKey + '/continuous/' + processorKey
                        # Check if raw data available
                        SpikesAvailable = False
                        with h5py.File(fpath,'r') as h5file:
                            raw_data_available = path_processor + '/data' in h5file
                        if raw_data_available:
                            # Check is spikes available
                            spikes_path = '/acquisition/timeseries/' + recordingKey + '/spikes/'
                            with h5py.File(fpath,'r') as h5file:
                                nchan = h5file[path_processor + '/data'].shape[1]
                            if check_if_path_exists(fpath, spikes_path):
                                # Establish how many tetrodes are in this recording
                                with h5py.File(fpath, 'r') as h5file:
                                    n_tetrodes = len(h5file[spikes_path].keys())
                                if nchan >= 128 and n_tetrodes == 32:
                                    SpikesAvailable = True
                                    auxChanStart = 128
                                if nchan < 128 and n_tetrodes == 16:
                                    SpikesAvailable = True
                                    auxChanStart = 64
                            if SpikesAvailable:
                                # Process the dataset
                                print('Downsample and Repack: ' + fpath)
                                input_args = ('python', 'DeleteRAWdataProcess.py', 
                                              fpath, path_processor, str(lowpass_freq), str(downsampling), 
                                              str(n_tetrodes), str(int(auxChanStart)), str(nwb_raw_deleted), str(nwb_repacked))
                                subprocess_output = subprocess_with_stdout(input_args)
                                subprocess_output = parse_subprocess_output(subprocess_output)
                                if subprocess_output['successful'] == 1:
                                    print('Successfully processed: ' + fpath)
                                    nwb_raw_deleted = subprocess_output['nwb_raw_deleted']
                                    nwb_repacked = subprocess_output['nwb_repacked']
                                else:
                                    print('Failed to process: ' + fpath)
                                    nwb_processing_failed += 1
                            else:
                                print('No Spikes: ' + fpath)
                                nwb_noSpikes += 1
                        else:
                            print('No RAW data: ' + fpath)
                            nwb_noRawData += 1
                    except IOError:
                        print('IOError in file: ' + fpath)
                        ioError += 1
    # Print counter values
    print('\nProcessing output:')
    print('nwb_noRawData ' + str(nwb_noRawData))
    print('nwb_raw_deleted ' + str(nwb_raw_deleted))
    print('nwb_repacked ' + str(nwb_repacked))
    print('nwb_processing_failed ' + str(nwb_processing_failed))
    print('nwb_noSpikes ' + str(nwb_noSpikes))
    print('ioError ' + str(ioError))


if __name__ == '__main__':
    # Get directory walk root path
    if len(sys.argv) < 2:
        raise ValueError('Enter path to process as the first argument!')
    else:
        root_path = sys.argv[1]
    main(root_path)
