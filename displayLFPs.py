
from openEPhys_DACQ import NWBio
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(fpath):
    print('Opening NWB file...')
    data = NWBio.load_continuous(fpath)
    # Load timestamps into memory
    timestamps = np.array(data['timestamps'])
    timestamps = timestamps - timestamps[0]
    # Get user specified start and end time
    print('Session length is ' + str(timestamps[-1]) + ' seconds.')
    start_time = float(raw_input('Enter segment start time: '))
    end_time = float(raw_input('Enter segment end time: '))
    start_idx = np.argmin(np.abs(timestamps - start_time))
    end_idx = np.argmin(np.abs(timestamps - end_time))
    # Get user specified start and end time
    print('There are ' + str(data['continuous'].shape[1]) + ' channels.')
    first_chan = int(raw_input('Enter first channel nr: ')) - 1
    last_chan = int(raw_input('Enter last channel nr: '))
    # Load continuous data of specified shape
    print('Loading continuous data for segment...')
    timestamps = timestamps[start_idx:end_idx]
    continuous = np.array(data['continuous'][start_idx:end_idx, first_chan:last_chan])
    continuous = continuous.astype(np.float32) * 0.195
    n_channels = continuous.shape[1]
    n_datapoints = continuous.shape[0]
    data['file_handle'].close()
    # Convert continuous data such that it can all be show in single plot
    print('Converting continuous data for display...')
    mean_std = np.mean(np.std(continuous, axis=0))
    channel_spacing = mean_std * 3
    channel_spacer = np.linspace(0, (n_channels - 1) * channel_spacing, n_channels)
    channel_spacer = np.repeat(np.transpose(channel_spacer[:, None]), n_datapoints, axis=0)
    continuous = continuous + channel_spacer
    # Plot data
    print('Plotting...')
    fig = plt.figure()
    ax = plt.gca()
    plt.axis('off')
    plt.plot(timestamps, continuous, linewidth=1)
    plt.xlim(np.min(timestamps), np.max(timestamps))
    plt.ylim(np.min(continuous), np.max(continuous))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Display specific section of RAW data from NWB file.')
    parser.add_argument('fpath', type=str, nargs=1, 
                        help='Path to NWB file.')
    args = parser.parse_args()
    # Get paths to recording files
    fpath = args.fpath[0]
    main(fpath)
