# -*- coding: utf-8 -*-
import sys
from scipy.signal import butter, lfilter
import os
import numpy as np
from PyQt4 import QtGui


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(signal_in, sampling_rate=30000.0, highpass_frequency=300.0, lowpass_frequency=6000.0, filt_order=4):
    b, a = butter_bandpass(highpass_frequency, lowpass_frequency, sampling_rate, order=filt_order)
    signal_out = lfilter(b, a, signal_in)
    return signal_out


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(signal_in, lowpass_frequency=125.0, sampling_rate=30000.0, filt_order=4):
    b, a = butter_lowpass(lowpass_frequency, sampling_rate, order=filt_order)
    signal_out = lfilter(b, a, signal_in)
    return signal_out


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


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=40, initiation=False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    if not initiation:
        sys.stdout.flush()
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('%s |%s| %s%s %s\r' % (prefix, bar, percents, '%', suffix)),

    if iteration >= total:
        sys.stdout.write('\n')
    sys.stdout.flush()
# # 
# # Sample Usage
# # 
# from time import sleep
# # A List of Items
# items = list(range(0, 57))
# l = len(items)
# # Initial call to print 0% progress
# print_progress(0, l, prefix = 'Progress:', suffix = 'Complete',initiation=True)
# for i, item in enumerate(items):
#     # Do stuff...
#     sleep(0.1)
#     # Update Progress Bar
#     print_progress(i + 1, l, prefix = 'Progress:', suffix = 'Complete')


def get_position_data_edges(fpath):
    # Get position data first and last timestamps
    PosFilePath = os.path.join(fpath,'PosLogComb.csv')
    pos_csv = np.genfromtxt(PosFilePath, delimiter=',')
    pos_timestamps = np.array(pos_csv[:,0], dtype=np.float64)
    pos_edges = [pos_timestamps[0], pos_timestamps[-1]]

    return pos_edges


def channels_tetrode(nchan):
    # Returns the tetrode number this channel is on (all counting starts from 0)
    nchan = float(nchan + 1)
    ntet = np.ceil(nchan / 4) - 1
    ntet = int(ntet)

    return ntet


def tetrode_channels(ntet):
    # Returns the list of channels on this tetrode all counting starts from 0)
    nchan = ntet * 4
    nchans = [nchan, nchan + 1, nchan + 2, nchan + 3]

    return nchans


def import_subdirectory_module(subdirectory, module_name):
    path = list(sys.path)
    sys.path.insert(0, subdirectory)
    try:
        module = __import__(module_name)
    finally:
        sys.path[:] = path

    return module

def openSingleFileDialog(loadsave, directory=os.path.expanduser("~"), suffix='', caption='Choose File'):
    # Pops up a GUI to select a single file.
    dialog = QtGui.QFileDialog(directory=directory, caption=caption)
    if loadsave is 'save':
        dialog.setFileMode(QtGui.QFileDialog.AnyFile)
    elif loadsave is 'load':
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
    dialog.setViewMode(QtGui.QFileDialog.List) # or Detail
    if len(suffix) > 0:
        dialog.setNameFilter('*.' + suffix)
        dialog.setDefaultSuffix(suffix)
    if dialog.exec_():
        # Get path and file name of selection
        tmp = dialog.selectedFiles()
        selected_file = str(tmp[0])

    return selected_file