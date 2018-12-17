# -*- coding: utf-8 -*-
import sys
from scipy.signal import butter, lfilter
import os
import numpy as np
from PyQt4 import QtGui
import subprocess
import copy
import multiprocessing
from time import sleep
import psutil
from PyQt4.QtCore import QThread, pyqtSignal

class multiprocess(object):
    '''
    This class makes it easy to use multiprocessing module in a loop.
    Use map method if memory usage is not an issue, otherwise
    use the map method as an example how to use run and results methods.
    '''
    def __init__(self):
        self.max_processes = multiprocessing.cpu_count()-1 or 1
        self.multiprocessingManager = multiprocessing.Manager()
        self.n_active = multiprocessing.Value('i', 0)
        self.n_active_Lock = multiprocessing.Lock()
        self.output_list = self.multiprocessingManager.list([])
        self.output_list_Lock = multiprocessing.Lock()
        self.n_total = 0
        self.processor_list = []
        self.run_in_progress = False

    def processor(self, n_in_list, f, args):
        '''
        This method called by run method in a separate process to utilize multiprocessing
        '''
        # Evaluate the function with input arguments
        output = f(*args)
        # Update output list and active process counter
        with self.output_list_Lock:
            self.output_list[n_in_list] = output
        with self.n_active_Lock:
            self.n_active.value -= 1

    def run(self, f, args):
        '''
        This method processes the function in a separate multiprocessing.Process,
        allowing the use of multiple CPU cores.
        If all CPU cores are in use, this function blocks until a core is from a process.

        f - function to be called in a separate process
        args - tuple of input arguments for that function
        '''
        if not (type(args) is tuple):
            raise ValueError('Input args must be a tuple.')
        # Avoid this method being called in separate threads
        if self.run_in_progress:
            raise Exception('multiprocessing.run method called before previous call finshed.')
        self.run_in_progress = True
        # If maximum number of processes active, wait until one has finished
        with self.n_active_Lock:
            currently_active = copy.deepcopy(self.n_active.value)
        while currently_active >= self.max_processes:
            sleep(0.05)
            with self.n_active_Lock:
                currently_active = copy.deepcopy(self.n_active.value)
        # Update counters
        with self.n_active_Lock:
            self.n_active.value += 1
        self.n_total += 1
        # Start independent processor
        with self.output_list_Lock:
            self.output_list.append(None)
        n_in_list = self.n_total - 1
        p = multiprocessing.Process(target=self.processor, args=(n_in_list, f, args))
        p.start()
        self.processor_list.append(p)
        # Open run method for another call
        self.run_in_progress = False

    def results(self):
        '''
        This method returns a list, where each element corresponds to output
        from functions as they are passed into run method, in the same order.
        This method blocks if run method is currently being called or
        if processors have not yet finished.
        '''
        # Ensure run method is not currently running
        while self.run_in_progress:
            sleep(0.05)
        # Ensure all processors have finished
        for p in self.processor_list:
            p.join()

        return self.output_list

    def map(self, f, args_list):
        '''
        This function evaluates function f with each set of arguments in args_list
        using multiprocessing module. It outputs a list where each element is the output
        from function f for each set of arguments in args_list.
        This function is more convenient to use that run and results separately,
        but it requires loading all input arguments into memory, which is not always ideal.

        f - function to evaluate
        args_list - list of tuples of arguments for function f
        '''
        for args in args_list:
            self.run(f, args)

        return self.results()

def proceed_when_enough_memory_available(memory_needed=None, percent=None, array_size=None, dtype=None):
    '''
    This function blocks until required memory is available.
    Input can be any of the following:
        memory_needed - in bytes
        percent - percent (e.g 0.75 for 75%) of total memory to be available
        array_size and dtype - to compute memory needed to store array in memory
    '''
    # Compute the memory_needed if not provided directly
    if memory_needed is None:
        if percent is None:
            if array_size is None or dtype is None:
                raise ValueError('Input needed.')
            else:
                memory_needed = array_size * dtype(1).nbytes
        else:
            memory_needed = psutil.virtual_memory().total * percent
    # Wait until this amount of memory is available
    memory_available = False
    while not memory_available:
        if psutil.virtual_memory().available > memory_needed:
            memory_available = True
        else:
            sleep(1)

    return memory_available

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

def get_tetrode_nrs(channels):
    firstTet = channels_tetrode(channels[0])
    lastTet = channels_tetrode(channels[-1])
    tetrode_nrs = range(firstTet, lastTet + 1, 1)

    return tetrode_nrs

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

def show_message(message, message_more=None):
    # This function is used to display a message in a separate window
    msg = QtGui.QMessageBox()
    msg.setIcon(QtGui.QMessageBox.Information)
    msg.setText(message)
    if message_more:
        msg.setInformativeText(message_more)
    msg.setWindowTitle('Message')
    msg.setStandardButtons(QtGui.QMessageBox.Ok)
    msg.exec_()

def test_pinging_address(address='localhost'):
    '''
    Returns True if successful in pinging the input IP address, False otherwise
    '''
    with open(os.devnull, 'w') as devnull:
        result = subprocess.call(['ping', '-c', '3', address], stdout=devnull, stderr=devnull)
    if result == 0:
        return True
    else:
        return False

class QThread_with_completion_callback(QThread):
    finishedSignal = pyqtSignal()
    '''
    Allows calling funcitons in separate threads with a callback function executed at completion.

    Note! Make sure QThread_with_completion_callback instance does not go out of scope during execution.
    '''
    def __init__(self, callback_function, function, return_output=False, callback_args=(), function_args=()):
        '''
        The callback_function is called when function has finished.
        The function is called with any input arguments that follow it.
        '''
        super(QThread_with_completion_callback, self).__init__()
        if return_output:
            self.finishedSignal.connect(lambda: callback_function(self.function_output, *self.callback_args))
        else:
            self.finishedSignal.connect(lambda: callback_function(*self.callback_args))
        self.function = function
        self.callback_args = callback_args
        self.function_args = function_args
        self.start()

    def run(self):
        self.function_output = self.function(*self.function_args)
        self.finishedSignal.emit()


def clearLayout(layout, keep=0):
    # This function clears the layout so that it could be regenerated
    if layout is not None:
        while layout.count() > keep:
            item = layout.takeAt(keep)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clearLayout(item.layout())
