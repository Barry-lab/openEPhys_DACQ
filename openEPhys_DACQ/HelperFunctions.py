# -*- coding: utf-8 -*-
import sys
from scipy.signal import butter, lfilter, decimate
import os
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import multiprocessing
import threading
from time import sleep
import psutil
from datetime import datetime
import codecs


def time_string():
    return datetime.now().strftime('%H:%M:%S')


def encode_bytes(x):
    return x if isinstance(x, bytes) else codecs.latin_1_encode(x)[0]


class CPU_availability_tracker(object):
    """
    Keeps track of available cores.
    """
    def __init__(self, n_cores):
        self._available_cores = [False for i in range(n_cores)] # Start unavailable to debug locks
        self._core_locks = [multiprocessing.Lock() for i in range(n_cores)]
        self._T__check_locks = threading.Thread(target=self._check_locks)
        self._T__check_locks.daemon = True
        self._T__check_locks.start()

    def _check_locks(self, frequency=10):
        """
        Checks locks of all unavailable cores at set interval.
        Unlocked locks indicate a core is available.

        frequency - int - how many times per second to check locks
        """
        while True:
            sleep(1.0 / frequency)
            # Check for unavailable cores
            unavailable_nrs = [i for i, state in enumerate(self._available_cores) if not state]
            # Check locks for unavailable cores
            for nr in unavailable_nrs:
                if self._core_locks[nr].acquire(block=False):
                    self._core_locks[nr].release()
                    self._available_cores[nr] = True

    def check_if_core_available(self):
        return True in self._available_cores

    def wait_for_available_core(self, check_interval=0.01):
        """
        check_interval - seconds to wait between checking for available cores.
        """
        while not self.check_if_core_available():
            sleep(check_interval)

    def use_next_available(self, block=True):
        """
        Get available core number and set it to unavailable.

        block - bool - if True, waits until core is available.
                Otherwise, if core not available, returns None

        Returns:
            nr - the number of CPU core to use, starting from 0
            lock - multiprocessing.Lock instance.
                   This must be called with lock.release()
                   for the CPU core nr to become available again.
        """
        if block:
            self.wait_for_available_core()
        elif not self.check_if_core_available():
            return None
        # Get available core number and set it to unavailable
        nr = self._available_cores.index(True)
        # Lock the core lock and set core as unavailable
        self._core_locks[nr].acquire()
        self._available_cores[nr] = False

        return nr, self._core_locks[nr]


class multiprocess(object):
    """
    This class makes it easy to use multiprocessing module in a loop.
    Use map method if memory usage is not an issue, otherwise
    use the map method as an example how to use run and results methods.
    """
    def __init__(self):
        self.CPU_availability_tracker = CPU_availability_tracker(multiprocessing.cpu_count()-1 or 1)
        self.multiprocessingManager = multiprocessing.Manager()
        self.output_list = self.multiprocessingManager.list([])
        self.output_list_Lock = multiprocessing.Lock()
        self.processor_list = []
        self.run_Lock = threading.Lock()

    @staticmethod
    def processor(output_list, output_list_Lock, list_pos, 
                  cpu_lock, f, args, kwargs):
        """
        This method called by run method in a separate process to utilize multiprocessing
        """
        # Evaluate the function with input arguments
        output = f(*args, **kwargs)
        # Update output list and active process counter
        with output_list_Lock:
            output_list[list_pos] = output
        # Release CPU core lock to inform CPU_availability_tracker
        cpu_lock.release()

    def run(self, f, args=(), kwargs=None, single_cpu_affinity=False):
        """
        This method processes the function in a separate multiprocessing.Process,
        allowing the use of multiple CPU cores.
        If all CPU cores are in use, this function blocks until a core is from a process.

        f      - function to be called in a separate process
        args   - tuple of arguments for function f
        kwargs - dictionary of keyword arguments for function f

        Optional input
            single_cpu_affinity - bool - if true, 'cpu_core_nr' keyword argument
                                  is added to kwargs or owerwritten. 
                                  The value of 'cpu_core_nr'
                                  circles through the list of available CPUs.
        """
        # If kwargs is None, create empty dict
        if kwargs is None:
            kwargs = {}
        with self.run_Lock:
            # Wait for next available cpu core count
            cpu_nr, cpu_lock = self.CPU_availability_tracker.use_next_available(block=True)
            # If requested, specify CPU to use
            if single_cpu_affinity:
                kwargs['cpu_core_nr'] = cpu_nr
            # Extend list and acquire current position
            with self.output_list_Lock:
                list_pos = len(self.output_list)
                self.output_list.append(None)
            # Start independent processor
            p = multiprocessing.Process(target=multiprocess.processor,
                                        args=(self.output_list, self.output_list_Lock, list_pos,
                                              cpu_lock, f, args, kwargs))
            p.start()
            self.processor_list.append(p)

    def results(self):
        """
        This method returns a list, where each element corresponds to output
        from functions as they are passed into run method, in the same order.
        This method blocks if run method is currently being called or
        if processors have not yet finished.
        """
        # Ensure run method is not currently running
        with self.run_Lock:
            # Ensure all processors have finished
            for p in self.processor_list:
                p.join()

        return self.output_list

    @staticmethod
    def args_kwargs_list_check(n, args_list, kwargs_list):
        # Set empty argument lists if not provided
        if args_list is None:
            args_list = [[] for x in range(n)]
        if kwargs_list is None:
            kwargs_list = [{} for x in range(n)]

        return args_list, kwargs_list

    def map(self, f, n, args_list=None, kwargs_list=None, 
            single_cpu_affinity=False, max_memory_usage=1):
        """
        This function evaluates function f for number of times specified by argument n, 
        with each set of arguments in args_list and kwargs_list.
        It outputs a list where each element is the output
        from function f for each set of arguments in args_list.
        This function is more convenient to use that run and results separately,
        but it requires loading all input arguments into memory, which is not always ideal.

        f - function to evaluate
        n - number of calls to f

        Optional input:
            args_list   - list of tuples of arguments for function f
            kwargs_list - list of dictionaries of keyword arguments for function f
            single_cpu_affinity - bool - if true, 'cpu_core_nr' keyword argument
                                  is added to kwargs or owerwritten. 
                                  The value of 'cpu_core_nr'
                                  circles through the list of available CPUs.
            max_memory_usage    - float - (0.0 - 1.0) percentage of memeory that must
                                  be available for function to process next element in list.
        """
        args_list, kwargs_list = multiprocess.args_kwargs_list_check(n, args_list, kwargs_list)
        for args, kwargs in zip(args_list, kwargs_list):
            if max_memory_usage < 1:
                proceed_when_enough_memory_available(percent=(1.0-max_memory_usage))
            self.run(f, args, kwargs, single_cpu_affinity)

        return self.results()


def proceed_when_enough_memory_available(memory_needed=None, percent=None, array_size=None, dtype=None):
    """
    This function blocks until required memory is available.
    Input can be any of the following:
        memory_needed - in bytes
        percent - percent (e.g 0.75 for 75%) of total memory to be available
        array_size and dtype - to compute memory needed to store array in memory
    """
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


def lowpass_and_downsample(signal_in, sampling_rate_in, sampling_rate_out, 
                           suppress_division_by_two_error=False):
    """
    Implements `scipy.signal.decimate` method with FIR forward pass filter and
    phase shift correction.

    signal_in         - numpy array with shape (N,).
                        Input shapes (N, 1) and (1, N) are also accepted but these
                        are flattened and output shape is (N,).
                        The output will have the same dtype as signal_in.
    sampling_rate_in  - int - sampling rate of input signal
    sampling_rate_out - int - desired sampling rate of output signal.
                        The division of sampling_rate_in by sampling_rate_out must have 
                        zero remainder.
                        Furthermore, the resulting quotient must have 0 remainder when
                        divided by 2. Otherwise scipy package gives a warning about
                        bad coefficients.

    Optional:

    suppress_division_by_two_error - bool - allows using sampling_rate_out that does not
                                            satisfy the requirement that the resulting 
                                            quotient with sampling_rate_in has remainder 0 
                                            when divided by sampling_rate_in. 
                                            Default is False.

    """

    # Ensure input signal has the correct shape
    if len(signal_in.shape) > 1:
        if len(signal_in.shape) == 2 and signal_in.shape[1] == 1:
            signal_in = signal_in.squeeze()
        elif len(signal_in.shape) == 2 and signal_in.shape[0] == 1:
            signal_in = signal_in.squeeze()
        else:
            raise Exception('signal_in must have shape (N,), (N, 1) or (1, N)')

    # Ensure sampling rates can be divided without remainder
    if sampling_rate_in % sampling_rate_out != 0:
        raise Exception('sampling_rate_out must be a factor of sampling_rate_in.')

    # Compute downsampling factor
    downsampling_factor = int(sampling_rate_in / sampling_rate_out)

    # Verify that downsampling factor can be divided by 2 without remainder
    if not suppress_division_by_two_error:
        if downsampling_factor % 2 != 0:
            raise Exception('Quotient of sampling_rate_in and sampling_rate_out '
                            + 'must have zero remainder when divided by 2.')

    # Filter and downsample the signal
    signal_out = decimate(signal_in, downsampling_factor, ftype='fir', zero_phase=True)

    # Ensure output is in same dtype as input signal
    if not (signal_in.dtype is signal_out.dtype):
        signal_out = signal_out.astype(signal_in.dtype)

    return signal_out


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
    if len(signal_in.shape) > 1:
        raise Exception('signal_in must have shape (N,), but has shape ' + str(signal_in.shape))
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
    tetrode_nrs = list(range(firstTet, lastTet + 1, 1))

    return tetrode_nrs

def import_subdirectory_module(subdirectory, module_name):
    path = list(sys.path)
    sys.path.insert(0, subdirectory)
    try:
        module = __import__(module_name)
    finally:
        sys.path[:] = path

    return module

def test_pinging_address(address='localhost'):
    """
    Returns True if successful in pinging the input IP address, False otherwise
    """
    with open(os.devnull, 'w') as devnull:
        result = subprocess.call(['ping', '-c', '3', address], stdout=devnull, stderr=devnull)
    if result == 0:
        return True
    else:
        return False


def closest_argmin(a, b):
    """Returns the closest matching element indices in b for each element in a.

    :param numpy.ndarray a: shape (N,)
    :param numpy.ndarray b: shape (M,)
    :return: closest_b_indices
    :rtype: numpy.ndarray
    """
    b_length = b.size
    sorted_idx_b = b.argsort()
    sorted_b = b[sorted_idx_b]
    sorted_idx = np.searchsorted(sorted_b, a)
    sorted_idx[sorted_idx == b_length] = b_length - 1
    mask = (sorted_idx > 0) & (np.abs(a - sorted_b[sorted_idx-1]) < np.abs(a - sorted_b[sorted_idx]))

    return sorted_idx_b[sorted_idx-mask]


def openSingleFileDialog(load_save, directory_path=os.path.expanduser("~"), suffix='', caption='Choose File'):
    """Opens a GUI dialog for browsing files when closed returns full path to file.

    If no file is selected but dialog is closed, then this function returns None.

    :param str load_save: either 'save' or 'load', which determines if all or only existing files can be selected
    :param str directory_path: optionally specify starting directory
    :param str suffix: specify filename suffixes to filter for
    :param str caption: caption to display on dialog
    :return: fpath
    """
    # Pops up a GUI to select a single file.
    dialog = QtWidgets.QFileDialog(directory=directory_path, caption=caption)
    if load_save is 'save':
        dialog.setFileMode(QtWidgets.QFileDialog.AnyFile)
    elif load_save is 'load':
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    dialog.setViewMode(QtWidgets.QFileDialog.List) # or Detail
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
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(message)
    if message_more:
        msg.setInformativeText(message_more)
    msg.setWindowTitle('Message')
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    return msg.exec()


class QThread_with_completion_callback(QThread):
    finishedSignal = pyqtSignal()
    """
    Allows calling funcitons in separate threads with a callback function executed at completion.

    Note! Make sure QThread_with_completion_callback instance does not go out of scope during execution.
    """
    def __init__(self, callback_function, function, return_output=False, callback_args=(), function_args=()):
        """
        The callback_function is called when function has finished.
        The function is called with any input arguments that follow it.
        """
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


class ClassInSeparateProcess(object):
    """
    This class can be used to instantiate and control another class in a separate process.

    The controlled class must have a `closed` attribute that indicates
    if the class has been closed after instantiation.

    Inputs to the task must be functional in a separate process created with multiprocessing package.
    """

    def __init__(self, cls, args=None, kwargs=None):
        """
        :param cls: class with `close` attribute to indicate if class has been closed after instantiation
        :param tuple args: positional arguments to pass into class
        :param dict kwargs: keyword arguments to pass into class
        """

        command_receiver_pipe, self._command_sender_pipe = multiprocessing.Pipe()

        self._class_process = multiprocessing.Process(target=self.class_process_method,
                                                      args=(cls, command_receiver_pipe,
                                                            () if args is None else args,
                                                            {} if kwargs is None else kwargs))
        self._class_process.start()

        while True:
            if self._command_sender_pipe.poll(0.1):
                if self._command_sender_pipe.recv() == 'initialization successful':
                    print('Class {} started successfully in separate process.'.format(cls))
                    break
                else:
                    raise Exception('Unknown input from class {} process.'.format(cls))

    @property
    def class_process_active(self):
        """Returns True if the processes with class instance is active and False otherwise.

        :return: alive_bool
        :rtype: bool
        """
        return self._class_process.is_alive()

    @staticmethod
    def class_process_method(cls, command_receiver_pipe, args, kwargs):
        """This method initializes the class and runs until it has completed `close` command.

        Any string received via `command_receiver_pipe` are executed as class method names
        and output returned via `command_receiver_pipe`.

        :param cls: class with `close` attribute to indicate if class has been closed after instantiation
        :param multiprocessing.connection command_receiver_pipe:
        :param tuple args: positional arguments to pass into class
        :param dict kwargs: keyword arguments to pass into class
        """

        class_instance = cls(*args, **kwargs)

        command_receiver_pipe.send('initialization successful')

        while not class_instance.closed:

            if command_receiver_pipe.poll(0.1):

                method_name = command_receiver_pipe.recv()

                ret = getattr(class_instance, method_name)()

                command_receiver_pipe.send(ret)

    def call_class_method(self, name):
        """Calls the named method on the class running in another process and returns output.

        This method blocks until the method in another process returns.

        :param str name: name of the method to be called
        :return: return_from_method
        """
        if self.class_process_active:
            self._command_sender_pipe.send(name)
            return self._command_sender_pipe.recv()
        else:
            raise Exception('Class Process is not running. Can not receive commands.')
