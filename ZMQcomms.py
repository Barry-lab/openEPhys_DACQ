import zmq
import threading
import thread
import time
import sys
import traceback
import socket

def get_localhost_ip():
    # Get local IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    address = s.getsockname()[0]
    s.close()

    return address

class paired_messenger(object):
    '''
    This class can exchange messages with another pair_messenger instance running on the same
        or different machine, pointed to the same IP address.
    Bind one of the instances to address='localhost' 
        and the other pointing to the corresponding IP address.
    Use add_callback method to add callbacks which will be called when a message
        is received. You can couple arguments with the callback if presented as a list
        or tuple, with callback being the first element.
    NOTE! On slower systems, the initialization process may take some time.
        If messages are expected to be received or sent immediately, a delay may be necessary
        after instantiation of this object. Depending on system, 0.25 to 1.0 seconds.
    '''
    def __init__(self, address='localhost', port=5884, timeout=0.5, printMessages=False):
        # Identify if client or server instance
        if address == 'localhost':
            self.localhost = True
            address = get_localhost_ip()
        else:
            self.localhost = False
        # Connect to an address
        self.url = "tcp://%s:%d" % (address, port)
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        if self.localhost:
            self.socket.bind(self.url)
        else:
            self.socket.connect(self.url)
        self.socket.RCVTIMEO = int(timeout * 1000)  # in milliseconds
        # Set callbacks list
        self.callbacks = []
        if printMessages:
            self.add_callback(self.printMessage)
        # Add verification callback
        self.verification_dict = {}
        self.add_callback(self.verification_check)
        # Start listening thread
        self.lock = threading.Lock()
        self.is_running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def sendMessage(self, message, verify=False):
        '''
        message - str - sent to paired device
        verify - str - if set, sendMessage() waits until such message is received back
        '''
        if not (verify is False):
            self.verification_dict[verify] = False
        self.socket.send(message)
        if not (verify is False):
            while not self.verification_dict[verify]:
                time.sleep(0.05)

    def close(self):

        if self.thread.is_alive():

            self.lock.acquire()
            self.is_running = False
            self.lock.release()

            self.thread.join()

            self.socket.disconnect(self.url)

    def add_callback(self, cb):
        '''
        The callback function is started in a new thread if a message is received.
        To pass arguments with callback function, input a list or a tuple with
            callback function as the first element and the rest as individual arguments.
        The callback function should be expecting a message input as a string. If additional
            arguments were passed using the list or tuple method, these arguments should be
            expected by the function in the same order after the message string.
        '''
        self.callbacks.append(cb)

    def _send_message_to_callbacks(self, msg):

        for cb in self.callbacks:
            if isinstance(cb, list) or isinstance(cb, tuple):
                threading.Thread(target=cb[0], args=(msg,) + tuple(cb[1:])).start()
            else:
                threading.Thread(target=cb, args=(msg,)).start()

    def run(self):

        while True:

            self.lock.acquire()
            running = self.is_running
            self.lock.release()

            if not running:
                break

            try:
                msg = self.socket.recv()
                self._send_message_to_callbacks(msg)

            except zmq.ZMQError:
                pass

            time.sleep(.01)

    def printMessage(self, msg):
        print(msg)

    def verification_check(self, msg):
        if msg in self.verification_dict.keys():
            self.verification_dict[msg] = True

class PublishToOpenEphys(object):
    '''
    This class allows sending messages to Open Ephys GUI over ZMQ.
    When created with defulat inputs, it will connect to Open Ephys GUI.
    Use sendMessage method to send messages to Open Ephys GUI
    '''
    def __init__(self, address='localhost', port=5556):
        # Set up ZMQ connection to OpenEphysGUI
        url = "tcp://%s:%d" % (address, port)
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(url)

    def sendMessage(self, message):
        self.socket.send(message)
        dump_response = self.socket.recv()

    def close(self):
        self.socket.close()


def SendOpenEphysSingleMessage(message):
    '''
    This function creates ZMQ connection with Open Ephys GUI just to send one message.
    This is sufficiently fast for single messages that are not very time sensitive.
    '''
    messenger = PublishToOpenEphys()
    messenger.sendMessage(message)
    messenger.close()


class SubscribeToOpenEphys(object):
    """subscription-based zmq network event receiver

        This can be used to receive events published using the EventPublisher
        plugin.
    """

    def __init__(self, address='localhost', port=5557, timeout=2,
                 message_filter="", verbose=True, save_messages=False):

        self.address = address
        self.port = port
        self.timeout = timeout
        self.message_filter = message_filter
        self.verbose = verbose
        self.save_messages = save_messages

        socket = None
        context = None
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.RCVTIMEO = int(timeout * 1000)  # in milliseconds

        self.socket = socket
        self.context = context

        self.current_url = None
        self.thread = None
        self.messages = []
        self.lock = threading.Lock()
        self.is_running = False
        self.callbacks = []

    def connect(self):

        if self.socket is None:
            return

        if self.is_connected():
            self.disconnect()

        url = "tcp://%s:%d" % (self.address, self.port)
        if self.verbose:
            print("Connecting subscriber to:", url)

        self.socket.connect(url)
        self.socket.setsockopt(zmq.SUBSCRIBE, self.message_filter)
        self.current_url = url

        self.is_running = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def disconnect(self):

        if self.socket is None:
            return

        if self.thread is not None and self.thread.is_alive():

            self.lock.acquire()
            self.is_running = False
            self.lock.release()

            self.thread.join()

        if self.is_connected():
            if self.verbose:
                print("Disconnecting subscriber from:", self.current_url)
            self.socket.disconnect(self.current_url)
            self.current_url = None

    def __del__(self):

        if self.socket is None:
            return

        if self.is_connected():
            if self.verbose:
                print("Disconnecting network subscriber ...")
            self.disconnect()

        if self.verbose:
            print("Terminating network context ...")
        self.socket.close()
        self.context.term()

    def is_connected(self):

        if self.socket is None:
            return False

        return self.current_url is not None

    def get_messages(self, clear=False):

        self.lock.acquire()
        msg = [m for m in self.messages]
        if clear:
            del self.messages[:]
        self.lock.release()

        return msg

    def add_callback(self, cb):

        self.callbacks.append(cb)

    def remove_callback(self, cb):

        if cb in self.callbacks:
            self.callbacks.remove(cb)

    def _send_message_to_callbacks(self, msg):

        for cb in self.callbacks:
            thread.start_new_thread(cb, (msg,))

    def run(self):

        while True:

            self.lock.acquire()
            running = self.is_running
            self.lock.release()

            if not running:
                break

            try:
                msg = self.socket.recv()

                if self.save_messages:
                    self.lock.acquire()
                    self.messages.append(msg)
                    self.lock.release()

                self._send_message_to_callbacks(msg)

            except zmq.ZMQError:
                pass

            time.sleep(.01)

def SubscribeToOpenEphys_message_callback(msg):
    print("received event:", msg)

def SubscribeToOpenEphys_run_example(args):
    '''
    An example script for using SubscribeToOpenEphys
    '''

    # parse arguments
    address = 'localhost'  # address of system on which OE is running
    port = 5557  # port
    T = 10  # how long to listen for messages

    if len(args) > 1:
        address = args[1]
    if len(args) > 2:
        port = int(args[2])
    if len(args) > 3:
        T = float(args[3])

    try:
        # connect subscriber to event publisher
        sub = SubscribeToOpenEphys(address=address, port=port)
        sub.add_callback(SubscribeToOpenEphys_message_callback)
        sub.connect()

        # run for T seconds
        time.sleep(T)

    except BaseException:
        traceback.print_exc()

    finally:
        # make sure the background thread is being stopped before exiting
        sub.disconnect()
