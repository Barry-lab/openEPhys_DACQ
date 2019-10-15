import zmq
from threading import Lock, Thread
from time import sleep, time
import traceback
import socket
import pickle


def get_localhost_ip():
    # Get local IP address
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    address = s.getsockname()[0]
    s.close()

    return address


class paired_messenger(object):
    """
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
    """
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
            self.add_callback(lambda msg: print(msg))

        # Add verification callback
        self.verification_dict = {}
        self.add_callback(self._verification_check)

        # Wait a moment to allow connection to be established
        sleep(2)
        
        # Start listening thread
        self.lock = Lock()
        self.is_running = True
        self.thread = Thread(target=self._run)
        self.thread.start()

    def sendMessage(self, message, verify=False):
        """Sends message to the paired messenger.

        :param bytes message: sent to paired device
        :param bool verify: if True, sendMessage() waits until such message is received back
        """
        if verify:
            self.verification_dict[message] = False
        self.socket.send(message)
        if verify:
            while not self.verification_dict[message]:
                sleep(0.05)
            del self.verification_dict[message]

    def close(self):

        if self.thread.is_alive():

            self.lock.acquire()
            self.is_running = False
            self.lock.release()

            self.thread.join()

            self.socket.disconnect(self.url)

    def add_callback(self, cb):
        """
        The callback function is started in a new thread if a message is received.
        To pass arguments with callback function, input a list or a tuple with
            callback function as the first element and the rest as individual arguments.
        The callback function should be expecting a message input as a string. If additional
            arguments were passed using the list or tuple method, these arguments should be
            expected by the function in the same order after the message string.
        """
        self.callbacks.append(cb)

    def _send_message_to_callbacks(self, msg):

        for cb in self.callbacks:
            if isinstance(cb, list) or isinstance(cb, tuple):
                Thread(target=cb[0], args=(msg,) + tuple(cb[1:])).start()
            else:
                Thread(target=cb, args=(msg,)).start()

    def _process_message(self, msg):
        """
        Called for each received message.
        Execution blocks reception of new messages. Threading recommended.
        """
        self._send_message_to_callbacks(msg)

    def _run(self):

        while True:

            self.lock.acquire()
            running = self.is_running
            self.lock.release()

            if not running:
                break

            try:
                msg = self.socket.recv()
                self._process_message(msg)

            except zmq.ZMQError:
                pass

            sleep(.01)

    def _verification_check(self, msg):
        if msg in list(self.verification_dict.keys()):
            self.verification_dict[msg] = True


def decode_pickled_message(msg):
    return pickle.loads(msg)


def encode_pickled_message(data):
    return pickle.dumps(data)


class remote_controlled_object(paired_messenger):
    """
    When instantiated with an object this class executes any incoming commands on that object.
    The incoming commands are expected to be sent using remote_object_controller.

    Note! remote_object_controller must be instantiated pair() method called before
    remote_controlled_object is instantiated.

    This class and the object must remain in scope of an active process.

    Sending a 'close' command will call close command on the object,
    return value if requested and then closes the remote_controlled_object instance.
    """
    def __init__(self, obj, *args, **kwargs):
        """
        remote_controlled_object must be instantiated with the object as first input argument.

        See paired_messenger for other input arguments.
        """
        self.obj = obj
        super(remote_controlled_object, self).__init__(*args, **kwargs)
        self.sendMessage('handshake'.encode())

    @staticmethod
    def _parse_message(msg):
        # Extract command to call from message string
        command, msg = msg.split(' '.encode(), 1)
        command = command.decode()
        # Extract return_value request from remaining message string
        return_value, msg = msg.split(' '.encode(), 1)
        if return_value == 'True'.encode():
            return_value = True
        elif return_value == 'False'.encode():
            return_value = False
        else:
            raise ValueError('return_value was not as expected.')
        # Extract input arguments from remaining message string
        input_arguments = decode_pickled_message(msg)
        args = input_arguments['args']
        kwargs = input_arguments['kwargs']

        return command, return_value, args, kwargs

    def _process_command(self, msg):
        # Parse raw message
        command, return_value, args, kwargs = remote_controlled_object._parse_message(msg)
        # Execute command with or without return value
        if return_value:
            self._execute_command_with_return(command, args, kwargs)
        else:
            self._execute_command(command, args, kwargs)
        # If close command sent, this remote_controlled_object is also closed.
        if command == 'close':
            self.close()

    def _execute_command(self, command, args, kwargs):
        """
        Parses incoming ZMQ message for function name, input arguments and calls that function.
        """
        getattr(self.obj, command)(*args, **kwargs)
    
    def _execute_command_with_return(self, command, args, kwargs):
        """
        Parses incoming ZMQ message for function name, input arguments and calls that function.
        """
        return_value = getattr(self.obj, command)(*args, **kwargs)
        encoded_return_value = encode_pickled_message(return_value)
        self.sendMessage(encoded_return_value)

    def _process_message(self, msg):
        Thread(target=self._process_command, args=(msg,)).start()


class remote_controlled_class(remote_controlled_object):
    """
    Allows using a class with remote_controlled_object before it is instantiated.

    Note! remote_object_controller must be instantiated pair() method called before
    remote_controlled_class is instantiated.

    remote_controlled_class must remain in scope to function. This can be achieved by calling
    it with block=True (see __init__() method) or by regularly checking isAlive() method
    to see if 'close' command has been received.

    On a paired remote_object_controller instance the sendInitCommand method
    must be called to instantiate the class before sendCommand method can be used.

    Once the class has been instantiated, remote_controlled_class behaves as remote_controlled_object.
    """
    def __init__(self, C, block, *args, **kwargs):
        """
        C - class to be used
        block - bool - if True, remote_controlled_class blocks until 'close' command is received.

        See paired_messenger for other input arguments.
        """
        self.C = C
        self.class_instantiated = False
        self.keep_class_alive = True
        super(remote_controlled_class, self).__init__(None, *args, **kwargs)
        if block:
            while self.keep_class_alive:
                sleep(0.1)

    def _init_object(self, args, kwargs):
        self.obj = self.C(*args, **kwargs)
        self.class_instantiated = True
        self.sendMessage('init_confirmation'.encode())

    def _process_command(self, msg):
        if self.class_instantiated:
            super(remote_controlled_class, self)._process_command(msg)
        else:
            command, return_value, args, kwargs = self._parse_message(msg)
            if command == '__init__':
                self._init_object(args, kwargs)

    def isAlive(self):
        """
        Returns boolean whether the remote_controlled_class is still alive.
        Returns False after close command has been received and instantiated class has been closed.
        """
        return self.keep_class_alive

    def close(self, *args, **kwargs):
        super(remote_controlled_class, self).close(*args, **kwargs)
        self.keep_class_alive = False


class remote_object_controller(paired_messenger):
    """
    Provides access to an object on a remote device via ZMQ.

    Can be paired with either remote_controlled_object or remote_controlled_class.

    Note! remote_object_controller must be instantiated and pair() method called
    before remote_controlled_object or remote_controlled_class is instantiated.

    If paired with remote_controlled_class, sendInitCommand must be called once paired
    to use other methods.

    See sendCommand() method for how to send commands and receive return values.
    """
    def __init__(self, *args, **kwargs):
        """
        See paired_messenger for other input arguments.
        """
        self.wait_for_handshake = True
        self.new_return_message = False
        self.wait_for_init_confirmation = False
        super(remote_object_controller, self).__init__(*args, **kwargs)

    def pair(self, timeout=0):
        """
        Returns True if successfully paired. Returns False if timeout reached.

        timeout - float - in seconds. If timeout=0 (default), pair() waits indefinitely.
        """
        wait_start_time = time()
        while self.wait_for_handshake:
            sleep(0.1)
            if timeout > 0 and (time() - wait_start_time) > timeout:
                break
        
        return not self.wait_for_handshake

    @staticmethod
    def encode_input_arguments(args, kwargs):
        input_arguments = {'args': args, 
                           'kwargs': kwargs}
        return encode_pickled_message(input_arguments)

    def sendInitCommand(self, timeout, *args, **kwargs):
        """
        Returns True if class successfully instantiated. Returns False if timeout reached.

        timeout - float - in seconds. If timeout=0 (default), sendInitCommand() waits indefinitely.

        Any following arguments are passed into the class __init__().
        """
        self.wait_for_init_confirmation = True
        encoded_input_arguments = remote_object_controller.encode_input_arguments(args, kwargs)
        self.sendMessage('__init__'.encode() + ' '.encode()
                         + 'True'.encode() + ' '.encode()
                         + encoded_input_arguments)
        wait_start_time = time()
        while self.wait_for_init_confirmation:
            sleep(0.1)
            if 0 < timeout < (time() - wait_start_time):
                break
        
        return not self.wait_for_init_confirmation

    def sendCommand(self, command, return_value, *args, **kwargs):
        """
        command - str - Name of the command to call on the object controlled via ZMQ
        return_value - bool - Whether to return value from command call.
                              return_value=True blocks until return value is received.
        Any additional input arguments are used as input arguments in command call on controlled object.
        These additional input arguments are pickled, compressed on this end
        and uncompressed, unpickled on the paired device.
        """
        encoded_input_arguments = remote_object_controller.encode_input_arguments(args, kwargs)
        self.sendMessage(command.encode() + ' '.encode()
                         + str(return_value).encode() + ' '.encode()
                         + encoded_input_arguments)
        if return_value:
            return self._wait_for_return_value()

    def _wait_for_return_value(self):
        """
        Blocks until return value is available.
        Returns the return value.
        """
        while not self.new_return_message:
            sleep(0.1)
        return_value = decode_pickled_message(self.return_message)
        self.new_return_message = False
        
        return return_value

    def _process_return_message(self, msg):
        assert not self.new_return_message  # Assuming return values are always processed before next arrives.
        self.return_message = msg
        self.new_return_message = True

    def _process_message(self, msg):
        if self.wait_for_handshake:
            if msg == 'handshake'.encode():
                self.wait_for_handshake = False
        elif self.wait_for_init_confirmation:
            if msg == 'init_confirmation'.encode():
                self.wait_for_init_confirmation = False
        else:
            Thread(target=self._process_return_message, args=(msg,)).start()


class PublishToOpenEphys(object):
    """
    This class allows sending messages to Open Ephys GUI over ZMQ.
    When created with defulat inputs, it will connect to Open Ephys GUI.
    Use sendMessage method to send messages to Open Ephys GUI
    """
    def __init__(self, address='localhost', port=5556):
        # Set up ZMQ connection to OpenEphysGUI
        url = "tcp://%s:%d" % (address, port)
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(url)

    def sendMessage(self, message):
        """Encodes message string into bytes and sends to OpenEphysGUI

        :param str message:
        :return:
        """
        self.socket.send(message.encode())
        _ = self.socket.recv()

    def close(self):
        self.socket.close()


def SendOpenEphysSingleMessage(message):
    """
    This function creates ZMQ connection with Open Ephys GUI just to send one message.
    This is sufficiently fast for single messages that are not very time sensitive.
    """
    messenger = PublishToOpenEphys()
    messenger.sendMessage(message)
    messenger.close()


class SubscribeToOpenEphys(object):
    """subscription-based zmq network event receiver

        This can be used to receive events published using the EventPublisher
        plugin.
    """

    def __init__(self, address='localhost', port=5557, timeout=2,
                 message_filter=''.encode(), verbose=True, save_messages=False):

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
        self.lock = Lock()
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
        self.thread = Thread(target=self._run)
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
            Thread(target=cb, args=(msg,)).start()

    def _run(self):

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

                self._send_message_to_callbacks(msg.decode())

            except zmq.ZMQError:
                pass

            sleep(.01)

def SubscribeToOpenEphys_message_callback(msg):
    print("received event:", msg)

def SubscribeToOpenEphys_run_example(args):
    """
    An example script for using SubscribeToOpenEphys
    """

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

        # _run for T seconds
        sleep(T)

    except BaseException:
        traceback.print_exc()

    finally:
        # make sure the background thread is being stopped before exiting
        sub.disconnect()
