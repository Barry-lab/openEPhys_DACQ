import zmq
import threading
import thread
import time
import sys
import traceback

class PublishToOpenEphys(object):
    # This class allows sending messages to open ephys over ZMQ
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
    # This function is for sending a single message that is not time-sensitive
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

def message_callback(msg):
    """add code here to handle incoming message"""

    print("received event:", msg)

def run_example(args):

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
        sub.add_callback(message_callback)
        sub.connect()

        # run for T seconds
        time.sleep(T)

    except BaseException:
        traceback.print_exc()

    finally:
        # make sure the background thread is being stopped before exiting
        sub.disconnect()
