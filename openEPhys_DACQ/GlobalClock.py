import pigpio
from ZMQcomms import paired_messenger
import argparse
from time import sleep

class GlobalClock_TTL_emitter(object):
    '''
    Produces very reliable pulses at regular intervals.
    Can be controlled over network with ZMQ.
    '''
    def __init__(self, ZMQcontrol=False, ZMQport=6000, ttlPin=18, frequency=10, pulse_width=1):
        '''
        ZMQcontrol - bool - default is False. Returns 'init_successful' over ZMQ and waits for commands.
        ZMQport - int - port to use for ZMQ communication at localhost.
        ttlPin - int - NOTE! Only hardware_PWM enabled pin can be used. See pigpio documentation.
        frequency - float - signal frequency (Hz)
        pulse_width - float - pulse width (milliseconds)
        '''
        self.ZMQcontrol = ZMQcontrol
        if self.ZMQcontrol:
            self.initialize_ZMQcomms(ZMQport)
        self.init_TTL_signalling(frequency=frequency, pulse_width=pulse_width, ttlPin=ttlPin)
        if self.ZMQcontrol:
            self.ZMQmessenger.sendMessage('init_successful')
            self.wait_for_ZMQ_commands()

    def initialize_ZMQcomms(self, port):
        '''
        Initializes ZMQ communication with Recording PC.
        '''
        self.ZMQmessenger = paired_messenger(port=int(port))
        self.ZMQmessenger.add_callback(self.command_parser)
        sleep(1) # This ensures all ZMQ protocols have been properly initated before finishing this process

    def command_parser(self, message):
        '''
        Parses incoming ZMQ message for function name, input arguments and calls that function.
        '''
        method_name = message.split(' ')[0]
        args = message.split(' ')[1:]
        getattr(self, method_name)(*args)

    @staticmethod
    def pwm_frequency_and_duty_cycle(signal_frequency, pulse_width):
        '''
        Returns PWM frequency and duty cycle to produce a signal that repeats at exactly signal_frequency

        frequency - float - pulse frequency (Hz)
        pulseWidth - float - pulse width (milliseconds)
        '''
        interval_between_pulse_starts = 1000.0 / float(signal_frequency)
        interval_between_pulse_end_and_start = interval_between_pulse_starts - float(pulse_width)
        pwm_frequency = int(round(1000.0 / interval_between_pulse_end_and_start))
        pwm_duty_cycle_ratio = float(pulse_width) / interval_between_pulse_end_and_start
        pwm_duty_cycle = int(1000000 * pwm_duty_cycle_ratio)

        return pwm_frequency, pwm_duty_cycle

    def init_TTL_signalling(self, frequency=10, pulse_width=1, ttlPin=18):
        '''
        Initializes RPi GPIO pin for sending regular TTL pulses.
        
        frequency - float - signal frequency (Hz)
        pulse_width - float - pulse width (milliseconds)
        ttlPin - int - NOTE! Only hardware_PWM enabled pin can be used. See pigpio documentation.
        '''
        self.ttlPin = ttlPin
        self.pwm_frequency, self.pwm_duty_cycle = self.pwm_frequency_and_duty_cycle(frequency, pulse_width)
        self.piPWM = pigpio.pi()
        self.piPWM.hardware_PWM(self.ttlPin, 0, 0)

    def wait_for_ZMQ_commands(self):
        '''
        Keeps the process alive so it can respond to ZMQ commands.
        '''
        self.keep_waiting_for_ZMQ_commands = True
        while self.keep_waiting_for_ZMQ_commands:
            sleep(0.1)

    def start(self):
        self.piPWM.hardware_PWM(self.ttlPin, self.pwm_frequency, self.pwm_duty_cycle)

    def stop(self):
        self.piPWM.hardware_PWM(self.ttlPin, 0, 0)

    def close(self):
        self.stop()
        self.piPWM.stop()
        if self.ZMQcontrol:
            self.ZMQmessenger.close()
            self.keep_waiting_for_ZMQ_commands = False


if __name__ == '__main__':
    # Input argument handling and help info.
    parser = argparse.ArgumentParser(description='Running this script initates GlobalClock_TTL_emitter class\n' + 
                                     'only if --remote flag is provided. Class waits for ZMQ commands.')
    parser.add_argument('--remote', action='store_true', 
                        help='Expects start, stop and close commands over ZMQ.')
    parser.add_argument('--port', type=int, nargs=1, 
                        help='The port to use for ZMQ paired_messenger with Recording PC.')
    args = parser.parse_args()
    # Only run the GlobalClock_TTL_emitter as a script if --remote flag provided.
    if args.remote:
        if args.port:
            Controller = GlobalClock_TTL_emitter(ZMQcontrol=True, ZMQport=args.port[0])
        else:
            print('--port was not provided, but is required to start the Class.')
    else:
        print('--remote flag was not used. Class can not be used as script without ZMQ commands.')
