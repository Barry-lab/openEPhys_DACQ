from copy import copy, deepcopy


class GameStateDetector(object):
    """
    Helps keep track of game state in log messages

    Attributes:
        data (dict): game states and list of their active periods
        name (str): data identifier
    """
    def __init__(self):
        self._identifier = 'GameState_'
        self._data = {}
        self._current_state = None
        self._name = 'GameState'

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    def check(self, message):
        """
        Checks if message contains game state information.

        :param message: single log message
        :type message: str
        :return: network_events_data
        :rtype: bool
        """
        return self._identifier in message or 'StopRecord' in message

    @staticmethod
    def parse(message):
        """
        Returns state_name extracted from message
        and any additional data the message contained
        in a list of str.

        :param message:
        :type message: str
        :return: state_name, additiona_data
        :rtype: str, list
        """
        parts = message.split()
        state_name = parts[0][10:]
        additional_data = parts[1:]

        return state_name, additional_data

    def _init_state(self, state_name, timestamp, additional_data):
        # Inserts state list into self._data
        self._data[state_name] = {}
        self._data[state_name]['timestamps'] = [[timestamp]]
        self._data[state_name]['data'] = [additional_data]

    def _check_if_state_name_previously_observed(self, state_name):
        return state_name in self._data

    def add(self, message, timestamp):
        """
        Adds information from message and its timestamp to GameStateDetector.data

        :param message:
        :type message: str
        :param timestamp:
        :type timestamp: float
        :return:
        """
        if message == 'StopRecord':
            self._data[self._current_state]['timestamps'][-1].append(timestamp)
        else:
            state_name, additional_data = GameStateDetector.parse(message)
            if self._check_if_state_name_previously_observed(state_name):
                self._data[self._current_state]['timestamps'][-1].append(timestamp)
                self._data[state_name]['timestamps'].append([timestamp])
                self._data[state_name]['data'].append(additional_data)
                self._current_state = state_name
            else:
                self._init_state(state_name, timestamp, additional_data)
                if not (self._current_state is None):
                    self._data[self._current_state]['timestamps'][-1].append(timestamp)
                self._current_state = state_name

    def finalize_message_processing(self, final_timestamp):
        """Ensures the final GameState epoch has end timestamp

        :param float final_timestamp: the end timestamp used if not already in place
        """
        if len(self._data[self._current_state]['timestamps'][-1]) == 1:
            self._data[self._current_state]['timestamps'][-1].append(final_timestamp)


class RewardDetector(object):
    """
    Helps keep track of rewards in log messages

    Attributes:
        data (dict): game states and list of their active periods
        name (str): data identifier
    """
    def __init__(self):
        self._identifiers = ('Reward pellet',
                             'Reward milk',
                             'FEEDER pellet',
                             'FEEDER milk')
        self._data = {}
        self._name = 'Reward'

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self._name

    def check(self, message):
        """
        Checks if message contains reward information.

        :param message: single log message
        :type message: str
        :return: network_events_data
        :rtype: bool
        """
        return any([x in message for x in self._identifiers])

    @staticmethod
    def parse(message):
        """
        Extracts reward type, feeder ID and action from message.
        Action is reward quantity as str or 'inactivated' message.

        :param message:
        :type message: str
        :return: reward_type, feeder_ID, data
        :rtype: str, str, str
        """
        reward_type, feeder_id, data = message.split()[1:4]

        return reward_type, feeder_id, data

    def _check_if_reward_type_previously_observed(self, reward_type):
        return True if reward_type in self._data else False

    def _check_if_feeder_id_previously_observed(self, reward_type, feeder_id):
        return True if feeder_id in self._data[reward_type] else False

    def _create_reward_type(self, reward_type):
        self._data[reward_type] = {}

    def _create_feeder_id(self, reward_type, feeder_id):
        self._data[reward_type][feeder_id] = {'timestamps': [], 'data': []}

    def add(self, message, timestamp):
        """
        Adds information from message and its timestamp to RewardDetector.data

        :param message:
        :type message: str
        :param timestamp:
        :type timestamp: float
        :return:
        """
        reward_type, feeder_id, data = RewardDetector.parse(message)
        if not self._check_if_reward_type_previously_observed(reward_type):
            self._create_reward_type(reward_type)
        if not self._check_if_feeder_id_previously_observed(reward_type, feeder_id):
            self._create_feeder_id(reward_type, feeder_id)
        self._data[reward_type][feeder_id]['timestamps'].append(timestamp)
        self._data[reward_type][feeder_id]['data'].append(data)


class SignalDetector(object):
    """
    Helps keep track of rewards in log messages

    Attributes:
        data (dict): game states and list of their active periods
        name (str): data identifier
    """
    def __init__(self, negative_audio_signal_duration=None):
        """
        :param float negative_audio_signal_duration: required for completing NegativeAudioSignal epochs
        """

        self._identifiers = ('AudioSignal',
                             'NegativeAudioSignal',
                             'LightSignal')
        self._data = {}
        self._name = 'Signal'
        self._negative_audio_signal_duration = negative_audio_signal_duration

    @property
    def data(self):
        return self.convert_start_stop_timestamps_to_timeline()

    @property
    def name(self):
        return self._name

    @property
    def negative_audio_signal_duration(self):
        return self._negative_audio_signal_duration

    def check(self, message):
        """
        Checks if message contains reward information.

        :param message: single log message
        :type message: str
        :return: network_events_data
        :rtype: bool
        """
        return any([x in message for x in self._identifiers])

    @staticmethod
    def parse(message):
        """
        Extracts reward type, feeder ID and action from message.
        Action is reward quantity as str or 'inactivated' message.

        :param message:
        :type message: str
        :return: signal_type, data
        :rtype: str, str
        """
        signal_type, data = message.split()

        return signal_type, data

    def _check_if_signal_type_previously_observed(self, signal_type):
        return True if signal_type in self._data else False

    def _create_signal_type(self, signal_type):
        self._data[signal_type] = {'timestamps': [], 'data': []}

    def add(self, message, timestamp):
        """
        Adds information from message and its timestamp to RewardDetector.data

        :param message:
        :type message: str
        :param timestamp:
        :type timestamp: float
        :return:
        """
        signal_type, data = SignalDetector.parse(message)
        if not self._check_if_signal_type_previously_observed(signal_type):
            self._create_signal_type(signal_type)
        self._data[signal_type]['timestamps'].append(timestamp)
        self._data[signal_type]['data'].append(data)

    def convert_start_stop_timestamps_to_timeline(self):
        """
        Converts SignalDetector.data Start and Stop points into
        timeline of On periods. Works for signal_type:
        'AudioSignal' and 'LightSignal'.
        Returns a converted copy of SignalDetector.data.

        :return: data
        :rtype: dict
        """
        data = deepcopy(self._data)
        signal_candidates = ('AudioSignal', 'LightSignal')
        for signal_type in signal_candidates:
            if signal_type in data:
                epochs = []
                signal_on = False
                for timestamp, event_data in zip(data[signal_type]['timestamps'],
                                                 data[signal_type]['data']):
                    if not signal_on and event_data == 'Start':
                        epochs.append([deepcopy(timestamp)])
                        signal_on = True
                    elif signal_on and event_data == 'Stop':
                        epochs[-1].append(deepcopy(timestamp))
                        signal_on = False
                    else:
                        raise ValueError('Unexpected sequence of Start Stop for a signal')
                # Overwrite timestamps key with epochs
                data[signal_type]['timestamps'] = epochs

        # Append end timestamp to negative audio signal timestamps
        if 'NegativeAudioSignal' in data:
            if self.negative_audio_signal_duration is None:
                raise Exception(
                    'NegativeAudioSignal in messages, but negative_audio_signal_duration not provided.'
                )
            signal_type = 'NegativeAudioSignal'
            for i, timestamp in enumerate(data[signal_type]['timestamps']):
                data[signal_type]['timestamps'][i] = [timestamp, timestamp + self.negative_audio_signal_duration]

        return data


class LogParser(object):
    """
    Extracts Task related messages from list of strings
    and presents in easy to use format in class attributes.

    Attributes:
        data (dict): complete set of data detected in messages
        messages (list): task related messages
        timestamps (list): timestamps of task related messages
    """

    def __init__(self, messages, timestamps, task_settings=None, final_timestamp=None):
        """Constructor for LogParser class

        Initializes the LogParser class for a full log of a single session.

        Task relevant information is extracted and made available through query methods.

        messages   - list of str
        timestamps - list of float

        :param messages:
        :type messages: list
        :param timestamps:
        :type timestamps: list
        :param dict task_settings: output fron NWBio.load_settings()['TaskSettings']
        :param float final_timestamp: used to close any open epochs at the end of recording
        """
        # Parse input
        if not (task_settings is None) and 'NegativeAudioSignal' in task_settings:
            negative_audio_signal_duration = task_settings['NegativeAudioSignal']
        else:
            negative_audio_signal_duration = None
        # Initialize message detectors
        self.detectors = (GameStateDetector(), 
                          RewardDetector(), 
                          SignalDetector(negative_audio_signal_duration=negative_audio_signal_duration))
        # Process messages
        self._messages = []
        self._timestamps = []
        for message, timestamp in zip(messages, timestamps):
            self._process_message(message, timestamp)

        # Finalize message processing in each detector if method available
        if not (final_timestamp is None):
            for detector in self.detectors:
                if hasattr(detector, 'finalize_message_processing'):
                    detector.finalize_message_processing(final_timestamp)

    def _process_message(self, message, timestamp):
        for detector in self.detectors:
            # Check if any detector identifies the message as relevant
            if detector.check(message):
                # Add message to detector data and append to local list
                detector.add(message, timestamp)
                self._messages.append(message)
                self._timestamps.append(timestamp)
                break

    @property
    def messages(self):
        return self._messages

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def data(self):
        data = {}
        for detector in self.detectors:
            data[detector.name] = detector.data
        return data

    @staticmethod
    def extract_milk_task_performance(game_state_data):
        """
        Extracts task performance from game state data.

        :param game_state_data: LogParser.data for 'GameState'
        :type game_state_data: dict
        :param float final_timestamp: used to close any open epochs at the end of recording
        :return: task_data
        :rtype: dict
        """
        # Separate milk trial game state from others
        other_states = copy(game_state_data)
        milk_state = other_states.pop('MilkTrial')
        # Create dictionary for accumulating trial information
        task_data = {'nr': [],
                     'type': [],
                     'start_timestamp': [],
                     'end_timestamp': [],
                     'outcome': [],
                     'feeder_id': []}
        # Loop through all milk trials
        trial_counter = 0
        for timestamps, data in zip(milk_state['timestamps'], milk_state['data']):
            trial_counter += 1
            task_data['nr'].append(trial_counter)
            task_data['start_timestamp'].append(timestamps[0])
            task_data['end_timestamp'].append(timestamps[1])
            # Identify feeder id
            task_data['feeder_id'].append(data[1])
            # Identify whether presentation or repeat trial
            if len(data) > 2 and (data[2] == 'presentation_trial' or data[2] == 'first_repetition'):
                task_data['type'].append('present')
            else:
                task_data['type'].append('repeat')
            # Identify trial outcome
            for other_state, value in other_states.items():
                other_state_start_timestamps = [state_times[0] for state_times in value['timestamps']]
                if timestamps[1] in other_state_start_timestamps:
                    if other_state == 'MilkReward':
                        task_data['outcome'].append('successful')
                    else:
                        other_state_data = value['data'][other_state_start_timestamps.index(timestamps[1])]
                        task_data['outcome'].append(other_state_data[0])
                    break

        return task_data
