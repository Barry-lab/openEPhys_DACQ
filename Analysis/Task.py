# Import generic modules
import sys
import os
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import euclidean
# Ensure repository root directory is on path
from os.path import dirname as up
sys.path.insert(0, up(up(os.path.abspath(__file__))))
# Import repository specific modules
import NWBio


def compute_milk_task_performance_by_feeder(milk_task_data_frame):
    """
    Returns LogParser module for the specific task.

    :param milk_task_data_frame: output from :py:func:`Task.load_milk_task_data`
    :type milk_task_data_frame: pandas.DataFrame
    :return: milk_task_performance_data
    :rtype: dict
    """
    df = copy(milk_task_data_frame)
    # Set outcome to binary
    df.outcome.replace(to_replace=['incorrect_feeder', 'timeout'], value=0, inplace=True)
    df.outcome.replace(to_replace=['successful'], value=1, inplace=True)
    # Compute outcome as percentage and count
    df_mean = df.groupby(['type', 'feeder_id']).outcome.mean().to_frame().reset_index()
    df_total = df.groupby(['type', 'feeder_id']).outcome.count().to_frame().reset_index()
    # Append first_feeder_correct_column as 0s and 1s
    first_feeder_correct = [f_id == first_f_id for f_id, first_f_id in zip(df.feeder_id, df.first_feeder_id)]
    df['first_feeder_correct'] = list(map(int, first_feeder_correct))
    # Compute percentage of presentation trials correct based on first feeder visit
    s_present_first_success = df.loc[df.type == 'present'].groupby(['feeder_id']).first_feeder_correct.mean()
    # Append preceding present_first_feeder_correct for all trials
    present_first_feeder_correct = []
    last_presentation_first_feeder_correct = False
    for trial_type, first_correct in zip(df.type, df.first_feeder_correct):
        if trial_type == 'present':
            last_presentation_first_feeder_correct = bool(first_correct)
        present_first_feeder_correct.append(last_presentation_first_feeder_correct)
    df['present_first_feeder_correct'] = present_first_feeder_correct
    # Calculate repeat trial success rate where presentation trial was successful at first feeder
    df_repeat_present_first_success = df.loc[(df.type == 'repeat') & df.present_first_feeder_correct]
    s_present_first_success_repeat_success = df_repeat_present_first_success.groupby(['feeder_id']).outcome.mean()

    return {'df_mean': df_mean,
            'df_total': df_total,
            's_present_first_success': s_present_first_success,
            's_present_first_success_repeat_success': s_present_first_success_repeat_success}


def extract_milk_task_outcome_with_delay(data):
    """
    Returns DataFrame with delay from previous trial and outcome of the trial,
    for trials after successful trial to the same feeder.

    :param milk_task_data_frame: output from :py:func:`Task.load_milk_task_data`
    :type milk_task_data_frame: pandas.DataFrame
    :return: trial_outcome_with_delay
    :rtype: pandas.DataFrame
    """
    data = data.reset_index(drop=True)
    repeat_trials = np.diff(list(map(int, data['feeder_id'].values))) == 0
    repeat_trials = np.where(np.concatenate(([False], repeat_trials)))[0]
    delays = []
    outcomes = []
    feeder_ids = []
    for trial in repeat_trials:
        if data.loc[trial - 1, 'outcome'] == 'successful':
            dur = data.loc[trial, 'start_timestamp'] - data.loc[trial - 1, 'end_timestamp']
            delays.append(dur)
            outcomes.append(data.loc[trial, 'outcome'])
            feeder_ids.append(data.loc[trial, 'feeder_id'])

    return pd.DataFrame({'delay': delays, 'outcome': outcomes, 'feeder_id': feeder_ids})


def compute_milk_task_outcome_by_delay_and_feeder_id(data, bin_edges):
    """
    Computes the outcome for groups by delay and feeder_id

    :param data: output from :py:func:`Task.compute_milk_task_performance_by_delay`
    :type milk_task_data_frame: pandas.DataFrame
    :param bin_edges: bin edges for grouping delay
    :type list
    :return: milk_task_outcome_by_delay_and_feeder
    :rtype: pandas.Series
    """
    data['delay_range'] = [''] * data.shape[0]
    for nbin in range(len(bin_edges) - 1):
        idx = data[(data['delay'] >= bin_edges[nbin]) & (data['delay'] < bin_edges[nbin + 1])].index
        data.loc[idx, 'delay_range'] = '{}-{}'.format(bin_edges[nbin], bin_edges[nbin + 1])
    data['delay'] = data['delay_range']
    data.drop(columns=['delay_range'], inplace=True)
    data['delay'].replace('', np.nan, inplace=True)
    data.dropna(axis=0, inplace=True)
    # data['outcome'].replace({'successful': 1, 'incorrect_feeder': 0}, inplace=True)

    return data




def plot_milk_task_performance_by_delay(data, ax):
    """
    Plots the task performance relative to delay in the provided axes.

    :param data: output from :py:func:`Task.compute_milk_task_performance_by_delay`
    :param ax: matplotlib axes to plot the data
    :return: None
    """




def plot_milk_task_performance_by_feeder(data):
    """
    Returns figure and plot for the provided milk task performance data.

    :param data: output from :py:func:`Task.compute_milk_task_performance_by_feeder`
    :type data: dict
    :return: fig, ax
    :rtype: matplotlib.figure.Figure, array of matplotlib.axes._subplots.AxesSubplot
    """
    # Extract plotting from input data
    success_rate = data['df_mean'].outcome
    total_trials = data['df_total'].outcome
    successful_trials = (total_trials * success_rate).map(int)
    s_present_first_success = data['s_present_first_success']
    s_present_first_success_repeat_success = data['s_present_first_success_repeat_success']
    # Prepare plotting variables
    colors = sns.color_palette(n_colors=len(pd.unique(data['df_mean'].feeder_id)))
    ccolors = colors + colors
    dark_ccolors = list(map(lambda x: [c * 0.5 for c in x], ccolors))
    x_labels = data['df_mean']['type'] + data['df_mean']['feeder_id'].map(lambda x: ', ' + str(x))
    fig, ax = plt.subplots(2, 2, sharex='col', figsize=(16, 9))
    plt.subplots_adjust(wspace=0.5)
    # Plot data
    ax[0][0].bar(x_labels, success_rate, color=ccolors)
    ax[1][0].bar(x_labels, total_trials, color=dark_ccolors)
    ax[1][0].bar(x_labels, successful_trials, color=ccolors)
    ax[0][1].bar(s_present_first_success.index, s_present_first_success, color=colors)
    ax[1][1].bar(s_present_first_success_repeat_success.index, s_present_first_success_repeat_success, color=colors)
    # Illustrate plot
    ax[0][0].set_title('Milk Task Performance')
    ax[0][1].set_title('Presentation first feeder accuracy')
    ax[1][1].set_title('Repeat accuracy when presentation correct at first feeder')
    ax[0][0].set_ylabel('success rate')
    ax[1][0].set_ylabel('count')
    ax[1][0].set_xlabel('type, feeder_id')
    ax[0][1].set_ylabel('success rate')
    ax[1][1].set_ylabel('success rate')
    ax[1][1].set_xlabel('feeder_id')
    # Set axes limits
    ax[0][0].set_ylim([0, 1])
    ax[0][1].set_ylim([0, 1])
    ax[1][1].set_ylim([0, 1])
    # Set y ticks on success rate plots
    ax[0][0].set_yticks(np.arange(0, 1.1, 0.1))
    ax[0][1].set_yticks(np.arange(0, 1.1, 0.1))
    ax[1][1].set_yticks(np.arange(0, 1.1, 0.1))
    # Draw horizontal line at 0.5 success rate on all the success rate plots
    ax[0][0].plot(np.array(ax[0][0].get_xlim()), np.array([0.5, 0.5]), '--r')
    ax[0][1].plot(np.array(ax[0][1].get_xlim()), np.array([0.5, 0.5]), '--r')
    ax[1][1].plot(np.array(ax[1][1].get_xlim()), np.array([0.5, 0.5]), '--r')
    # Create custom labels for both left column subplots
    n_legend_lines = int(data['df_mean']['feeder_id'].size / 2)
    from matplotlib.lines import Line2D
    for plt_nr, color_map, label_str in zip([0, 1], [ccolors, dark_ccolors], ['successful', 'failed']):
        custom_lines = [Line2D([0], [0], color=color, lw=8) for color in color_map[:n_legend_lines]]
        legend_labels = [label_str + ' feeder ' + str(nr) for nr in data['df_mean'].feeder_id[:n_legend_lines]]
        ax[plt_nr][0].legend(custom_lines, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    return fig, ax


def append_trial_end_closest_feeder_id(task_data, posdata, task_settings,
                                       column_name='closest_feeder_id'):
    """
    Appends a column to DataFrame that contains the identities of the
    closest feeders at the end of the each trial.

    :param task_data: output from :py:func:`Task.load_milk_task_data`
    :type task_data: pandas.DataFrame
    :param posdata: output from :py:func:`NWBio.load_processed_tracking_data`
    :type posdata: numpy.ndarray
    :param task_settings: TaskSettings as returned by :py:func:`NWBio.load_settings`
    :type task_settings: dict
    :param column_name: name of the appended column ('closest_feeder_id' by default)
    :type column_name: str
    """
    # Parse input posdata
    pos_timestamps = posdata[:, 0]
    pos_xy = posdata[:, 1:3]
    # Parse input task_settings
    feeder_ids = sorted(task_settings['FEEDERs']['milk'].keys())
    feeder_locs = [task_settings['FEEDERs']['milk'][x]['Position'] for x in feeder_ids]
    # Find positions at the end of each trial
    trial_end_positions = []
    for timestamp in task_data['end_timestamp']:
        best_match_pos_timestamp = np.argmin(np.abs(pos_timestamps - timestamp))
        trial_end_positions.append(pos_xy[best_match_pos_timestamp, :])
    # Find closest feeder for each trial end position
    closest_feeder_id = []
    for position in trial_end_positions:
        closest_feeder_id.append(
            feeder_ids[int(np.argmin([euclidean(position, loc) for loc in feeder_locs]))]
        )
    # Append closest_feeder_id to DataFrame
    task_data[column_name] = closest_feeder_id


def append_trial_first_visited_feeder_id(task_data, posdata, task_settings,
                                         column_name='first_feeder_id'):
    """
    Appends a column to DataFrame that contains the identities of the
    first feeders in each trial.

    :param task_data: output from :py:func:`Task.load_milk_task_data`
    :type task_data: pandas.DataFrame
    :param posdata: output from :py:func:`NWBio.load_processed_tracking_data`
    :type posdata: numpy.ndarray
    :param task_settings: TaskSettings as returned by :py:func:`NWBio.load_settings`
    :type task_settings: dict
    :param column_name: name of the appended column ('first_feeder_id' by default)
    :type column_name: str
    """
    # Parse input posdata
    pos_timestamps = posdata[:, 0]
    pos_xy = posdata[:, 1:3]
    # Parse input task_settings
    feeder_ids = sorted(task_settings['FEEDERs']['milk'].keys())
    feeder_locs = [task_settings['FEEDERs']['milk'][x]['Position'] for x in feeder_ids]
    min_distance = task_settings['MilkTaskMinGoalDistance']
    # Find positions series for each trial
    trial_positions = []
    for trial_start_t, trial_end_t in zip(task_data['start_timestamp'], task_data['end_timestamp']):
        start_position_ind = np.argmin(np.abs(pos_timestamps - trial_start_t))
        end_position_ind = np.argmin(np.abs(pos_timestamps - trial_end_t))
        trial_positions.append(pos_xy[start_position_ind:end_position_ind, :])
    # Find closest feeder for each trial end position
    first_feeder_id = []
    successful = False
    for positions in trial_positions:
        for npos in range(positions.shape[0]):
            feeder_distances = [euclidean(positions[npos, :], feeder_loc) for feeder_loc in feeder_locs]
            if any(np.array(feeder_distances) < min_distance):
                first_feeder_id.append(feeder_ids[int(np.argmin(feeder_distances))])
                successful = True
                break
        if successful:
            successful = False
        else:
            first_feeder_id.append('')
    # Append closest_feeder_id to DataFrame
    task_data[column_name] = first_feeder_id


def append_trial_if_light_on(task_data, light_signal_epochs,
                             column_name='light_on', max_misalignment=0.5):
    """
    Appends a column to DataFrame with booleans specifying if light was on in the trial.

    :param task_data: output from :py:func:`Task.load_milk_task_data`
    :type task_data: pandas.DataFrame
    :param light_signal_epochs: list of epoch timestamps [start, end]
    :type light_signal_epochs: list
    :param column_name: name of the appended column ('first_feeder_id' by default)
    :type column_name: str
    :param max_misalignment: maximum seconds light may precede trial start time (default is 0.5)
    :type max_misalignment: float
    """
    # Get light signal start times
    light_signal_start_times = np.array([x[0] for x in light_signal_epochs])
    # Check which trials epoch times contain light signal start time
    light_on = []
    for i, trial_epoch in enumerate(zip(task_data['start_timestamp'], task_data['end_timestamp'])):
        trial_start, trial_end = trial_epoch
        trial_start -= max_misalignment
        # Identify closest light epoch after trial start
        closest_light_epoch = np.argmin(abs(light_signal_start_times - trial_start))
        if light_signal_start_times[closest_light_epoch] < trial_start:
            closest_light_epoch += 1
        if closest_light_epoch < len(light_signal_start_times) and \
                light_signal_start_times[closest_light_epoch] < trial_end:
            light_on.append(True)
        else:
            light_on.append(False)
    # Append light signal boolean array to DataFrame
    task_data[column_name] = np.array(light_on, dtype=np.bool)


def load_milk_task_data(filename, full_data=True):
    """
    Returns data frame for milk task related data from NWB recording file.

    :param filename: absolute path to NWB recording file
    :type filename: str
    :param full_data: (optional) appends all additional columns
    :type full_data: bool
    :return: task_data
    :rtype: pandas.DataFrame
    """
    # Load milk task data with task specific LogParser
    log_parser = NWBio.get_recording_log_parser(filename)
    df = pd.DataFrame(log_parser.extract_milk_task_performance(log_parser.data['GameState']))
    df = df.set_index('nr')
    # If full_data requested, append all additional columns
    if full_data:
        posdata = NWBio.load_processed_tracking_data(filename)
        task_settings = NWBio.load_settings(filename, '/TaskSettings/')
        append_trial_end_closest_feeder_id(df, posdata, task_settings)
        append_trial_first_visited_feeder_id(df, posdata, task_settings)
        append_trial_if_light_on(df, log_parser.data['Signal']['LightSignal']['timestamps'])

    return df


if __name__ == '__main__':
    filename = sys.argv[1]
    recording_info = NWBio.extract_recording_info(
        filename, {'Time': None, 'General': {'animal': None, 'experiment_id': None}})
    figure_title = \
        recording_info['General']['animal'] + ' ' \
        + recording_info['Time'] + ' ' \
        + recording_info['General']['experiment_id']
    df = load_milk_task_data(filename)
    if sum(df['light_on']) > 0:
        milk_task_performance_data = compute_milk_task_performance_by_feeder(df.loc[df['light_on']])
        fig, ax = plot_milk_task_performance_by_feeder(milk_task_performance_data)
        fig.suptitle(figure_title + ' ' + 'Light On', fontsize=16)
        plt.show(block=True)
    if sum(~df['light_on']) > 0:
        milk_task_performance_data = compute_milk_task_performance_by_feeder(df.loc[~df['light_on']])
        fig, ax = plot_milk_task_performance_by_feeder(milk_task_performance_data)
        fig.suptitle(figure_title + ' ' + 'Light Off', fontsize=16)
        plt.show(block=True)
