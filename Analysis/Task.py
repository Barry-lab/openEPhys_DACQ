# Import generic modules
import importlib
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


def load_task_name(filename):
    """
    Returns the name of the task active in the recording.

    :param filename: absolute path to NWB recording file
    :type filename: str
    :return: task_name
    :rtype: str
    """
    return NWBio.load_settings(filename, path='/TaskSettings/name')


def import_task_specific_log_parser(task_name):
    """
    Returns LogParser module for the specific task.

    :param task_name: name of the task
    :type task_name: str
    :return: TaskLogParser
    :rtype: module
    """
    return importlib.import_module('Tasks.' + task_name + '_LogParser')


def plot_milk_task_performance_by_feeder(milk_task_data_frame):
    """
    Returns LogParser module for the specific task.

    :param milk_task_data_frame: task data
    :type milk_task_data_frame: pandas.DataFrame
    :return: fig, ax
    :rtype: matplotlib.figure.Figure, array of matplotlib.axes._subplots.AxesSubplot
    """
    df = copy(milk_task_data_frame)
    # Set outcome to binary
    df.outcome.replace(to_replace=['incorrect_feeder', 'timeout'], value=0, inplace=True)
    df.outcome.replace(to_replace=['successful'], value=1, inplace=True)
    # Compute outcome as percentage and count
    df_mean = df.groupby(['type', 'feeder_id']).outcome.mean().to_frame().reset_index()
    success_rate = df_mean.outcome
    df_total = df.groupby(['type', 'feeder_id']).outcome.count().to_frame().reset_index()
    total_trials = df_total.outcome
    successful_trials = (total_trials * success_rate).map(int)
    # Append first_feeder_correct_column as 0s and 1s
    first_feeder_correct = [f_id == first_f_id for f_id, first_f_id in zip(df.feeder_id, df.first_feeder_id)]
    df['first_feeder_correct'] = list(map(int, first_feeder_correct))
    # Compute percentage of presentation trials correct based on first feeder visit
    s_present_first_success = df.loc[df.type == 'present'].groupby(['feeder_id']).first_feeder_correct.mean()
    # Append present_first_feeder_correct for all series
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
    # Prepare plotting variables
    colors = sns.color_palette(n_colors=len(pd.unique(df.feeder_id)))
    ccolors = colors + colors
    dark_ccolors = list(map(lambda x: [c * 0.5 for c in x], ccolors))
    x_labels = df_mean['type'] + df_mean['feeder_id'].map(lambda x: ', ' + str(x))
    fig, ax = plt.subplots(2, 2, sharex='col', figsize=(16, 6))
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
    # Create custom labels for both left column subplots
    n_legend_lines = int(df_mean['feeder_id'].size / 2)
    from matplotlib.lines import Line2D
    for plt_nr, color_map, label_str in zip([0, 1], [ccolors, dark_ccolors], ['successful', 'failed']):
        custom_lines = [Line2D([0], [0], color=color, lw=8) for color in color_map[:n_legend_lines]]
        legend_labels = [label_str + ' feeder ' + str(nr) for nr in df_mean['feeder_id'][:n_legend_lines]]
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
    data = NWBio.load_network_events(filename)
    TaskLogParser = import_task_specific_log_parser(load_task_name(filename))
    log_parser = TaskLogParser.LogParser(**data)
    df = pd.DataFrame(TaskLogParser.extract_milk_task_performance(log_parser.data['GameState']))
    df = df.set_index('nr')
    if full_data:
        posdata = NWBio.load_processed_tracking_data(filename)
        task_settings = NWBio.load_settings(filename, '/TaskSettings/')
        append_trial_end_closest_feeder_id(df, posdata, task_settings)
        append_trial_first_visited_feeder_id(df, posdata, task_settings)

    return df


if __name__ == '__main__':
    filename = '/media/sander/BarryL_STF1/MilkTaskTrainingData/2019-02-06_16-02-51/experiment_1.nwb'
    fig, ax = plot_milk_task_performance_by_feeder(load_milk_task_data(sys.argv[1]))
    plt.show(block=True)
