# Import generic modules
import importlib
import sys
import os
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    df['outcome'].replace(to_replace=['incorrect_feeder', 'timeout'], value=0, inplace=True)
    df['outcome'].replace(to_replace=['successful'], value=1, inplace=True)
    # Compute outcome as percentage and count
    df_mean = df.groupby(['type', 'feeder_id'])['outcome'].mean().to_frame().reset_index()
    success_rate = df_mean['outcome']
    df_total = df.groupby(['type', 'feeder_id'])['outcome'].count().to_frame().reset_index()
    total_trials = df_total['outcome']
    successful_trials = (total_trials * success_rate).map(int)
    # Prepare plotting variables
    colors = sns.color_palette(n_colors=len(pd.unique(df['feeder_id'])))
    colors = colors + colors
    dark_colors = list(map(lambda x: [c * 0.5 for c in x], colors))
    x_labels = df_mean['type'] + df_mean['feeder_id'].map(lambda x: ', ' + str(x))
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    # Plot data
    ax[0].bar(x_labels, success_rate, color=colors)
    ax[1].bar(x_labels, total_trials, color=dark_colors)
    ax[1].bar(x_labels, successful_trials, color=colors)
    # Illustrate plot
    ax[0].set_title('Milk Task Performance')
    ax[0].set_ylabel('success rate')
    ax[1].set_ylabel('count')
    ax[1].set_xlabel('type, feeder_id')
    # Create custom labels for both subplots
    n_legend_lines = int(df_mean['feeder_id'].size / 2)
    from matplotlib.lines import Line2D
    for plt_nr, color_map, label_str in zip([0, 1], [colors, dark_colors], ['successful', 'failed']):
        custom_lines = [Line2D([0], [0], color=color, lw=8) for color in color_map[:n_legend_lines]]
        legend_labels = [label_str + ' feeder ' + str(nr) for nr in df_mean['feeder_id'][:n_legend_lines]]
        ax[plt_nr].legend(custom_lines, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5))

    return fig, ax


def load_task_data(filename):
    """
    Returns data frame for task related data from NWB recording file.

    :param filename: absolute path to NWB recording file
    :type filename: str
    :return: task_data
    :rtype: pandas.DataFrame
    """
    data = NWBio.load_network_events(filename)
    TaskLogParser = import_task_specific_log_parser(load_task_name(filename))
    log_parser = TaskLogParser.LogParser(**data)
    df = pd.DataFrame(TaskLogParser.extract_milk_task_performance(log_parser.data['GameState']))
    df = df.set_index('nr')

    return df

#
# if __name__ == '__main__':
#     filename = '/media/sander/BarryL_STF1/MilkTaskTrainingData/2019-02-06_16-02-51/experiment_1.nwb'
#     fig, ax = plot_milk_task_performance_by_feeder(load_task_data(sys.argv[1]))
#     fig.show()
