
import os
from copy import deepcopy
from importlib import import_module
from PyQt5 import QtWidgets

from openEPhys_DACQ import NWBio
from openEPhys_DACQ.HelperFunctions import openSingleFileDialog, clearLayout


class TaskSettingsApp(object):

    def __init__(self, arena_size, apply_settings_method, task_settings=None):

        self._arena_size = arena_size
        self.apply_settings_method = apply_settings_method
        self._task_settings = {} if task_settings is None else task_settings
        self._task_settings['arena_size'] = arena_size

    @property
    def arena_size(self):
        return deepcopy(self._arena_size)

    @property
    def task_settings(self):
        return deepcopy(self._task_settings)

    def update_settings(self, key, value):
        self._task_settings[key] = value

    def load_settings(self, fpath):
        self._task_settings = NWBio.load_settings(fpath, path='/TaskSettings/')

    def save_settings(self, fpath):
        NWBio.save_settings(fpath, self.task_settings, path='/TaskSettings/')

    def apply(self):
        self.apply_settings_method(self.task_settings)


class TaskSettingsGUI(QtWidgets.QDialog):

    def __init__(self, task_settings_app, parent):
        """
        :param parent:
        :param TaskSettingsApp task_settings_app:
        """
        super().__init__(parent=parent)

        self.task_settings_app = task_settings_app

        self.task_specific_gui = None

        self.resize(400, 150)
        self.setWindowTitle('Task Settings')

        # Create top menu items
        self.task_selection_list_widget = QtWidgets.QListWidget(self)
        self.task_selection_list_widget.setMaximumWidth(300)
        self.task_selection_list_widget.itemSelectionChanged.connect(self.task_selection_list_callback)
        load_button = QtWidgets.QPushButton('Load')
        load_button.setMaximumWidth(100)
        load_button.clicked.connect(self.load_button_callback)
        save_button = QtWidgets.QPushButton('Save')
        save_button.setMaximumWidth(100)
        save_button.clicked.connect(self.save_button_callback)
        apply_button = QtWidgets.QPushButton('Apply')
        apply_button.setMaximumWidth(100)
        apply_button.clicked.connect(self.apply_button_callback)
        cancel_button = QtWidgets.QPushButton('Cancel')
        cancel_button.setMaximumWidth(100)
        cancel_button.clicked.connect(self.cancel_button_callback)

        self.setLayout(QtWidgets.QHBoxLayout(self))

        # Create top menu and general settings widget
        top_menu_general_settings_widget = QtWidgets.QWidget(self)
        top_menu_general_settings_layout = QtWidgets.QVBoxLayout(top_menu_general_settings_widget)
        self.layout().addWidget(top_menu_general_settings_widget)

        # Create top menu layout
        top_menu_widget = QtWidgets.QWidget(top_menu_general_settings_widget)
        top_menu_widget.setFixedSize(300, 125)
        top_menu_general_settings_layout.addWidget(top_menu_widget)
        top_menu_layout = QtWidgets.QHBoxLayout(top_menu_widget)
        top_menu_layout.addWidget(self.task_selection_list_widget)
        top_menu_buttons_widget = QtWidgets.QWidget(top_menu_widget)
        top_menu_layout.addWidget(top_menu_buttons_widget)
        top_menu_buttons_layout = QtWidgets.QVBoxLayout(top_menu_buttons_widget)
        top_menu_buttons_layout.addWidget(load_button)
        top_menu_buttons_layout.addWidget(save_button)
        top_menu_buttons_layout.addWidget(apply_button)
        top_menu_buttons_layout.addWidget(cancel_button)
        top_menu_buttons_layout.setContentsMargins(0, 0, 0, 0)
        top_menu_buttons_layout.setSpacing(2)

        # Create task general menu items
        self.general_settings_widget = QtWidgets.QWidget(top_menu_general_settings_widget)
        self.general_settings_layout = QtWidgets.QVBoxLayout(self.general_settings_widget)
        top_menu_general_settings_layout.addWidget(self.general_settings_widget)

        # Create task specific menu items
        self.further_settings_widget = QtWidgets.QWidget(self)
        self.further_settings_layout = QtWidgets.QHBoxLayout(self.further_settings_widget)
        self.layout().addWidget(self.further_settings_widget)

        # Populate task selection list
        self.task_selection_list_items = {}
        tasks_dir = os.path.join(os.path.dirname(NWBio.__file__), 'Tasks')
        for task_name in os.listdir(tasks_dir):
            if os.path.isdir(os.path.join(tasks_dir, task_name)) and not task_name.startswith('__'):
                self.task_selection_list_items[task_name] = \
                    QtWidgets.QListWidgetItem(task_name, self.task_selection_list_widget)
                self.task_selection_list_widget.addItem(self.task_selection_list_items[task_name])

        # Load task settings if any available on task_settings_app
        if 'name' in self.task_settings_app.task_settings:
            self.select_task_selection_list_item(self.task_settings_app.task_settings['name'])
            self.update_task_specific_settings_data()

        self.exec()

    def select_task_selection_list_item(self, task_name):
        self.task_selection_list_widget.setCurrentItem(self.task_selection_list_items[task_name])

    def task_selection_list_callback(self, item=None):
        if item is None:
            item = self.task_selection_list_widget.currentItem()
        current_task = str(item.text())
        self.load_task_specific_gui(current_task)

    def load_task_specific_gui(self, current_task):
        # Load the GUI for currently selected task
        task_module = import_module('Tasks.' + current_task + '.Task')
        clearLayout(self.general_settings_layout)
        clearLayout(self.further_settings_layout)
        self.task_specific_gui = task_module.SettingsGUI(self.general_settings_layout,
                                                         self.further_settings_layout,
                                                         self.task_settings_app.arena_size)
        self.resize(self.task_specific_gui.min_size[0], self.task_specific_gui.min_size[1])

    def load_button_callback(self):
        fpath = openSingleFileDialog('load', suffix='nwb', caption='Select file to load')
        self.task_settings_app.load_settings(fpath)
        self.select_task_selection_list_item(self.task_settings_app.task_settings['name'])
        self.update_task_specific_settings_data()

    def update_task_settings_app_data(self):
        task_settings = self.task_specific_gui.export_settings_from_gui()
        task_settings['name'] = str(self.task_selection_list_widget.currentItem().text())
        for key, value in task_settings.items():
            self.task_settings_app.update_settings(key, value)

    def update_task_specific_settings_data(self):
        task_settings = deepcopy(self.task_settings_app.task_settings)
        _ = task_settings.pop('name')
        self.task_specific_gui.import_settings_to_gui(task_settings)

    def save_button_callback(self):
        self.update_task_settings_app_data()
        fpath = openSingleFileDialog('save', suffix='nwb', caption='Save file name and location')
        self.task_settings_app.save_settings(fpath)

    def apply_button_callback(self):
        self.update_task_settings_app_data()
        self.task_settings_app.apply()
        self.close()

    def cancel_button_callback(self):
        self.close()
