
from PyQt4 import QtGui
from HelperFunctions import openSingleFileDialog, clearLayout
import os
import NWBio
from copy import deepcopy
from importlib import import_module

class TaskSettingsGUI(object):
    def __init__(self, parent, arena_size):
        self.parent = parent
        self.arena_size = arena_size
        # Initialize GUI window
        self.mainWindow = QtGui.QWidget()
        self.mainWindow.resize(1050, 1000)
        self.mainWindow.setWindowTitle('Task Settings')
        # Create top menu items
        self.taskSelectionList = QtGui.QListWidget()
        self.taskSelectionList.setMaximumHeight(100)
        self.taskSelectionList.setMaximumWidth(200)
        self.taskSelectionList.itemActivated.connect(self.taskSelection)
        self.loadButton = QtGui.QPushButton('Load')
        self.loadButton.setMaximumWidth(100)
        self.loadButton.clicked.connect(self.loadSettings)
        self.saveButton = QtGui.QPushButton('Save')
        self.saveButton.setMaximumWidth(100)
        self.saveButton.clicked.connect(self.saveSettings)
        self.applyButton = QtGui.QPushButton('Apply')
        self.applyButton.setMaximumWidth(100)
        self.applyButton.clicked.connect(self.applySettings)
        self.cancelButton = QtGui.QPushButton('Cancel')
        self.cancelButton.setMaximumWidth(100)
        self.cancelButton.clicked.connect(self.cancelSettings)
        top_menu_hbox = QtGui.QHBoxLayout()
        top_menu_hbox.addWidget(self.taskSelectionList)
        top_menu_vbox = QtGui.QVBoxLayout()
        top_menu_vbox.addWidget(self.loadButton)
        top_menu_vbox.addWidget(self.saveButton)
        top_menu_vbox.addWidget(self.applyButton)
        top_menu_vbox.addWidget(self.cancelButton)
        top_menu_hbox.addLayout(top_menu_vbox)
        # Populate task selection list
        tasks_dir = 'Tasks'
        for x in os.listdir(tasks_dir):
            if os.path.isdir(os.path.join(tasks_dir, x)):
                self.taskSelectionList.addItem(x)
        # Create task general menu items
        self.main_settings_layout = QtGui.QVBoxLayout()
        # Create task specific menu items
        self.further_settings_layout = QtGui.QHBoxLayout()
        # Put all boxes into main window
        left_vbox = QtGui.QVBoxLayout()
        left_vbox.addLayout(top_menu_hbox)
        left_vbox.addLayout(self.main_settings_layout)
        main_hbox = QtGui.QHBoxLayout()
        main_hbox.addLayout(left_vbox)
        main_hbox.addLayout(self.further_settings_layout)
        self.mainWindow.setLayout(main_hbox)
        # Show MainWindow
        self.mainWindow.show()

    def taskSelection(self, item):
        currentTask = str(item.text())
        self.loadTaskGUI(currentTask)

    def loadTaskGUI(self, currentTask):
        # Load the GUI for currently selected task
        if currentTask == 'Pellets_and_Rep_Milk_Task': # Temporary workaround after changes
            currentTask = 'Pellets_and_Rep_Milk'
        TaskModule = import_module('Tasks.' + currentTask + '.Task')
        clearLayout(self.main_settings_layout)
        clearLayout(self.further_settings_layout)
        self.TaskGUI = TaskModule.SettingsGUI(self.main_settings_layout, self.further_settings_layout, 
                                              self.arena_size)
        self.mainWindow.resize(self.TaskGUI.min_size[0], self.TaskGUI.min_size[1])

    def loadSettings(self, TaskSettings=None):
        if not TaskSettings:
            filename = openSingleFileDialog('load', suffix='nwb', caption='Select file to load')
            TaskSettings = NWBio.load_settings(filename, path='/TaskSettings/')
        self.loadTaskGUI(TaskSettings['name'])
        TaskSettings.pop('name')
        self.TaskGUI.importSettingsToGUI(TaskSettings)

    def saveSettings(self):
        # Grab all info from self.settings and put to TaskSettings
        # Save TaskSettings to disk using dialog box
        TaskSettings = self.TaskGUI.exportSettingsFromGUI()
        TaskSettings['name'] = str(self.taskSelectionList.currentItem().text())
        filename = openSingleFileDialog('save', suffix='nwb', caption='Save file name and location')
        NWBio.save_settings(filename, TaskSettings, path='/TaskSettings/')
        print('Settings saved.')

    def applySettings(self):
        # Grab all info from self.settings and put to TaskSettings
        # By overwriting self.TaskSettings, it should also overwrite it in RecGUI
        TaskSettings = self.TaskGUI.exportSettingsFromGUI()
        TaskSettings['name'] = str(self.taskSelectionList.currentItem().text())
        self.parent.Settings['TaskSettings'] = deepcopy(TaskSettings)
        self.mainWindow.close()

    def cancelSettings(self):
        # close the window without any further action
        self.mainWindow.close()
