
from PyQt4 import QtGui
import HelperFunctions as hfunct
import os
import NWBio
from copy import deepcopy

class TaskSettingsGUI(object):
    def __init__(self, parent=None):
        self.parent = parent
        # Initialize GUI window
        self.mainWindow = QtGui.QWidget()
        self.mainWindow.resize(950, 800)
        self.mainWindow.setWindowTitle('Task Settings')
        # Create top menu items
        self.taskSelectionList = QtGui.QListWidget()
        self.taskSelectionList.setMaximumHeight(100)
        self.taskSelectionList.setMaximumWidth(200)
        self.taskSelectionList.itemActivated.connect(self.taskSelection)
        self.loadButton = QtGui.QPushButton('Load')
        self.loadButton.clicked.connect(self.loadSettings)
        self.saveButton = QtGui.QPushButton('Save')
        self.saveButton.clicked.connect(self.saveSettings)
        self.applyButton = QtGui.QPushButton('Apply')
        self.applyButton.clicked.connect(self.applySettings)
        self.cancelButton = QtGui.QPushButton('Cancel')
        self.cancelButton.clicked.connect(self.cancelSettings)
        top_menu_vbox = QtGui.QVBoxLayout()
        top_menu_vbox.addWidget(self.taskSelectionList)
        top_menu_vbox.addWidget(self.loadButton)
        top_menu_vbox.addWidget(self.saveButton)
        top_menu_vbox.addWidget(self.applyButton)
        top_menu_vbox.addWidget(self.cancelButton)
        # Populate task selection list
        tmp_files = os.listdir('Tasks')
        if '__init__.py' in tmp_files:
            tmp_files.remove('__init__.py')
        for filename in tmp_files:
            if filename.endswith('.py'):
                self.taskSelectionList.addItem(filename[:-3])
        # Create task general menu items
        self.task_general_settings = QtGui.QWidget()
        self.task_general_settings_layout = QtGui.QGridLayout(self.task_general_settings)
        # Create task specific menu items
        self.task_specific_settings = QtGui.QWidget()
        self.task_specific_settings_layout = QtGui.QHBoxLayout(self.task_specific_settings)
        # Put all boxes into main window
        top_hbox = QtGui.QHBoxLayout()
        top_hbox.addItem(top_menu_vbox)
        top_hbox.addWidget(self.task_general_settings)
        top_vbox = QtGui.QVBoxLayout()
        top_vbox.addItem(top_hbox)
        top_vbox.addWidget(self.task_specific_settings)
        self.mainWindow.setLayout(top_vbox)
        # Show MainWindow
        self.mainWindow.show()

    def clearLayout(self, layout, keep=0):
        # This function clears the layout so that it could be regenerated
        if layout is not None:
            while layout.count() > keep:
                item = layout.takeAt(keep)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())

    def taskSelection(self, item):
        currentTask = str(item.text())
        self.loadTaskGUI(currentTask)

    def loadTaskGUI(self, currentTask):
        # Load the GUI for currently selected task
        TaskModule = hfunct.import_subdirectory_module('Tasks', currentTask)
        self.clearLayout(self.task_general_settings_layout)
        self.clearLayout(self.task_specific_settings_layout)
        self = TaskModule.SettingsGUI(self)

    def loadSettings(self, TaskSettings=None):
        if not TaskSettings:
            filename = hfunct.openSingleFileDialog('load', suffix='nwb', caption='Select file to load')
            TaskSettings = NWBio.load_settings(filename, path='/TaskSettings/')
        self.loadTaskGUI(TaskSettings['name'])
        TaskSettings.pop('name')
        self.importSettingsToGUI(self, TaskSettings)

    def saveSettings(self):
        # Grab all info from self.settings and put to TaskSettings
        # Save TaskSettings to disk using dialog box
        TaskSettings = self.exportSettingsFromGUI(self)
        TaskSettings['name'] = str(self.taskSelectionList.currentItem().text())
        filename = hfunct.openSingleFileDialog('save', suffix='nwb', caption='Save file name and location')
        NWBio.save_settings(filename, TaskSettings, path='/TaskSettings/')
        print('Settings saved.')

    def applySettings(self):
        # Grab all info from self.settings and put to TaskSettings
        # By overwriting self.TaskSettings, it should also overwrite it in RecGUI
        TaskSettings = self.exportSettingsFromGUI(self)
        TaskSettings['name'] = str(self.taskSelectionList.currentItem().text())
        self.parent.Settings['TaskSettings'] = deepcopy(TaskSettings)
        self.mainWindow.close()

    def cancelSettings(self):
        # close the window without any further action
        self.mainWindow.close()
