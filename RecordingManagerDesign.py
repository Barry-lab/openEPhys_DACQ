# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'RecordingManagerDesign.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(835, 601)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.rec_settings = QtGui.QFrame(self.centralwidget)
        self.rec_settings.setFrameShape(QtGui.QFrame.StyledPanel)
        self.rec_settings.setFrameShadow(QtGui.QFrame.Raised)
        self.rec_settings.setObjectName(_fromUtf8("rec_settings"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.rec_settings)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.f_info = QtGui.QFrame(self.rec_settings)
        self.f_info.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_info.setFrameShadow(QtGui.QFrame.Raised)
        self.f_info.setObjectName(_fromUtf8("f_info"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.f_info)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.scroll_info = QtGui.QScrollArea(self.f_info)
        self.scroll_info.setMinimumSize(QtCore.QSize(370, 0))
        self.scroll_info.setWidgetResizable(True)
        self.scroll_info.setObjectName(_fromUtf8("scroll_info"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 492, 478))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.verticalLayout = QtGui.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.f_root_folder = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.f_root_folder.setMaximumSize(QtCore.QSize(16777215, 49))
        self.f_root_folder.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_root_folder.setFrameShadow(QtGui.QFrame.Raised)
        self.f_root_folder.setObjectName(_fromUtf8("f_root_folder"))
        self.horizontalLayout_21 = QtGui.QHBoxLayout(self.f_root_folder)
        self.horizontalLayout_21.setObjectName(_fromUtf8("horizontalLayout_21"))
        self.plainTextEdit_38 = QtGui.QPlainTextEdit(self.f_root_folder)
        self.plainTextEdit_38.setMaximumSize(QtCore.QSize(16777215, 29))
        self.plainTextEdit_38.setReadOnly(True)
        self.plainTextEdit_38.setObjectName(_fromUtf8("plainTextEdit_38"))
        self.horizontalLayout_21.addWidget(self.plainTextEdit_38)
        self.pb_root_folder = QtGui.QPushButton(self.f_root_folder)
        self.pb_root_folder.setMinimumSize(QtCore.QSize(85, 29))
        self.pb_root_folder.setMaximumSize(QtCore.QSize(85, 29))
        self.pb_root_folder.setObjectName(_fromUtf8("pb_root_folder"))
        self.horizontalLayout_21.addWidget(self.pb_root_folder)
        self.pt_root_folder = QtGui.QPlainTextEdit(self.f_root_folder)
        self.pt_root_folder.setMaximumSize(QtCore.QSize(16777215, 29))
        self.pt_root_folder.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_root_folder.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_root_folder.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.pt_root_folder.setObjectName(_fromUtf8("pt_root_folder"))
        self.horizontalLayout_21.addWidget(self.pt_root_folder)
        self.verticalLayout.addWidget(self.f_root_folder)
        self.f_animal = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.f_animal.setMaximumSize(QtCore.QSize(16777215, 49))
        self.f_animal.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_animal.setFrameShadow(QtGui.QFrame.Raised)
        self.f_animal.setObjectName(_fromUtf8("f_animal"))
        self.horizontalLayout_9 = QtGui.QHBoxLayout(self.f_animal)
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.plainTextEdit_14 = QtGui.QPlainTextEdit(self.f_animal)
        self.plainTextEdit_14.setMaximumSize(QtCore.QSize(16777215, 29))
        self.plainTextEdit_14.setReadOnly(True)
        self.plainTextEdit_14.setObjectName(_fromUtf8("plainTextEdit_14"))
        self.horizontalLayout_9.addWidget(self.plainTextEdit_14)
        self.pb_animal = QtGui.QPushButton(self.f_animal)
        self.pb_animal.setEnabled(False)
        self.pb_animal.setMinimumSize(QtCore.QSize(85, 29))
        self.pb_animal.setMaximumSize(QtCore.QSize(85, 29))
        self.pb_animal.setObjectName(_fromUtf8("pb_animal"))
        self.horizontalLayout_9.addWidget(self.pb_animal)
        self.pt_animal = QtGui.QPlainTextEdit(self.f_animal)
        self.pt_animal.setMaximumSize(QtCore.QSize(16777215, 29))
        self.pt_animal.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_animal.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_animal.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.pt_animal.setObjectName(_fromUtf8("pt_animal"))
        self.horizontalLayout_9.addWidget(self.pt_animal)
        self.verticalLayout.addWidget(self.f_animal)
        self.f_experiment = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.f_experiment.setMaximumSize(QtCore.QSize(16777215, 49))
        self.f_experiment.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_experiment.setFrameShadow(QtGui.QFrame.Raised)
        self.f_experiment.setObjectName(_fromUtf8("f_experiment"))
        self.horizontalLayout_19 = QtGui.QHBoxLayout(self.f_experiment)
        self.horizontalLayout_19.setObjectName(_fromUtf8("horizontalLayout_19"))
        self.plainTextEdit_34 = QtGui.QPlainTextEdit(self.f_experiment)
        self.plainTextEdit_34.setMaximumSize(QtCore.QSize(16777215, 29))
        self.plainTextEdit_34.setReadOnly(True)
        self.plainTextEdit_34.setObjectName(_fromUtf8("plainTextEdit_34"))
        self.horizontalLayout_19.addWidget(self.plainTextEdit_34)
        self.pb_experiment = QtGui.QPushButton(self.f_experiment)
        self.pb_experiment.setEnabled(False)
        self.pb_experiment.setMinimumSize(QtCore.QSize(85, 29))
        self.pb_experiment.setMaximumSize(QtCore.QSize(85, 29))
        self.pb_experiment.setObjectName(_fromUtf8("pb_experiment"))
        self.horizontalLayout_19.addWidget(self.pb_experiment)
        self.pt_experiment_id = QtGui.QPlainTextEdit(self.f_experiment)
        self.pt_experiment_id.setMaximumSize(QtCore.QSize(16777215, 29))
        self.pt_experiment_id.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_experiment_id.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_experiment_id.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.pt_experiment_id.setObjectName(_fromUtf8("pt_experiment_id"))
        self.horizontalLayout_19.addWidget(self.pt_experiment_id)
        self.verticalLayout.addWidget(self.f_experiment)
        self.f_experimenter = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.f_experimenter.setMaximumSize(QtCore.QSize(16777215, 49))
        self.f_experimenter.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_experimenter.setFrameShadow(QtGui.QFrame.Raised)
        self.f_experimenter.setObjectName(_fromUtf8("f_experimenter"))
        self.horizontalLayout_17 = QtGui.QHBoxLayout(self.f_experimenter)
        self.horizontalLayout_17.setObjectName(_fromUtf8("horizontalLayout_17"))
        self.plainTextEdit_30 = QtGui.QPlainTextEdit(self.f_experimenter)
        self.plainTextEdit_30.setMaximumSize(QtCore.QSize(16777215, 29))
        self.plainTextEdit_30.setReadOnly(True)
        self.plainTextEdit_30.setObjectName(_fromUtf8("plainTextEdit_30"))
        self.horizontalLayout_17.addWidget(self.plainTextEdit_30)
        self.pb_experimenter = QtGui.QPushButton(self.f_experimenter)
        self.pb_experimenter.setEnabled(False)
        self.pb_experimenter.setMinimumSize(QtCore.QSize(85, 29))
        self.pb_experimenter.setMaximumSize(QtCore.QSize(85, 29))
        self.pb_experimenter.setObjectName(_fromUtf8("pb_experimenter"))
        self.horizontalLayout_17.addWidget(self.pb_experimenter)
        self.pt_experimenter = QtGui.QPlainTextEdit(self.f_experimenter)
        self.pt_experimenter.setMaximumSize(QtCore.QSize(16777215, 29))
        self.pt_experimenter.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_experimenter.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_experimenter.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.pt_experimenter.setObjectName(_fromUtf8("pt_experimenter"))
        self.horizontalLayout_17.addWidget(self.pt_experimenter)
        self.verticalLayout.addWidget(self.f_experimenter)
        self.f_badChan_2 = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.f_badChan_2.setMaximumSize(QtCore.QSize(16777215, 49))
        self.f_badChan_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_badChan_2.setFrameShadow(QtGui.QFrame.Raised)
        self.f_badChan_2.setObjectName(_fromUtf8("f_badChan_2"))
        self.horizontalLayout_23 = QtGui.QHBoxLayout(self.f_badChan_2)
        self.horizontalLayout_23.setObjectName(_fromUtf8("horizontalLayout_23"))
        self.plainTextEdit_32 = QtGui.QPlainTextEdit(self.f_badChan_2)
        self.plainTextEdit_32.setMaximumSize(QtCore.QSize(16777215, 29))
        self.plainTextEdit_32.setReadOnly(True)
        self.plainTextEdit_32.setObjectName(_fromUtf8("plainTextEdit_32"))
        self.horizontalLayout_23.addWidget(self.plainTextEdit_32)
        self.frame_4 = QtGui.QFrame(self.f_badChan_2)
        self.frame_4.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_4.setObjectName(_fromUtf8("frame_4"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout(self.frame_4)
        self.horizontalLayout_5.setMargin(0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.pt_chan_map_1 = QtGui.QPlainTextEdit(self.frame_4)
        self.pt_chan_map_1.setMinimumSize(QtCore.QSize(50, 29))
        self.pt_chan_map_1.setObjectName(_fromUtf8("pt_chan_map_1"))
        self.horizontalLayout_5.addWidget(self.pt_chan_map_1)
        self.pt_chan_map_1_chans = QtGui.QPlainTextEdit(self.frame_4)
        self.pt_chan_map_1_chans.setMinimumSize(QtCore.QSize(70, 29))
        self.pt_chan_map_1_chans.setObjectName(_fromUtf8("pt_chan_map_1_chans"))
        self.horizontalLayout_5.addWidget(self.pt_chan_map_1_chans)
        self.pt_chan_map_2 = QtGui.QPlainTextEdit(self.frame_4)
        self.pt_chan_map_2.setMinimumSize(QtCore.QSize(50, 29))
        self.pt_chan_map_2.setObjectName(_fromUtf8("pt_chan_map_2"))
        self.horizontalLayout_5.addWidget(self.pt_chan_map_2)
        self.pt_chan_map_2_chans = QtGui.QPlainTextEdit(self.frame_4)
        self.pt_chan_map_2_chans.setMinimumSize(QtCore.QSize(70, 29))
        self.pt_chan_map_2_chans.setObjectName(_fromUtf8("pt_chan_map_2_chans"))
        self.horizontalLayout_5.addWidget(self.pt_chan_map_2_chans)
        self.horizontalLayout_23.addWidget(self.frame_4)
        self.horizontalLayout_23.setStretch(0, 2)
        self.horizontalLayout_23.setStretch(1, 3)
        self.verticalLayout.addWidget(self.f_badChan_2)
        self.f_badChan = QtGui.QFrame(self.scrollAreaWidgetContents)
        self.f_badChan.setMaximumSize(QtCore.QSize(16777215, 49))
        self.f_badChan.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_badChan.setFrameShadow(QtGui.QFrame.Raised)
        self.f_badChan.setObjectName(_fromUtf8("f_badChan"))
        self.horizontalLayout_22 = QtGui.QHBoxLayout(self.f_badChan)
        self.horizontalLayout_22.setObjectName(_fromUtf8("horizontalLayout_22"))
        self.plainTextEdit_31 = QtGui.QPlainTextEdit(self.f_badChan)
        self.plainTextEdit_31.setMaximumSize(QtCore.QSize(16777215, 29))
        self.plainTextEdit_31.setReadOnly(True)
        self.plainTextEdit_31.setObjectName(_fromUtf8("plainTextEdit_31"))
        self.horizontalLayout_22.addWidget(self.plainTextEdit_31)
        self.pt_badChan = QtGui.QPlainTextEdit(self.f_badChan)
        self.pt_badChan.setMaximumSize(QtCore.QSize(16777215, 29))
        self.pt_badChan.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_badChan.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_badChan.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.pt_badChan.setPlainText(_fromUtf8(""))
        self.pt_badChan.setObjectName(_fromUtf8("pt_badChan"))
        self.horizontalLayout_22.addWidget(self.pt_badChan)
        self.horizontalLayout_22.setStretch(0, 1)
        self.horizontalLayout_22.setStretch(1, 1)
        self.verticalLayout.addWidget(self.f_badChan)
        self.scroll_info.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_3.addWidget(self.scroll_info)
        self.horizontalLayout.addWidget(self.f_info)
        self.f_buttons = QtGui.QFrame(self.rec_settings)
        self.f_buttons.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_buttons.setFrameShadow(QtGui.QFrame.Raised)
        self.f_buttons.setObjectName(_fromUtf8("f_buttons"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.f_buttons)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.pb_load_last = QtGui.QPushButton(self.f_buttons)
        self.pb_load_last.setObjectName(_fromUtf8("pb_load_last"))
        self.verticalLayout_2.addWidget(self.pb_load_last)
        self.pb_load = QtGui.QPushButton(self.f_buttons)
        self.pb_load.setObjectName(_fromUtf8("pb_load"))
        self.verticalLayout_2.addWidget(self.pb_load)
        self.pb_cam_set = QtGui.QPushButton(self.f_buttons)
        self.pb_cam_set.setObjectName(_fromUtf8("pb_cam_set"))
        self.verticalLayout_2.addWidget(self.pb_cam_set)
        self.f_rec_folder = QtGui.QFrame(self.f_buttons)
        self.f_rec_folder.setMaximumSize(QtCore.QSize(16777215, 140))
        self.f_rec_folder.setFrameShape(QtGui.QFrame.StyledPanel)
        self.f_rec_folder.setFrameShadow(QtGui.QFrame.Raised)
        self.f_rec_folder.setObjectName(_fromUtf8("f_rec_folder"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.f_rec_folder)
        self.verticalLayout_4.setMargin(0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.plainTextEdit = QtGui.QPlainTextEdit(self.f_rec_folder)
        self.plainTextEdit.setMaximumSize(QtCore.QSize(16777215, 29))
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setObjectName(_fromUtf8("plainTextEdit"))
        self.verticalLayout_4.addWidget(self.plainTextEdit)
        self.pt_rec_folder = QtGui.QPlainTextEdit(self.f_rec_folder)
        self.pt_rec_folder.setMinimumSize(QtCore.QSize(250, 0))
        self.pt_rec_folder.setMaximumSize(QtCore.QSize(16777215, 90))
        self.pt_rec_folder.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_rec_folder.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pt_rec_folder.setLineWrapMode(QtGui.QPlainTextEdit.WidgetWidth)
        self.pt_rec_folder.setReadOnly(True)
        self.pt_rec_folder.setObjectName(_fromUtf8("pt_rec_folder"))
        self.verticalLayout_4.addWidget(self.pt_rec_folder)
        self.verticalLayout_2.addWidget(self.f_rec_folder)
        self.frame_2 = QtGui.QFrame(self.f_buttons)
        self.frame_2.setMaximumSize(QtCore.QSize(16777215, 40))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.frame_2)
        self.horizontalLayout_3.setMargin(3)
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.plainTextEdit_2 = QtGui.QPlainTextEdit(self.frame_2)
        self.plainTextEdit_2.setMaximumSize(QtCore.QSize(16777215, 29))
        self.plainTextEdit_2.setReadOnly(True)
        self.plainTextEdit_2.setObjectName(_fromUtf8("plainTextEdit_2"))
        self.horizontalLayout_3.addWidget(self.plainTextEdit_2)
        self.frame_3 = QtGui.QFrame(self.frame_2)
        self.frame_3.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_3.setObjectName(_fromUtf8("frame_3"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.frame_3)
        self.horizontalLayout_4.setMargin(0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.rb_posPlot_yes = QtGui.QRadioButton(self.frame_3)
        self.rb_posPlot_yes.setChecked(True)
        self.rb_posPlot_yes.setObjectName(_fromUtf8("rb_posPlot_yes"))
        self.buttonGroup = QtGui.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName(_fromUtf8("buttonGroup"))
        self.buttonGroup.addButton(self.rb_posPlot_yes)
        self.horizontalLayout_4.addWidget(self.rb_posPlot_yes)
        self.rb_posPlot_no = QtGui.QRadioButton(self.frame_3)
        self.rb_posPlot_no.setObjectName(_fromUtf8("rb_posPlot_no"))
        self.buttonGroup.addButton(self.rb_posPlot_no)
        self.horizontalLayout_4.addWidget(self.rb_posPlot_no)
        self.horizontalLayout_3.addWidget(self.frame_3)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 1)
        self.verticalLayout_2.addWidget(self.frame_2)
        self.pb_start_rec = QtGui.QPushButton(self.f_buttons)
        self.pb_start_rec.setEnabled(True)
        self.pb_start_rec.setObjectName(_fromUtf8("pb_start_rec"))
        self.verticalLayout_2.addWidget(self.pb_start_rec)
        self.pb_stop_rec = QtGui.QPushButton(self.f_buttons)
        self.pb_stop_rec.setEnabled(False)
        self.pb_stop_rec.setObjectName(_fromUtf8("pb_stop_rec"))
        self.verticalLayout_2.addWidget(self.pb_stop_rec)
        self.pb_process_data = QtGui.QPushButton(self.f_buttons)
        self.pb_process_data.setEnabled(False)
        self.pb_process_data.setObjectName(_fromUtf8("pb_process_data"))
        self.verticalLayout_2.addWidget(self.pb_process_data)
        self.pb_sync_server = QtGui.QPushButton(self.f_buttons)
        self.pb_sync_server.setEnabled(False)
        self.pb_sync_server.setObjectName(_fromUtf8("pb_sync_server"))
        self.verticalLayout_2.addWidget(self.pb_sync_server)
        self.pb_open_rec_folder = QtGui.QPushButton(self.f_buttons)
        self.pb_open_rec_folder.setEnabled(False)
        self.pb_open_rec_folder.setObjectName(_fromUtf8("pb_open_rec_folder"))
        self.verticalLayout_2.addWidget(self.pb_open_rec_folder)
        self.horizontalLayout.addWidget(self.f_buttons)
        self.verticalLayout_5.addWidget(self.rec_settings)
        self.signals = QtGui.QFrame(self.centralwidget)
        self.signals.setFrameShape(QtGui.QFrame.StyledPanel)
        self.signals.setFrameShadow(QtGui.QFrame.Raised)
        self.signals.setObjectName(_fromUtf8("signals"))
        self.verticalLayout_5.addWidget(self.signals)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 835, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Open Ephys Recording Manager", None))
        self.plainTextEdit_38.setPlainText(_translate("MainWindow", "Root Folder", None))
        self.pb_root_folder.setText(_translate("MainWindow", "Browse", None))
        self.plainTextEdit_14.setPlainText(_translate("MainWindow", "Animal ID", None))
        self.pb_animal.setText(_translate("MainWindow", "Options", None))
        self.plainTextEdit_34.setPlainText(_translate("MainWindow", "Experiment ID", None))
        self.pb_experiment.setText(_translate("MainWindow", "Options", None))
        self.plainTextEdit_30.setPlainText(_translate("MainWindow", "Experimenter Name", None))
        self.pb_experimenter.setText(_translate("MainWindow", "Options", None))
        self.pt_experimenter.setPlainText(_translate("MainWindow", "sander", None))
        self.plainTextEdit_32.setPlainText(_translate("MainWindow", "Channel mapping", None))
        self.pt_chan_map_1.setPlainText(_translate("MainWindow", "MEC", None))
        self.pt_chan_map_1_chans.setPlainText(_translate("MainWindow", "1-64", None))
        self.pt_chan_map_2.setPlainText(_translate("MainWindow", "CA1", None))
        self.pt_chan_map_2_chans.setPlainText(_translate("MainWindow", "65-128", None))
        self.plainTextEdit_31.setPlainText(_translate("MainWindow", "Bad Channels", None))
        self.pb_load_last.setText(_translate("MainWindow", "Load Last", None))
        self.pb_load.setText(_translate("MainWindow", "Load Settings", None))
        self.pb_cam_set.setText(_translate("MainWindow", "Camera Settings", None))
        self.plainTextEdit.setPlainText(_translate("MainWindow", "Current Recording Folder:", None))
        self.plainTextEdit_2.setPlainText(_translate("MainWindow", "Position Plot:", None))
        self.rb_posPlot_yes.setText(_translate("MainWindow", "Yes", None))
        self.rb_posPlot_no.setText(_translate("MainWindow", "No", None))
        self.pb_start_rec.setText(_translate("MainWindow", "Start Recording", None))
        self.pb_stop_rec.setText(_translate("MainWindow", "Stop Recording", None))
        self.pb_process_data.setText(_translate("MainWindow", "Process Data", None))
        self.pb_sync_server.setText(_translate("MainWindow", "Synch with QNAP", None))
        self.pb_open_rec_folder.setText(_translate("MainWindow", "Open Recording Folder", None))

