# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ApplyKlustakwikGUIDesign.ui'
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
        MainWindow.resize(392, 498)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setMinimumSize(QtCore.QSize(188, 433))
        self.frame.setMaximumSize(QtCore.QSize(188, 600))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayout = QtGui.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.pb_load_specific = QtGui.QPushButton(self.frame)
        self.pb_load_specific.setMinimumSize(QtCore.QSize(0, 29))
        self.pb_load_specific.setObjectName(_fromUtf8("pb_load_specific"))
        self.verticalLayout.addWidget(self.pb_load_specific)
        self.pb_load_all = QtGui.QPushButton(self.frame)
        self.pb_load_all.setMinimumSize(QtCore.QSize(0, 29))
        self.pb_load_all.setObjectName(_fromUtf8("pb_load_all"))
        self.verticalLayout.addWidget(self.pb_load_all)
        self.tb_fpath = QtGui.QPlainTextEdit(self.frame)
        self.tb_fpath.setMinimumSize(QtCore.QSize(168, 60))
        self.tb_fpath.setMaximumSize(QtCore.QSize(16777215, 300))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tb_fpath.setFont(font)
        self.tb_fpath.setReadOnly(True)
        self.tb_fpath.setObjectName(_fromUtf8("tb_fpath"))
        self.verticalLayout.addWidget(self.tb_fpath)
        self.tb_files = QtGui.QPlainTextEdit(self.frame)
        self.tb_files.setMinimumSize(QtCore.QSize(168, 61))
        self.tb_files.setMaximumSize(QtCore.QSize(16777215, 300))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.tb_files.setFont(font)
        self.tb_files.setReadOnly(True)
        self.tb_files.setObjectName(_fromUtf8("tb_files"))
        self.verticalLayout.addWidget(self.tb_files)
        self.pb_cluster_selection = QtGui.QPushButton(self.frame)
        self.pb_cluster_selection.setMinimumSize(QtCore.QSize(0, 29))
        self.pb_cluster_selection.setObjectName(_fromUtf8("pb_cluster_selection"))
        self.verticalLayout.addWidget(self.pb_cluster_selection)
        self.pb_cluster_all = QtGui.QPushButton(self.frame)
        self.pb_cluster_all.setMinimumSize(QtCore.QSize(0, 29))
        self.pb_cluster_all.setObjectName(_fromUtf8("pb_cluster_all"))
        self.verticalLayout.addWidget(self.pb_cluster_all)
        self.pb_waveformGUI = QtGui.QPushButton(self.frame)
        self.pb_waveformGUI.setObjectName(_fromUtf8("pb_waveformGUI"))
        self.verticalLayout.addWidget(self.pb_waveformGUI)
        self.horizontalLayout.addWidget(self.frame)
        self.lw_tetrodes = QtGui.QListWidget(self.centralwidget)
        self.lw_tetrodes.setMinimumSize(QtCore.QSize(180, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lw_tetrodes.setFont(font)
        self.lw_tetrodes.setObjectName(_fromUtf8("lw_tetrodes"))
        self.horizontalLayout.addWidget(self.lw_tetrodes)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 392, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.pb_load_specific.setText(_translate("MainWindow", "Load specifc tetrodes", None))
        self.pb_load_all.setText(_translate("MainWindow", "Load entire folder", None))
        self.pb_cluster_selection.setText(_translate("MainWindow", "Cluster selection", None))
        self.pb_cluster_all.setText(_translate("MainWindow", "Cluster all tetrodes", None))
        self.pb_waveformGUI.setText(_translate("MainWindow", "WaveformGUI format", None))

