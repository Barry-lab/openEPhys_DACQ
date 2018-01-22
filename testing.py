import HelperFunctions as hfunct
import RPiInterface as rpiI
from OpenEphysInterface import SendOpenEphysSingleMessage, SubscribeToOpenEphys
from CumulativePosPlot import PosPlot
import pickle
from PyQt4 import QtGui
import sys
import RecordingManagerDesign
import threading

def MessageToOE(text):
    print(text)


class RecordingManager(QtGui.QMainWindow, RecordingManagerDesign.Ui_MainWindow):

    def __init__(self, parent=None):
        super(RecordingManager, self).__init__(parent=parent)
        self.setupUi(self)
        self.pb_start_rec.clicked.connect(lambda:self.start_rec())
        self.pb_stop_rec.clicked.connect(lambda:self.stop_rec())
        self.pb_stop_rec.setEnabled(True)

    def start_task(self, TaskSettings, TaskInputData):
        print('Starting Task')
        TaskModule = hfunct.import_subdirectory_module('Tasks', TaskSettings['name'])
        self.current_task = TaskModule.Core(TaskSettings, TaskInputData)
        self.current_task.run()

    def start_rec(self):
        self.OEmessages = SubscribeToOpenEphys(verbose=False)
        self.OEmessages.connect()

        with open('/home/room418/RecordingData/RecordingManagerData/TEMP' + '/RPi/RPiSettings.p','rb') as file:
            self.RPiSettings = pickle.load(file)
        # Initialize onlineTrackingData class
        histogramParameters = {'margins': 5, # histogram data margins in centimeters
                               'binSize': 2, # histogram binSize in centimeters
                               'speedLimit': 10}# centimeters of distance in last second to be included
        self.RPIpos = rpiI.onlineTrackingData(self.RPiSettings, HistogramParameters=histogramParameters, SynthData=True)
        self.PosPlot = PosPlot(self.RPiSettings, self.RPIpos, histogramParameters)

        TaskSettings = {'name': 'Foraging_Pellets'}
        # Put input streams in a default dictionary that can be used by task as needed
        TaskInputData = {'RPIPos': self.RPIpos, 
                         'OEmessages': self.OEmessages, 
                         'MessageToOE': MessageToOE}
        threading.Thread(target=self.start_task, args=(TaskSettings, TaskInputData)).start()

    def stop_rec(self):
        self.current_task.stop()
        self.OEmessages.disconnect()
        self.PosPlot.close()
        self.RPIpos.close()

def main():
    app = QtGui.QApplication(sys.argv)
    form = RecordingManager()
    form.show()
    app.exec_()
    
if __name__ == '__main__':
    main()