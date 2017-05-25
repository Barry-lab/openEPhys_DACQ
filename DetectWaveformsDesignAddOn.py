# These lines need to be added to the end of the DetectWaveformsDesign.py
# that is created from QtDesigner's .ui file. This allows clicking on the
# plots to be detected.

# By Sander Tanni, January 2017, UCL

from PyQt4.QtCore import QObject, pyqtSignal
class MouseClick(QObject):
    clicked = pyqtSignal()
    def __init__(self):
        # Initialize the MouseClick as a QObject
        QObject.__init__(self)
    def SendClick(self):
        self.clicked.emit()
class PlotWidget(PlotWidget):
    def __init__(self, parent=None):
        super(PlotWidget, self).__init__(parent=parent)
        self.mousePress = pyqtSignal()
        self.mouseClickObject = MouseClick()
    def mousePressEvent(self, ev):
        super(PlotWidget, self).mousePressEvent(ev=ev)
        self.mouseClickObject.SendClick()    
