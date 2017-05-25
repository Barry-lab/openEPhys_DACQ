### This script produces a window where it plots current tracking position
### and also history of tracked position.

### Usage:
### from CumulativePosPlot import PosPlot
### PosPlotApp = PosPlot(RPiSettings)
### PosPlotApp.window.show()

### By Sander Tanni, May 2017, UCL

import pyqtgraph as pg
from PyQt4 import QtGui
import numpy as np
from PyQt4.QtCore import QTimer
import RPiInterface as rpiI
import time

def angle_clockwise(p1, p2):
    # Takes two numpy 2D vectors and computes the angle of their difference
    # In other words, computes the angle of a vector from p1 to p2
    p0 = np.array(p2) - np.array(p1)
    angle = np.degrees(np.arctan2(p0[0], p0[1]))
    if angle < 0:
        angle = 360 + angle
    return angle

class PosPlot(object):
    def __init__(self, RPiSettings):
        self.RPiSettings = RPiSettings
        # Initialize plot window
        self.window = QtGui.QMainWindow()
        XandYratio = np.float32(self.RPiSettings['arena_size'][0]) / np.float32(self.RPiSettings['arena_size'][1])
        self.window.resize(int(500 * XandYratio), 500)
        self.plotWidget = pg.PlotWidget(self.window)
        # Set plot axes ranges
        margin_x = self.RPiSettings['arena_size'][0] * 0.05
        margin_y = self.RPiSettings['arena_size'][1] * 0.05
        self.plotWidget.setXRange(-margin_x, self.RPiSettings['arena_size'][0] + margin_x)
        self.plotWidget.setYRange(-margin_y, self.RPiSettings['arena_size'][1] + margin_y)
        # Invert Y-axis #### This messes up the arrow direction
        # self.plotWidget.invertY(True)
        # Put plot into plot window
        self.window.setCentralWidget(self.plotWidget)
        # Draw arena boundaries
        self.draw_arena_boundaries()
        # Prepare arrow item for showing head direction
        if self.RPiSettings['LEDmode'] == 'double':
            self.arrow = pg.ArrowItem(pos=(-10,-10), angle=45, headLen=30, tailLen=15,headWidth=30,tailWidth=5)
            self.plotWidget.addItem(self.arrow)
        # Initialize local position data monitoring
        self.RPIpos = rpiI.latestPosData(self.RPiSettings)
        self.lastPosition = np.array([None, None],dtype=np.float32)
        while np.any(np.isnan(self.lastPosition)): # Only continue once position data is updated
            time.sleep(0.1)
            linedatas = self.RPIpos.linedatas # Retrieve latest position data
            self.lastPosition, _ = rpiI.computeAbsolutePosition(linedatas, self.RPiSettings)
        # Start constant update of the plot
        plotUpdateInterval = 500 # This sets the plot update interval in milliseconds
        self.cumulativePlot_timer = QTimer()
        self.cumulativePlot_timer.timeout.connect(lambda:self.updatePlot())
        self.cumulativePlot_timer.start(plotUpdateInterval)

    def draw_arena_boundaries(self):
        # Draw boundaries of the arena to the plot
        boundaries = np.array([[0, 0], \
                               [self.RPiSettings['arena_size'][0], 0], \
                               [self.RPiSettings['arena_size'][0], self.RPiSettings['arena_size'][1]], \
                               [0, self.RPiSettings['arena_size'][1]], \
                               [0, 0]])
        item = pg.PlotDataItem()
        item.setData(boundaries)
        item.setPen(pg.mkPen('b', width=4))
        self.plotWidget.addItem(item)

    def updatePlot(self):
        # Update position data
        linedatas = self.RPIpos.linedatas # Retrieve latest position data
        position, positions_2 = rpiI.computeAbsolutePosition(linedatas, self.RPiSettings) # Combine position data
        # Plot recent position change
        item = pg.PlotDataItem()
        item.setData(np.array([self.lastPosition, position]))
        item.setPen(pg.mkPen('w'))
        self.plotWidget.addItem(item)
        self.lastPosition = position # Update last position variable
        # Draw arrow if data for two LEDs available
        if self.RPiSettings['LEDmode'] == 'double':
            self.plotWidget.removeItem(self.arrow) # Remove previous arrow
            if np.any(np.isnan(positions_2)):
                # If no 2nd LED data, draw arrow to outside of the position -10,-10 outside the arena boundary
                print('no 2nd LED data')
                self.arrow = pg.ArrowItem(pos=(-10,-10), angle=45, headLen=30, tailLen=15,headWidth=30,tailWidth=5)
                self.plotWidget.addItem(self.arrow)
            else:
                # Compute arrow angle with to align the line connecting the points
                head_angle = angle_clockwise(position, positions_2) + 90 + self.RPiSettings['LED_angle']
                # Draw the arro with the head/tip at the location of primary LED
                self.arrow = pg.ArrowItem(pos=(position[0], position[1]), angle=head_angle, headLen=30, tailLen=15,headWidth=30,tailWidth=5)
                self.plotWidget.addItem(self.arrow)

    def close(self):
        # Close the update loop, RPi Position tracking loop and application window
        self.cumulativePlot_timer.stop()
        self.RPIpos.close()
        self.window.close()
        print('Stopped Cumulative Position Plot.')
