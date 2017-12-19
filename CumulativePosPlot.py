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
    angle = 360 - angle
    angle = angle - 180
    if angle < 0:
        angle = 360 + angle

    return angle

class PosPlot(object):
    def __init__(self, RPiSettings):
        self.RPiSettings = RPiSettings
        # Initialize plot window
        self.window = QtGui.QMainWindow()
        XandYratio = np.float32(self.RPiSettings['arena_size'][0]) / np.float32(self.RPiSettings['arena_size'][1])
        self.window.resize(int(700 * XandYratio), 700)
        self.plotWidget = pg.PlotWidget(self.window)
        # Set plot axes ranges
        margin_x = self.RPiSettings['arena_size'][0] * 0.05
        margin_y = self.RPiSettings['arena_size'][1] * 0.05
        self.plotWidget.setXRange(-margin_x, self.RPiSettings['arena_size'][0] + margin_x)
        self.plotWidget.setYRange(-margin_y, self.RPiSettings['arena_size'][1] + margin_y)
        # Invert Y-axis #### This messes up the arrow direction
        self.plotWidget.invertY(True)
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
        self.lastCombPos = None
        while not np.any(self.lastCombPos): # Only continue once first position data is obtained
            time.sleep(0.1)
            self.lastCombPos = self.RPIpos.lastCombPos # Retrieve latest position data
        # Start constant update of the plot
        plotUpdateInterval = 250 # This sets the plot update interval in milliseconds
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
        newCombPos = self.RPIpos.lastCombPos # Retrieve latest position data
        if np.any(newCombPos):
            # Plot recent position change
            item = pg.PlotDataItem()
            item.setData(np.array([self.lastCombPos[:2], newCombPos[:2]]))
            item.setPen(pg.mkPen('w'))
            self.plotWidget.addItem(item)
            # Draw arrow data for two LEDs available
            self.plotWidget.removeItem(self.arrow) # Remove previous arrow
            if self.RPiSettings['LEDmode'] == 'double' and not np.any(np.isnan(newCombPos[2:])):
                # Compute arrow angle with to align the line connecting the points
                head_angle = angle_clockwise(newCombPos[:2], newCombPos[2:]) + 90 + self.RPiSettings['LED_angle']
                # Draw the arro with the head/tip at the location of primary LED
                self.arrow = pg.ArrowItem(pos=(newCombPos[0], newCombPos[1]), 
                                          angle=head_angle, headLen=30, tailLen=15, 
                                          headWidth=30, tailWidth=5, 
                                          brush=pg.mkBrush('g'), pen=pg.mkPen('c', width=3))
                self.plotWidget.addItem(self.arrow)
            else:
                # If no 2nd LED data, draw arrow based on movement direction
                head_angle = angle_clockwise(self.lastCombPos[:2], newCombPos[:2]) + 90 + self.RPiSettings['LED_angle']
                self.arrow = pg.ArrowItem(pos=(newCombPos[0], newCombPos[1]), 
                                          angle=head_angle, headLen=30, tailLen=15, 
                                          headWidth=30, tailWidth=5, 
                                          brush=pg.mkBrush('b'), pen=pg.mkPen('c', width=3))
                self.plotWidget.addItem(self.arrow)                    
            self.lastCombPos = newCombPos

    def close(self):
        # Close the update loop, RPi Position tracking loop and application window
        self.cumulativePlot_timer.stop()
        self.RPIpos.close()
        self.window.close()
        print('Stopped Cumulative Position Plot.')
