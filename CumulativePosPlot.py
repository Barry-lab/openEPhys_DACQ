### This script produces a window where it plots current tracking position
### and also history of tracked position.

### By Sander Tanni, January 2018, UCL

import pyqtgraph as pg
from PyQt4 import QtGui
import numpy as np
from PyQt4.QtCore import QTimer, QRectF
import RPiInterface as rpiI
import time
from scipy import ndimage
from copy import copy

def angle_clockwise(p1, p2, invertedY=True):
    # Takes two numpy 2D vectors and computes the angle of their difference
    # In other words, computes the angle of a vector from p1 to p2
    p0 = np.array(p2) - np.array(p1)
    angle_rad = np.arctan2(p0[0], p0[1])
    angle_rad = np.arctan2(np.sin(angle_rad), np.cos(angle_rad))
    angle_deg = np.degrees(angle_rad)
    # Correct angle for inveted image
    if invertedY:
        angle_deg = -angle_deg + 180

    return angle_deg

class PosPlot(object):
    def __init__(self, RPIPos, LED_angle=None):
        self.RPIpos = RPIPos
        self.histogramParameters = copy(self.RPIpos.HistogramParameters)
        self.LED_angle = LED_angle
        # Only continue once first two position datas are obtained
        combPosHistory = []
        while len(combPosHistory) < 2:
            time.sleep(0.1)
            with self.RPIpos.combPosHistoryLock:
                combPosHistory = self.RPIpos.combPosHistory
        # Initialize plot window
        self.PlotGraphicsWidget = pg.GraphicsLayoutWidget()
        XandYratio = np.float32(self.RPIpos.arena_size[0]) / np.float32(self.RPIpos.arena_size[1])
        self.plotBox = self.PlotGraphicsWidget.addViewBox(enableMouse=False)
        self.plotBox.setAspectLocked(True)
        self.imageItem = pg.ImageItem()
        self.plotBox.addItem(self.imageItem)
        self.plotBox.invertY(True)
        # Create color map box
        self.ColormapGraphicsWidget = pg.GraphicsLayoutWidget()
        self.ColormapGraphicsWidget.setFixedWidth(50)
        self.colorMapBox = self.ColormapGraphicsWidget.addViewBox(enableMouse=False)
        self.colorMapBox.setAspectLocked(False)
        self.colorMapImageItem = pg.ImageItem()
        self.colorMapBox.addItem(self.colorMapImageItem)
        # Create UI controls
        self.maxValueBox = QtGui.QLineEdit('10')
        self.thresholdValueBox = QtGui.QLineEdit('2')
        self.maxValueBox.returnPressed.connect(self.prepareColormap)
        self.thresholdValueBox.returnPressed.connect(self.prepareColormap)
        self.smoothingValueBox = QtGui.QLineEdit('3')
        self.smoothingValueBox.returnPressed.connect(self.updateSmoothingValue)
        self.speedLimitBox = QtGui.QLineEdit(str(self.histogramParameters['speedLimit']))
        self.speedLimitBox.returnPressed.connect(self.updateHistogramParameters)
        self.binSizeBox = QtGui.QLineEdit(str(self.histogramParameters['binSize']))
        self.binSizeBox.returnPressed.connect(self.updateHistogramParameters)
        self.marginsBox = QtGui.QLineEdit(str(self.histogramParameters['margins']))
        self.marginsBox.returnPressed.connect(self.updateHistogramParameters)
        self.updateSmoothingValue()
        self.showPathButton = QtGui.QPushButton('Show Path')
        self.showPathButton.setCheckable(True)
        self.showPathButton.setChecked(False)
        self.showPathButton.clicked.connect(self.showPath)
        # Put all plots into main window and display
        self.mainWindow = QtGui.QWidget()
        self.mainWindow.setWindowTitle('Cumulative Position Plot')
        XandYratio = float(self.RPIpos.arena_size[0]) / float(self.RPIpos.arena_size[1])
        self.mainWindow.resize(int(700 * XandYratio) + 200, 700)
        vboxWidget = QtGui.QWidget()
        vboxWidget.setFixedWidth(100)
        vbox = QtGui.QVBoxLayout(vboxWidget)
        vbox.addWidget(QtGui.QLabel('Maximum'))
        vbox.addWidget(QtGui.QLabel('value'))
        vbox.addWidget(QtGui.QLabel('n visits'))
        vbox.addWidget(self.maxValueBox)
        vbox.addStretch()
        vbox.addWidget(QtGui.QLabel('Threshold'))
        vbox.addWidget(QtGui.QLabel('value'))
        vbox.addWidget(self.thresholdValueBox)
        vbox.addStretch()
        vbox.addWidget(QtGui.QLabel('Smoothing'))
        vbox.addWidget(QtGui.QLabel('in cm'))
        vbox.addWidget(self.smoothingValueBox)
        vbox.addWidget(QtGui.QLabel('Speed lim'))
        vbox.addWidget(QtGui.QLabel('in cm/s'))
        vbox.addWidget(self.speedLimitBox)
        vbox.addWidget(QtGui.QLabel('Bin size'))
        vbox.addWidget(QtGui.QLabel('in cm'))
        vbox.addWidget(self.binSizeBox)
        vbox.addWidget(QtGui.QLabel('Margins'))
        vbox.addWidget(QtGui.QLabel('in cm'))
        vbox.addWidget(self.marginsBox)
        vbox.addWidget(self.showPathButton)
        hbox = QtGui.QHBoxLayout(self.mainWindow)
        hbox.addWidget(self.PlotGraphicsWidget)
        hbox.addWidget(self.ColormapGraphicsWidget)
        hbox.addWidget(vboxWidget)
        self.mainWindow.show()
        # Prepare plot
        self.prepareColormap()
        self.updatePlotAxes()
        self.draw_arena_boundaries()
        self.arrow = pg.ArrowItem(pos=(0,0), angle=0, headLen=0, tailLen=0,headWidth=0,tailWidth=0)
        self.plotBox.addItem(self.arrow)
        # Start constant update of the plot
        plotUpdateInterval = 100 # This sets the plot update interval in milliseconds
        self.cumulativePlot_timer = QTimer()
        self.cumulativePlot_timer.timeout.connect(lambda:self.updatePlot())
        self.cumulativePlot_timer.start(plotUpdateInterval)

    def updatePlotAxes(self):
        binSize = self.histogramParameters['binSize']
        margins = self.histogramParameters['margins']
        xRange = (0, (self.RPIpos.arena_size[0] + 2 * margins) / binSize)
        yRange = (0, (self.RPIpos.arena_size[1] + 2 * margins) / binSize)
        self.plotBox.setRange(xRange=xRange, yRange=yRange)

    def prepareColormap(self):
        # This function prepares colormap for displaying the histogram
        # Get colormap parameters
        self.hist_max_range = float(str(self.maxValueBox.text()))
        self.hist_threshold_value = float(str(self.thresholdValueBox.text()))
        # Set colors for below and above threshold
        # Change colorval here to edit colors
        colormidpos = self.hist_threshold_value / float(self.hist_max_range)
        colorpos = np.array([0.0, 0.001, colormidpos, colormidpos + 0.001, 1.0], dtype=np.float32)
        colorval = np.array([[0,0,0], [255, 0, 0], [127,127,255], [0,255,0], [0,255,255]], dtype=np.ubyte)
        # Interpolate at high resolution to provide input to pyqtgraph colormap function
        colorposFull = np.array([], dtype=np.float32)
        colorvalFull = np.zeros((0,3), dtype=np.float32)
        cincr = 0.0001
        for cstep in range(len(colorpos) -  1):
            cpos1, cpos2 = colorpos[cstep:cstep+2]
            tmpcolorpos = np.arange(cpos1, cpos2, cincr)
            colorposFull = np.append(colorposFull, tmpcolorpos)
            tmpcolorval = np.zeros((tmpcolorpos.size, colorval.shape[1]), dtype=np.float32)
            for ncolor in range(colorval.shape[1]):
                cval1 = colorval[cstep][ncolor]
                cval2 = colorval[cstep + 1][ncolor]
                tmpcolorval[:,ncolor] = np.interp(tmpcolorpos, np.array([cpos1, cpos2], dtype=np.float32), np.array([cval1, cval2], dtype=np.float32))
            colorvalFull = np.concatenate((colorvalFull, tmpcolorval.astype(np.uint8)), axis=0)
        # Form colormap lookup table for plotting
        CMap = pg.ColorMap(colorposFull, colorvalFull)
        self.CMapLut = CMap.getLookupTable(start=0.0, stop=1.0, nPts=512, alpha=False, mode='float')
        # Display colormap in a separate plot
        colormap_image = np.repeat(np.linspace(0.0,1.0,300), 15).reshape((300, 15))
        colormap_image = np.uint8(colormap_image * 255)
        self.colorMapImageItem.setImage(colormap_image.T, autoLevels=False, lut=self.CMapLut)

    def draw_arena_boundaries(self):
        # Draw boundaries of the arena to the plot
        boundaries = np.array([[0, 0], \
                               [self.RPIpos.arena_size[0], 0], \
                               [self.RPIpos.arena_size[0], self.RPIpos.arena_size[1]], \
                               [0, self.RPIpos.arena_size[1]], \
                               [0, 0]])
        boundaries = boundaries + self.histogramParameters['margins']
        boundaries = boundaries / float(self.histogramParameters['binSize'])
        self.boundaryBox = pg.PlotDataItem()
        self.boundaryBox.setData(boundaries)
        self.boundaryBox.setPen(pg.mkPen('b', width=4))
        self.plotBox.addItem(self.boundaryBox)

    def updateSmoothingValue(self):
        # This is in centimeters
        self.smoothingValue = float(str(self.smoothingValueBox.text()))

    def updateHistogramParameters(self):
        # Load parameters into dictionary
        speedLimit = float(str(self.speedLimitBox.text()))
        binSize = float(str(self.binSizeBox.text()))
        margins = float(str(self.marginsBox.text()))
        histogramParameters = {'margins': margins, 
                               'binSize': binSize, 
                               'speedLimit': speedLimit}
        # Initiate update function
        self.RPIpos.initializePosHistogram(histogramParameters, update=True)
        self.histogramParameters = histogramParameters
        self.updatePlotAxes()
        self.plotBox.removeItem(self.boundaryBox)
        self.draw_arena_boundaries()

    def showPath(self):
        if self.showPathButton.isChecked():
            with self.RPIpos.combPosHistoryLock:
                posHistory = self.RPIpos.combPosHistory
            posHistory = [i for i in posHistory if i is not None]
            posHistory = np.array(posHistory)[:, :2]
            posHistory = posHistory + self.histogramParameters['margins']
            posHistory = posHistory / float(self.histogramParameters['binSize'])
            self.trackedPath = pg.PlotDataItem()
            self.trackedPath.setData(posHistory)
            self.trackedPath.setPen(pg.mkPen('w', width=1))
            self.plotBox.addItem(self.trackedPath)
        else:
            self.plotBox.removeItem(self.trackedPath)

    def updatePlot(self):
        binSize = self.histogramParameters['binSize']
        margins = self.histogramParameters['margins']
        # Get latest position data
        with self.RPIpos.combPosHistoryLock:
            pastPos = self.RPIpos.combPosHistory[-2]
            currPos = self.RPIpos.combPosHistory[-1]
        with self.RPIpos.histogramLock:
            positionHistogram = self.RPIpos.positionHistogram
        # Smooth histogram
        smoothGaussStd = self.smoothingValue / float(binSize)
        image = ndimage.gaussian_filter(positionHistogram.T, sigma=(smoothGaussStd, smoothGaussStd), order=0)
        # Set image data maximum value as requested
        image = np.float32(image)
        image[image > self.hist_max_range] = np.float32(self.hist_max_range)
        # Display image of histogram
        image = np.uint8(image / np.float32(self.hist_max_range) * 255)
        self.imageItem.setImage(image, autoLevels=False, lut=self.CMapLut)
        # Draw arrow for position if position data available
        self.plotBox.removeItem(self.arrow) # Remove previous arrow
        if not (currPos is None) and not (pastPos is None):
            arrowPos = ((currPos[0] + margins) / binSize, (currPos[1] + margins) / binSize)
            if not (self.LED_angle is None) and not np.any(np.isnan(currPos[2:])):
                # Compute arrow angle with to align the line connecting the points
                head_angle = angle_clockwise(currPos[:2], currPos[2:]) + 90 + self.LED_angle
                # Draw the arro with the head/tip at the location of primary LED
                self.arrow = pg.ArrowItem(pos=arrowPos, 
                                          angle=head_angle, headLen=40, tailLen=20, 
                                          headWidth=40, tailWidth=7, 
                                          brush=pg.mkBrush('b'), pen=pg.mkPen('c', width=3))
                self.plotBox.addItem(self.arrow)
            else:
                # If no 2nd LED data, draw arrow based on movement direction
                head_angle = angle_clockwise(pastPos[:2], currPos[:2]) + 90
                self.arrow = pg.ArrowItem(pos=arrowPos, 
                                          angle=head_angle, headLen=40, tailLen=0, 
                                          headWidth=40, tailWidth=7, 
                                          brush=pg.mkBrush('b'), pen=pg.mkPen('c', width=3))
                self.plotBox.addItem(self.arrow)
        else:
            # Add invisible arrow item to keep loop consistent
            self.arrow = pg.ArrowItem(pos=[0, 0], 
                                      angle=0, headLen=0, tailLen=0, 
                                      headWidth=0, tailWidth=0, 
                                      brush=pg.mkBrush('b'))
            self.plotBox.addItem(self.arrow)

    def close(self):
        # Close the update loop, RPi Position tracking loop and application window
        self.cumulativePlot_timer.stop()
        self.mainWindow.close()
        print('Stopped Cumulative Position Plot.')
