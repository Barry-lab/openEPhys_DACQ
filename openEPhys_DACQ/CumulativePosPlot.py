### This script produces a window where it plots current tracking position
### and also history of tracked position.

### By Sander Tanni, January 2018, UCL

import sys
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtGui
import numpy as np
from PyQt5.QtCore import QTimer
from time import sleep
from scipy import ndimage
from copy import copy, deepcopy


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


class PosPlot(QtWidgets.QWidget):
    """
    Displays position data as provided by OnlineTracker class.

    Closes automatically when OnlineTracker class is no longer active.
    """

    def __init__(self, processed_position_list, position_histogram_dict,
                 position_histogram_update_parameters, position_histogram_dict_updating,
                 online_tracker_is_alive, arena_size, LED_angle=None):
        super().__init__()

        self.processed_position_list = processed_position_list
        self.position_histogram_dict = position_histogram_dict
        self.position_histogram_update_parameters = position_histogram_update_parameters
        self.position_histogram_dict_updating = position_histogram_dict_updating
        self.online_tracker_is_alive = online_tracker_is_alive
        self.arena_size = arena_size
        self.LED_angle = LED_angle

        self.histogramParameters = copy(self.position_histogram_dict['parameters'])

        # Only continue once first two position samples are available
        while len(self.processed_position_list) < 2:
            sleep(0.1)

        # Initialize plot window
        self.PlotGraphicsWidget = pg.GraphicsLayoutWidget()
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
        self.maxValueBox = QtWidgets.QLineEdit('10')
        self.thresholdValueBox = QtWidgets.QLineEdit('2')
        self.maxValueBox.returnPressed.connect(self.prepareColormap)
        self.thresholdValueBox.returnPressed.connect(self.prepareColormap)
        self.smoothingValueBox = QtWidgets.QLineEdit('3')
        self.smoothingValueBox.returnPressed.connect(self.updateSmoothingValue)
        self.speedLimitBox = QtWidgets.QLineEdit(str(self.histogramParameters['speedLimit']))
        self.speedLimitBox.returnPressed.connect(self.updateHistogramParameters)
        self.binSizeBox = QtWidgets.QLineEdit(str(self.histogramParameters['binSize']))
        self.binSizeBox.returnPressed.connect(self.updateHistogramParameters)
        self.marginsBox = QtWidgets.QLineEdit(str(self.histogramParameters['margins']))
        self.marginsBox.returnPressed.connect(self.updateHistogramParameters)
        self.updateSmoothingValue()
        self.showPathButton = QtWidgets.QPushButton('Show Path')
        self.showPathButton.setCheckable(True)
        self.showPathButton.setChecked(False)
        self.showPathButton.clicked.connect(self.showPath)

        # Put all plots into main window and display
        self.setWindowTitle('Cumulative Position Plot')
        XandYratio = (float(self.arena_size[0])
                      / float(self.arena_size[1]))
        self.resize(int(700 * XandYratio) + 200, 700)
        vboxWidget = QtWidgets.QWidget()
        vboxWidget.setFixedWidth(100)
        vbox = QtWidgets.QVBoxLayout(vboxWidget)
        vbox.addWidget(QtWidgets.QLabel('Maximum'))
        vbox.addWidget(QtWidgets.QLabel('value'))
        vbox.addWidget(QtWidgets.QLabel('n visits'))
        vbox.addWidget(self.maxValueBox)
        vbox.addStretch()
        vbox.addWidget(QtWidgets.QLabel('Threshold'))
        vbox.addWidget(QtWidgets.QLabel('value'))
        vbox.addWidget(self.thresholdValueBox)
        vbox.addStretch()
        vbox.addWidget(QtWidgets.QLabel('Smoothing'))
        vbox.addWidget(QtWidgets.QLabel('in cm'))
        vbox.addWidget(self.smoothingValueBox)
        vbox.addWidget(QtWidgets.QLabel('Speed lim'))
        vbox.addWidget(QtWidgets.QLabel('in cm/s'))
        vbox.addWidget(self.speedLimitBox)
        vbox.addWidget(QtWidgets.QLabel('Bin size'))
        vbox.addWidget(QtWidgets.QLabel('in cm'))
        vbox.addWidget(self.binSizeBox)
        vbox.addWidget(QtWidgets.QLabel('Margins'))
        vbox.addWidget(QtWidgets.QLabel('in cm'))
        vbox.addWidget(self.marginsBox)
        vbox.addWidget(self.showPathButton)
        hbox = QtWidgets.QHBoxLayout(self)
        hbox.addWidget(self.PlotGraphicsWidget)
        hbox.addWidget(self.ColormapGraphicsWidget)
        hbox.addWidget(vboxWidget)
        self.show()

        # Prepare plot
        self.prepareColormap()
        self.updatePlotAxes()
        self.draw_arena_boundaries()
        self.arrow = pg.ArrowItem(pos=(0, 0), angle=0, headLen=0, tailLen=0, headWidth=0, tailWidth=0)
        self.plotBox.addItem(self.arrow)

        # Start constant update of the plot
        self.keep_updating_plot = True
        plotUpdateInterval = 100  # This sets the plot update interval in milliseconds
        self.cumulativePlot_timer = QTimer()
        self.cumulativePlot_timer.timeout.connect(self.updatePlot)
        self.cumulativePlot_timer.start(plotUpdateInterval)

        # Keep track of whether Online Tracker is still alive. If not, close position plot
        self.online_tracking_data_alive_checker_timer = QTimer()
        self.online_tracking_data_alive_checker_timer.timeout.connect(self.close_if_online_tracker_not_alive)
        self.online_tracking_data_alive_checker_timer.start(100)

    def close_if_online_tracker_not_alive(self):
        if not self.keep_updating_plot or not self.online_tracker_is_alive.get():
            self.close()

    def updatePlotAxes(self):
        binSize = self.histogramParameters['binSize']
        margins = self.histogramParameters['margins']
        xRange = (0, (self.arena_size[0] + 2 * margins) / binSize)
        yRange = (0, (self.arena_size[1] + 2 * margins) / binSize)
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
                tmpcolorval[:, ncolor] = np.interp(tmpcolorpos,
                                                   np.array([cpos1, cpos2], dtype=np.float32),
                                                   np.array([cval1, cval2], dtype=np.float32))
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
        boundaries = np.array([[0, 0],
                               [self.arena_size[0], 0],
                               [self.arena_size[0],
                                self.arena_size[1]],
                               [0, self.arena_size[1]],
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

        # Initiate update function
        self.position_histogram_update_parameters['margins'] = float(str(self.marginsBox.text()))
        self.position_histogram_update_parameters['binSize'] = float(str(self.binSizeBox.text()))
        self.position_histogram_update_parameters['speedLimit'] = float(str(self.speedLimitBox.text()))
        self.position_histogram_dict_updating.set(True)
        while self.position_histogram_dict_updating.get():
            sleep(0.1)

        # Use new established histogram parameters
        self.histogramParameters = deepcopy(self.position_histogram_dict['parameters'])
        self.updatePlotAxes()
        self.plotBox.removeItem(self.boundaryBox)
        self.draw_arena_boundaries()

    def showPath(self):
        if self.showPathButton.isChecked():
            posHistory = copy(self.processed_position_list[0:])
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
        if self.keep_updating_plot and self.online_tracker_is_alive.get():
            binSize = self.histogramParameters['binSize']
            margins = self.histogramParameters['margins']
            # Get latest position data
            pastPos, currPos = copy(self.processed_position_list[-2:])
            positionHistogram = self.position_histogram_dict['data']
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

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.close()
        super(PosPlot, self).closeEvent(a0)

    def close(self):
        if self.keep_updating_plot:
            # Close the update loop, RPi Position tracking loop and application window
            self.keep_updating_plot = False
            self.cumulativePlot_timer.stop()
            self.online_tracking_data_alive_checker_timer.stop()
            sleep(0.25)
            print('Stopped Cumulative Position Plot.')
        super(PosPlot, self).close()
