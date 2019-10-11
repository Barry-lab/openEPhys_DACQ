.. _userManual:

=====================================
Tutorial for using Open Ephys Scripts
=====================================

Using OpenEphysGUI
------------------

Use OpenEphysGUI to capture electrophysiological signals.

Using Recording Manager
-----------------------

Use RecordingManager to handle all the other aspects of the recording (cameras and interactive task).

Camera Settings and Calibration
-------------------------------

Through Recording Manager you can access Camera Settings, which allows you to Calibrate the cameras.

Use a bright spot, for example a very bright LED (can be IR) to pick the ``(0,0)`` corner.

Camera Calibration as limitations where position ``(0,0)`` can be. The position ``(0,0)`` can be found to be either in the bottom left or top right of the image taken by each RPi camera (this is because of how ``cv2`` module ``findChessboardCorners`` method works). This means that if two cameras were pointed at 90 degree angles from each other, they would not find the same ``(0,0)`` position using the calibration squares poster. However, as long as cameras are facing in parallel or opposite directions, there should not be any issues.

Processing
-----------

After the recording is finished, use DetectWaveforms.py to detect waveforms on each tetrode. Through this application you will be able to ApplyKlustakwik to the waveforms.

Then you can use createWaveformGUIdata.py to create files in the form that can be read into Waveform GUI online application.

NOTE! WaveformGUI speed data is incorrect due to problems with conversion from RPi data to Axona format.