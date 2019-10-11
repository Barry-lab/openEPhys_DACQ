
.. _userManual:

###########
User Manual
###########

.. _userManual_SystemOverview:

System overview
===============

The system can be described in two distinctive states: setting up the experiment and active recording state.

Setting up the experiment
-------------------------

When setting up the experiment, ``RecordingManager`` uses modules ``CameraSettings`` and ``TaskSettings`` to calibrate the cameras and set recording parameters.

Both ``CameraSettings`` and ``TaskSettings`` use ``ZMQcomms`` module to control any number of peripheral devices:

* Camera Raspbery Pis running ``CameraRPiController``
* Pellet Feeder Raspberry Pis running ``pelletFeederController``
* Milk Feeder Raspberry Pis running ``milkFeederController``

``ZMQcomms`` utilizes `PyZMQ <https://pyzmq.readthedocs.io/en/latest/>`_ to allow sharing data and direct control over Python processes operating on the peripheral devices over ethernet or WiFi connection.

``CameraSettings`` is used to configure the Camera Raspberry Pis and setting video recording and tracking parameters. Most importantly, ``CameraSettings`` calibrates all the cameras to the same reference frame, allowing online tracking with multiple cameras with non-overlapping fields of view. It does this using ``RPiInterface.CameraControl``, which gives it full control of Python process running on each Camera Raspbery Pi. ``CameraSettings`` can also display a live feed from each camera, making camera view adjustment and calibration very convenient.

``TaskSettings`` is used to choose a task program and to adjust parameters. The further functionality of this module depends on the specific task program. For example, the available ``Pellets_and_Rep_Milk`` task module can use ``RPiInterface.RewardControl`` to test functionality of reward devices.

.. code-block::
    
    RecordingManager
    |
    |
    |----   CameraSettings  ----    RPiInterface.CameraControl    ----    ZMQcomms
    |                                                                         |
    |                                                                         |
    |                                        CameraRPiController 1    <-->    |
    |                                        CameraRPiController 2    <-->    |
    |                                        CameraRPiController 3    <-->    |
    |                                                ....             <-->    |
    |
    |
    |
    |----   TaskSettings    ----    TaskModule      ----    RPiInterface.RewardControl
                                                                              |
                                                                          ZMQcomms
                                                                              |
                                                                              |
                                          pelletFeederController 1    <-->    |
                                          pelletFeederController 2    <-->    |
                                                    ....              <-->    |
                                                                              |
                                                                              |
                                            milkFeederController 1    <-->    |
                                            milkFeederController 2    <-->    |
                                                     ....             <-->    |

Druing recording
----------------

The recording is controlled by ``RecordingManager``. ``OpenEphysGUI`` must be initialized indpendently with a functioning signal chain that contains the `Network Events module <https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/23265310/Network+Events>`_. ``RecordingManager`` uses ``ZMQcomms`` to send start and stop messages to the ``OpenEphysGUI``.

As the recording starts, ``RecordingManager`` uses ``RPiInterface.TrackingControl`` to initialize and start cameras and ``RPiInterface.onlineTrackingData`` for combining incoming tracking information from cameras in real time. The online tracking data in ``RPiInterface.onlineTrackingData`` is also made available to the task module.

The function of ``TaskModule`` again depends on the specific task program that was chosen in the settings. In the case of ``Pellets_and_Rep_Milk`` task program it will be integrating information from ``RPiInterface.onlineTrackingData`` to monitor animal behaviour and using ``RPiInterface.RewardControl`` to activate reward devices as required. ``Pellets_and_Rep_Milk`` task program can also use the Milk Feeders as well as speakers connected to the PC for delivering light and audio signals. ``TaskModule`` is also provided with a messaging pipe to ``OpenEphysGUI`` via ``ZMQcomms`` module. This allows it to monitor any messages sent by the ``OpenEphysGUI`` (such as chewing artifact detection) as well as sending it messages about events in the task, which will be stored with a timestamp with alongside the neural data.

Finally, ``RPiInterface.GlobalClockControl`` is controlled by ``RecordingManager`` and it has a very simple function. It is started right after ``OpenEphysGUI`` and it deliveres simultaneous TTL pulses to ``OpenEphysGUI`` and ``CameraRPiController`` running on Camera Raspberry Pis. This allows very accurate synchronisation of neural data with position data from all the cameras.

.. code-block::
    
    RecordingManager    --------------------    ZMQcomms    -----------------    OpenEphysGUI
    |
    |
    |
    |
    |----   RPiInterface.TrackingControl    ----------    RPiInterface.CameraControl
    |                                                                        |
    |                                                                        |
    |----   RPiInterface.onlineTrackingData                                  |
    |                      /      |                                          |
    |                     /   ZMQcomms                                   ZMQcomms
    |                    /        |                                          |
    |                   /         | <---    CameraRPiController 3    <-->    |
    |                  /          | <---            ....             <-->    |
    |                 /
    |                /
    |               /
    |              /
    |             /
    |----   TaskModule  ----    RPiInterface.RewardControl    ----    ZMQcomms
    |                                                                     |
    |                                 pelletFeederController 1    <-->    |
    |                                           ....              <-->    |
    |                                                                     |
    |                                   milkFeederController 1    <-->    |
    |                                            ....             <-->    |
    |
    |
    |----   RPiInterface.GlobalClockControl

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