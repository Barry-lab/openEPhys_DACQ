====================================
Welcome to openEPhys_DACQ repository
====================================

This is a system for automated experiments acquiring neural data using `OpenEhysGUI <https://github.com/open-ephys/plugin-GUI>`_. It combines online tracking from any number of cameras with a behavioural task program and peripherals for interacting with the subject. The package comes with an easy to use graphical interface.

The original purpose for this system was to collect neural data from rodents performing a navigational task in a large environment.

Features
========

* Data synchronisation with `OpenEhysGUI <https://github.com/open-ephys/plugin-GUI>`
    - Global clock TTL signal for precision synchrony
    - Logging messages with recording timestamps (e.g. pellet dropped)
* Real-time tracking and 1080p video recording
    - Any number of cameras
        + Built-in calibration to same reference frame
    - Multiple tracking methods
        + Motion
        + 1 LED (brightest spot)
        + 2 LED (different luminance)
* Reward devices
    - Pellet Feeder
        + Releases specified number of pellets on command
        + Activation automated based on animal behaviour
        + WiFi controlled
    - Milk Feeder
        + Releases milk or other liquid with pinch valve
        + Animal friendly - can be placed in the enviornment
        + Battery powered
        + WiFi controlled
* Automated behavioural task
    - New task programs can be integrated independently
    - Has messaging ability with OpenEphysGUI
    - Has real-time tracking information
    - Controls reward devices
    - Navigational task available
        + Fully configurable flexible task program
        + Milk feeder reward based navigational task
        + Audio and/or light signals for trials
        + Concurrent foraging task with intelligent pellet scattering

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


Installation
============

This document details every step required for setting up an Ubuntu PC for electrophysiological data collection using `OpenEhysGUI <https://github.com/open-ephys/plugin-GUI>`_, separate tracking system using Raspberry Pis and two different Rasperry Pi based reward modules. The instructions detail every single step, every line of code required to get things running. Therefore, it is definitely painfully long to work through for an advanced user, if not amusing at times. The reason for such detail is that anyone could set it up.

Occasionally the document includes instructions in double brackets, where the steps may not me necessary but were made by me during the setup process. E.g. *((Do this, in case otherwise something will not work later on))*.

All terminal commands written here should be run while in home directory, unless instructed to call on a specific file. This ensures a sensible folder structure is created.

The instructions can be worked through in the order they are presented in the below table to set up the system from ground up.

.. toctree::
    :maxdepth: 2

    recordingPCandOpenEphysGUI
    cameraRPiSetup
    wirelessRewardSetup
    recordingPCforDataProcessing
    userManual
