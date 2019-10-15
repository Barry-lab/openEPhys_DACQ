====================================
Welcome to openEPhys_DACQ repository
====================================

This is a system for automated experiments acquiring neural data using `OpenEhysGUI <https://github.com/open-ephys/plugin-GUI>`_. It combines online tracking from any number of cameras with a behavioural task program and peripherals for interacting with the subject. The package comes with an easy to use graphical interface.

The original purpose for this system was to collect neural data from rodents performing a navigational task in a large environment.

Installation
^^^^^^^^^^^^

Installation of the Python package is straightforward:

.. code-block:: none
    
    pip install openEPhys_DACQ
    
    openEPhys_Configuration  # Starts package configuration script

For full functionality, the package has dependencies beyond Python repositories:

- OpenEphysGUI for neural data acquisition
- Kilosort and/or KlustaKwik for spike sorting
- Raspberry Pis for video recording, tracking and task reward systems

For detailed instructions on how to set everything up, follow the :ref:`installationInstructions`.

GitHub repository
^^^^^^^^^^^^^^^^^

https://github.com/Barry-lab/openEPhys_DACQ

Features
========

* Data synchronisation with `OpenEhysGUI <https://github.com/open-ephys/plugin-GUI>`_
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
* Processing pipeline for tetrode recordings
    - Parallel implementation of 3 spike sorting methods
        + Kilosort
        + Klustakwik - offline detected spikes
        + Klustakwik - on OpenEphysGUI detected spikes

System overview and the user manual
===================================

In depth description of the system is in the :ref:`userManual_SystemOverview` section of the :ref:`userManual`

.. _installationInstructions:

Installation instructions
=========================

This document details every step required for setting up an Ubuntu PC for electrophysiological data collection using `OpenEhysGUI <https://github.com/open-ephys/plugin-GUI>`_, separate tracking system using Raspberry Pis and two different Rasperry Pi based reward modules. The instructions detail every single step, every line of code required to get things running. Such detail is provided so that anyone could set it up, even those who have never used Linux before.

Occasionally the document includes instructions in double brackets, where it is not entirely clear if the step is important, but best to follow the instructions. E.g. *((Do this, in case otherwise something will not work later on))*.

All terminal commands written here should be run while in home directory, unless instructed to call on a specific file. This ensures a sensible folder structure is created.

The instructions can be worked through in the order they are presented in the below table to set up the system from ground up.

.. toctree::
    :maxdepth: 2

    recordingPCandOpenEphysGUI
    cameraRPiSetup
    wirelessRewardSetup
    recordingPCforDataProcessing
    userManual
