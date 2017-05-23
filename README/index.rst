============================================
Setup for Open Ephys Recordings and Analysis
============================================

This document details every step required for setting up a PC for electrophysiological data collection using Open Ephys Acquisition Board and separate tracking system using Raspberry Pis.

To allow recreating this setup, this document is an attempt to detail every single step, every line of code required to get things running. Therefore, it is definitely painfully long to work through for an advanced user, if not amusing at times. But the hope is that based on this, anyone could set it up.

Occasionally the document includes instructions in double brackets, where the steps may not me necessary but were made by me during the setup process. E.g. *((Do this, in case otherwise something will not work later on))*.

All terminal commands written here should be run while in home directory, unless instructed to call on a specific file. This ensures a sensible folder structure is created.

The instructions can be worked through in the order they are presented in the below table to set up the system from ground up.

.. toctree::
	
	recordingPCandOpenEphysGUI
	raspberryPiSetup
	recordingPCforDataProcessing
	howToUseTheseScripts
