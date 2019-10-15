.. _recordingPCforDataProcessing:

======================================
Recording PC setup for Data Processing
======================================

This guide assumes you have already set up a PC with Ubuntu and installed much of the software as instructed in this guide: :ref:`recordingPCandOpenEphysGUI`. If you are setting up a system for data processing only, then the relavant parts are just the following: :ref:`settingUpTheOperatingSystem` and :ref:`otherUsefulSteps`.

Install MATLAB for KiloSort
---------------------------

MATLAB is only required to use KiloSort. If you do not intend to use KiloSort, you can skip this part.

Depending on the source of the MATLAB license, your setup may vary. The following section only details the setup for those who can access University College London Software Database. If you are installing MATLAB from another source, move to :ref:`installingMatlabAddons` after installation is finished.

Installing MATLAB from UCL Software Database website
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instructions are available at `UCL Software Database website <http://swdb.ucl.ac.uk/>`_. But below is the summary of steps.

Download the *.iso* files for MATLAB on Linux from `UCL Software Database <http://swdb.ucl.ac.uk/>`_. Right click on *...dvd1.iso* and select *Open With -> Archive Manager*. Click on *Extract* in the top left corner of Archive Manager. Select the location for the files to be unpacked, preferably the Downloads folder. Create a new folder called *MATLAB_INSTALL* using the button on the top right of the Archive Manager. Select this folder and choose to *Extract All files* and to *Keep directory structure*.

Unpacked files may be set to read-only mode. To fix this, right click on the *MATLAB_INSTALL* folder you just created and chose *Properties*. There go to *Permissions* tab and make sure all rights are set to *Create and delete files*. Click on *Change Permissions for Enclosed Files...* and make the same changes there.

Now open the *...dvd2.iso* in the Archive Manager and Extract these files to the same *MATLAB_INSTALL* folder you previously used. Afterwards, you may have to change the folder permissions again, as instructed in the previous paragraph.

Now go to the *MATLAB_INSTALL* folder where you unpacked the files, right click and choose *Open in Terminal*. Enter the terminal command ``sudo ./install``. Follow through the installation. Refer to the instructions `UCL Software Database <http://swdb.ucl.ac.uk/>`_ if necessary. Feel free to delete the *MATLAB_INSTALL* folder once the installation has finished.

.. _installingMatlabAddons:

Installing MATLAB addons
^^^^^^^^^^^^^^^^^^^^^^^^


MATLAB is installed at ``/usr/local/MATLAB/R2017b/bin`` and can be run by calling the terminal command ``./matlab`` while in that folder. Alternatively you can install an add-on package with the terminal command ``sudo apt-get install matlab-support``, entering ``usr/local/MATLAB/R2017b`` as the MATLAB path and leaving the username request empty, but do rename the GCC files as it recommends. Now you can start MATLAB by opening the terminal anywhere and entering command ``matlab``. It may also work simply by searching it from Dash and in that case you could also Lock it to the Launcher.

To use similar keyboard shortucts in MATLAB as is used on Windows, such as Ctrl+C for copying, you need to edit some preferences. Start MATLAB, go to Home Tab and select *Preferences*. In the list on the left, choose *Keyboard -> Shortcuts* and change the *Active settings:* to *Windows default set*.

When you add Paths to MATLAB and try to save it, you may promted to save ``pathdef.m`` to a different location. A good place would be the Home folder, so that by opening the terminal with ``Ctrl+Alt+T`` you can just type in ``matlab`` and it should load the correct paths.

Install KiloSort
----------------

If you do not intend to use KiloSort, you can skip this section.

Download and unpack or ``git clone`` `KiloSort GitHub repository <https://github.com/cortex-lab/KiloSort>`_ to a convenient location, for example the Programs folder. You can use the following command:

.. code-block:: none
	
	cd Programs
	git clone https://github.com/cortex-lab/KiloSort
	cd ~

Follow further instructions on `KiloSort GitHub repository <https://github.com/cortex-lab/KiloSort>`_ for ensuring MATLAB can use the GPU. In short, MATLAB GPU access via CUDA must be configured first and then you should run the ``KiloSort/CUDA/mexGPUall.m`` file in the downloaded repository using MATLAB.

Download and unpack or ``git clone`` `npy matlab GitHub repository <https://github.com/kwikteam/npy-matlab>`_ to a convenient location, for example the Programs folder. You can use the following command:

.. code-block:: none
	
	cd Programs
	git clone https://github.com/kwikteam/npy-matlab
	cd ~

The paths to these folders just created should be provided to ``openEPhys_Configuration`` when promted.

Install Klustakwik
------------------

If you do not intend to use Klustakwik, you can skip this section.

Use the following terminal commands to install Klustakwik

.. code-block:: none
	
	cd Programs
	git clone https://github.com/klusta-team/klustakwik
	cd klustakwik
	make

The paths to the KlustaKwik file in klustakwik folder just created should be provided to ``openEPhys_Configuration`` when promted.

Install Google Chrome to use Waveform GUI
-----------------------------------------

`Waveform GUI <http://d1manson.github.io/waveform/>`_ is a browser based method for conveniently viewing clustered unit waveforms and other unit properties (e.g. spatial correlograms) of data in Axona format. The data recorded with this repository can be converted into Axona format using the processing pipeline. See :ref:`userManual` for more details.

To use Waveform GUI you need Google Chrome.

To install Google Chrome go to `website <https://www.google.com/chrome/>`_ and download 64 bit *.deb* package for Ubuntu. Once the download is finished, go to the downloaded file and Right Click -> Open With -> GDebi Package Installer. Click Install Package.

Install openEPhys_DACQ package
------------------------------

Installation of the Python package is straightforward:

.. code-block:: none
    
    pip install openEPhys_DACQ
    
    openEPhys_Configuration  # Starts package configuration script

To process data recorded with this package, use terminal command:

.. code-block:: none
    
    openEPhys_Processing

You are now ready to use the Open Ephys Data Processing scripts to detect spike, cluster them with Klustakwik and view the result with spatial correlograms in `Waveform GUI <http://d1manson.github.io/waveform/>`_.

See :ref:`userManual` for more detailed instructions on using the package for data processing.
