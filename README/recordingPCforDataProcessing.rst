.. _recordingPCforDataProcessing:

======================================
Recording PC setup for Data Processing
======================================

This guide assumes you have already set up a PC with Ubuntu and installed much of the software as instructed in this guide: :ref:`recordingPCandOpenEphysGUI`. Possibly the relavant part may be just the instructions in the following parts: :ref:`settingUpTheOperatingSystem` and :ref:`otherUsefulSteps`.

Install MATLAB
--------------

Instructions are available at `UCL Software Database website <http://swdb.ucl.ac.uk/>`_. But below is the summary of steps.

Download the *.iso* files for MATLAB on Linux from `UCL Software Database <http://swdb.ucl.ac.uk/>`_. Right click on *...dvd1.iso* and select *Open With -> Archive Manager*. Click on *Extract* in the top left corner of Archive Manager. Select the location for the files to be unpacked, preferably the Downloads folder. Create a new folder called *MATLAB_INSTALL* using the button on the top right of the Archive Manager. Select this folder and choose to *Extract All files* and to *Keep directory structure*.

Unpacked files may be set to read-only mode. To fix this, right click on the *MATLAB_INSTALL* folder you just created and chose *Properties*. There go to *Permissions* tab and make sure all rights are set to *Create and delete files*. Click on *Change Permissions for Enclosed Files...* and make the same changes there.

Now open the *...dvd2.iso* in the Archive Manager and Extract these files to the same *MATLAB_INSTALL* folder you previously used. Afterwards, you may have to change the folder permissions again, as instructed in the previous paragraph.

Now go to the *MATLAB_INSTALL* folder where you unpacked the files, right click and choose *Open in Terminal*. Enter the terminal command ``sudo ./install``. Follow through the installation. Refer to the instructions `UCL Software Database <http://swdb.ucl.ac.uk/>`_ if necessary. Feel free to delete the *MATLAB_INSTALL* folder once the installation has finished.

MATLAB is installed at ``/usr/local/MATLAB/R2016b/bin`` and can be run by calling the terminal command ``./matlab`` while in that folder. Alternatively you can install an add-on package with the terminal command ``sudo apt-get install matlab-support``, entering ``usr/local/MATLAB/R2016b`` as the MATLAB path and leaving the username request empty, but do rename the GCC files as it recommends. Now you can start MATLAB by opening the terminal anywhere and entering command ``matlab``. It may also work simply by searching it from Dash and in that case you could also Lock it to the Launcher.

To use similar keyboard shortucts in MATLAB as is used on Windows, such as Ctrl+C for copying, you need to edit some preferences. Start MATLAB, go to Home Tab and select *Preferences*. In the list on the left, choose *Keyboard -> Shortcuts* and change the *Active settings:* to *Windows default set*.

When you add Paths to MATLAB and try to save it, you may promted to save ``pathdef.m`` to a different location. A good place would be the Home folder, so that by opening the terminal with ``Ctrl+Alt+T`` you can just type in ``matlab`` and it should load the correct paths.

Install Klustakwik
------------------

Use the following terminal commands to install Klustakwik

.. code-block:: none

	cd Programs
	git clone https://github.com/klusta-team/klustakwik
	cd klustakwik
	make


Install Scripts from GitHub
---------------------------

You have likely already downloaded the scripts for processing Open Ephys recording from GitHub, if you worked through this guide: :ref:`RecordingManagerSetup`. In that case, you should have in your *Home* folder a folder called ``openEPhys_DACQ`` in which there are also scripts for processing recorded data. If not, use the referred guide to download the files into folder structure just described.

You will need to download *scikit-learn* package for python to use these scripts. To do so, use the following terminal command ``pip install -U scikit-learn``.

You will should not need to do this step if you installed Klustakwik to the location ``~/Programs/klustakwik/`` as you would have if you followed the commands above to the letter. Edit file ``ApplyKlustakwikScripts.py`` to specify to location of the *Klustakwik* install. Open the file with text editor, e.g. with terminal command ``gedit ~/openEPhys_DACQ/ApplyKlustakwikScripts.py`` and find the line where ``kk_path`` is specified (probably line 132). It should say something like ``kk_path = r'/home/room418/Programs/klustakwik/KlustaKwik'``. Edit the path to where you installed Klustakwik.

Install Google Chrome to use Waveform GUI
-----------------------------------------

Go to Google Chrome `website <https://www.google.com/chrome/>`_ and download 64 bit *.deb* package for Ubuntu. Once the download is finished, go to the downloaded file and Right Click -> Open With -> GDebi Package Installer. Click Install Package.

You are now ready to use the Open Ephys Data Processing scripts to detect spike, cluster them with Klustakwik and view the result with spatial correlograms in `Waveform GUI <http://d1manson.github.io/waveform/>`_.

See :ref:`howToUseTheseScripts`.