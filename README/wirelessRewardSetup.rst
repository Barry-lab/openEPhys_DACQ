.. _wirelessRewardSetup:

============
FEEDER setup
============

FEEDER - Food Ejector Executing Directions Established via Radio
or Wireless Reward Module.

Setting up Raspberry Pi Zero software
=====================================

It is advisable to follow through this part first, before assembling the FEEDER. This way it is more convenient to plug the RPi Zero to the power socket, keyboard, mouse and monitor for working through the following steps.

This tutorial is based on setting up `Raspberry Pi Zero W <https://www.raspberrypi.org/products/raspberry-pi-zero-w/>`_ with Recording PC that has a wireless networking adapter and a Wi-Fi router.

If you are setting up multiple Pi Zeros, you can make copies after you have set up the software and networking on the first one. To make copies of a configured RPi, use this guide: :ref:`duplicatingRPis`. Ensure to change the static IP information on each new copy, as instructed in the guide.

Installing Raspberry Pi OS: Raspbian
------------------------------------

Download **Raspbian Jessie With Pixel** (Here used version 4.9) from `Raspberry website <ttps://www.raspberrypi.org/downloads/raspbian/>`_. Unpack the downloaded *.zip* file with ``Archive Manager`` to get access to the *.img* file.

Download **Etcher** from this `website <https://etcher.io/>`_. Unpack the downloaded *.zip* and run (double-click) the Etcher *.AppImage*. Select the Raspbian *.img* file and your inserted microSD card. Write the image (*Flash* button).

Insert the microSD card into Raspberry Pi.

Install necessary software on RPi Zero
--------------------------------------

Connect your Raspberry Pi Zero W with a keyboard/mouse, monitor and power socket. Note, RPi Zero has no regular USB output, therefore you will need to use an adaptor to connect the keyboard and mouse. You will also need a mini HDMI to HDMI adaptor to connect your monitor.

You need to connect your RPi Zero to the internet to install the necessary software. This can be done using a Wi-Fi connection, for example, a Wi-Fi hotspot created by your laptop or smartphone. Preferably use a network you can delete later, so that the RPi Zero would not be confused about which network to connect to: the one with internet connection or the one you set up to talk to the Recording PC. You can connect your RPi Zero to a Wi-Fi network as a common laptop. As you boot up the RPi Zero, look for the networking icon on top right and connect with the correct password for the network.

To update the RPi Zero software with the following terminal commands:

.. code-block:: none

	sudo apt-get update        # Fetches the list of available updates
	sudo apt-get upgrade       # Strictly upgrades the current packages
	sudo apt-get dist-upgrade  # Installs updates (new ones)

Install the necessary libraries for working with the Picon Zero controller with the following terminal commands:

.. code-block:: none

	sudo apt-get install python-smbus python3-smbus python-dev python3-dev
	wget -q http://4tronix.co.uk/piconzero/piconzero.py -O piconzero.py

You will need to edit the file ``/boot/config.txt``. Do this with terminal command ``sudo leafpad /boot/config.txt`` and add the following lines to the end of this file:

.. code-block:: none

	dtparam=i2c1=on
	dtparam=i2c_arm=on

Save the file and reboot the RPi Zero.

If you have assembled the RPi Zero with Picon Zero, you are able to test if everything is running smoothly by entering terminal command ``i2cdetect -y 1``. You should see an output table with empty values everywhere but one element, which should say ``22``.

Finally, :download:`this script (openPinchValve.py) <../RecordingPCsetup/openPinchValve.py>` to the  RPi Zero home folder ``/home/pi``. The script is called by Recording PC to open the valve for a specified amount of time.

Setting up Raspberry Pi Zero networking with Recording PC
=========================================================

There are many ways the netowork could be set up. In our case, we have a wireless card in Recording PC, but it is too weak to create a reliable hotspot. Therefore we have set up a standard router to which we connect the Recording PC and RPi Zeros, both via Wi-Fi.

Setting up Networking with RPi Zero and Recording PC
----------------------------------------------------

Turn on the router and connect to it with the Recording PC using the router's ID (SSID, the Wi-Fi network name) and password, which are likely printed on the router. 

You may wish to change the router's ID and the password through its graphical interface. You can access this by entering the router's IP in the browser address bar, after you have joined the Wi-Fi network. In our case the router's user's manual specified that the default IP for the router would be ``192.168.0.1``. The username and password should also be printed on the router. In our case we changed the SSID (Wi-Fi network name) to ``r418`` and the password to randomly generated long string. After these changes you will need to reconnect to the router with the Recording PC.

Connect to the Wi-Fi network ``r418`` with the RPi Zero using the graphical interface. Click on the network connections icon on top right of the screen and select ``r418``. Use the same password that was used by Recording PC to connect. To ensure RPi Zero connects to Wi-Fi automatically, you should edit the network interfaces file with terminal command ``sudo leafpad /etc/network/interfaces`` and find the line that says ``allow-hotplug wlan0`` and add just before it a line that says ``auto wlan0``, such that it would read:

.. code-block:: none

	auto wlan0
	allow-hotplug wlan0

Setting a static IP address on the RPi Zero
-------------------------------------------

The following instructions are based on this `video guide <https://www.youtube.com/watch?v=r3UIQXn8Zp0>`_. 

To set static IP address, you need to make changes to ``/etc/dhcpcd.conf``. For that you need to clarify the identity of the correct wireless device on your RPi Zero. Enter into terminal ``ifconfig`` and you should see network identities. It is most likely ``wlan0`` that has inforamtion such as ``Link encap:Ethernet``. You also need the IP address of the Wi-Fi router. This is the IP address you typed into your browser to find access the router's graphical interface. Alternatively, you can find the router's IP by entering the terminal command ``ip route show``. This should output information on your current connection and should show the Wi-Fi hotspot IP address as something like: ``default via 192.168.0.1``.

Now open ``dhcpcd.conf`` with terminal command ``sudo leafpad /etc/dhcpcd.conf``. Add to the very end of the file the following lines:

.. code-block:: none

	interface wlan0

	static ip_address=192.168.0.101/24
	static routers=192.168.0.1
	static domain_name_servers=192.168.0.1

Here you want to use the correct device identity that you found on the RPi Zero with ``ifconfig`` command, e.g. ``wlan0``. Set the ``router`` and ``domain_name_server`` values to the IP of the router that you found. Finally, the ``ip_address`` should be the same as the router, only the final value should be different, as in this example it is ``101``. If you have multipe RPi Zeros, set this to different value on each, e.g. 101, 102, 103 etc. The ``/24`` indicates the port number. Keep this the same in all cases.

As you save the changes to ``dhcpcd.conf`` and reboot your RPi Zero, it should connect to the Wi-Fi hotspot automatically and have the IP address you assigned. You can check this now with the ``ifconfig`` terminal command on the RPi Zero.

You should do this also on the Recording PC. You can follow the same exact steps, except using *gedit* instead of *leafpad* to edit the ``/etc/dhcpcd.conf``. The terminal command would be ``sudo gedit /etc/dhcpcd.conf``.

Now each time you connect the RPi Zero to a power supply, thereby powering it on, it should connect to the Wi-Fi network ``r418`` and have the IP you set.

Configuring SSH on RPi Zero
---------------------------

SSH needs to be enabled on RPi Zero. You can do this by accessing RPi settings via terminal command ``sudo raspi-config`` and choosing *Interfacing Options* with arrow keys and pressing Enter. Select *SSH* option and choose to *Enable* it. Restart RPi Zero.

The SSH login may be slow. This can be fixed by editing the ``sshd_config`` file. Open it with terminal command ``sudo leafpad /etc/ssh/sshd_config`` and add this line to the very end:

.. code-block:: none

	UseDNS no

Restart the RPi Zero

The following instructions allow connecting with the RPi via SSH without entering password each time. This is necessary for scripts on Recording PC. The instructions are based on `this guide <https://www.raspberrypi.org/documentation/remote-access/ssh/passwordless.md>`_. If you have previously generated the SSH key on the Recording PC, you can skip the first step of generating a new key.

Generate a new SSH key on Recording PC with terminal command ``ssh-keygen -t rsa -C recpc@pi``. Use the default location to save the key by pressing Enter. Leave the passphrase empty by pressing Enter.

Open terminal on Recording PC and enter the connect to your RPi using SSH with command ``ssh pi@192.168.0.101`` and enter ``raspberry`` as password. Enter this command in the terminal where you opened the SSH connection ``install -d -m 700 ~/.ssh``.

Now exit the SSH session or open a new terminal on Recording PC and enter this command ``cat ~/.ssh/id_rsa.pub | ssh pi@192.168.0.101 'cat >> .ssh/authorized_keys'``. Use the correct IP address (the numbers: ``192.168.0.101``) in that command for the IP address of the RPi you are connecting to. Enter the password ``raspberry`` for your RPi.

Now your RPi should be able to connect to the RPi via SSH without a password.

Now the RPi Zero software and networking is fully configured and after assembling the FEEDER, it will be ready to use with the Recording PC.

Assembling all parts of Wireless Reward Module
==============================================

The materials you need for the FEEDER:

Raspberry Pi Zero
Picon Zero
Current Converter and Battery Guard - HUBOSD eco X Type w/STOSD8 & XT60
LiPo battery with high current output
Solenoid Pinch Valve

Ensure you use the correct motor to match the command in openPinchValve.py, set motor 1.