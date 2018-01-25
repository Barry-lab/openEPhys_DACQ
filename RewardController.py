# Call this program with RPiNr as an argument (1, 2, 3 or 4)
# A tone is played followed by opening of the valve on the correct FEEDER for specified duration.
# A Signal is sent to OpenEphysGUI NetworkEvents module.

# Program can be stopped by entering 0 as duration.

from sshScripts import ssh

class Pellets(object):
    # This class allows control of pellet FEEDERs
    def __init__(self, n_feeder):
        # Edit these to correspond to the RPi Zeros controlling the FEEDERs
        RPiIPs = ['192.168.0.60', '192.168.0.61', '192.168.0.62', '192.168.0.63']
        RPiUsername = 'pi'
        RPiPassword = 'raspberry'
        self.n_feeder = n_feeder
        # Set up SSH connection
        print('Connecting to Pellet Feeder Nr: ' + str(n_feeder) + ' . . .')
        self.ssh_connection = ssh(RPiIPs[n_feeder], RPiUsername, RPiPassword)
        self.ssh_connection.sendCommand('pkill python') # Ensure any past processes have closed
        print('Connecting to Pellet Feeder Nr: ' + str(n_feeder) + ' successful')

    def release(self, n_pellets=1):
        self.ssh_connection.sendCommand('nohup python releasePellet.py ' + str(n_pellets) + ' &')

class Milk(object):
    # This class allows control of milk FEEDERs
    def __init__(self, n_feeder):
        # Edit these to correspond to the RPi Zeros controlling the FEEDERs
        RPiIPs = ['192.168.0.40', '192.168.0.41', '192.168.0.42', '192.168.0.43']
        RPiUsername = 'pi'
        RPiPassword = 'raspberry'
        self.n_feeder = n_feeder
        # Set up SSH connection
        print('Connecting to Milk Feeder Nr: ' + str(n_feeder) + ' . . .')
        self.ssh_connection = ssh(RPiIPs[n_feeder], RPiUsername, RPiPassword)
        self.ssh_connection.sendCommand('pkill python') # Ensure any past processes have closed
        print('Connecting to Milk Feeder Nr: ' + str(n_feeder) + ' successful')

    def release(self, duration=1.0):
        # Opening duration determines how much milk is released
        self.ssh_connection.sendCommand('nohup python openPinchValve.py ' + str(duration) + ' &')
