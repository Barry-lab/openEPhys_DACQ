# Call this program with RPiNr as an argument (1, 2, 3 or 4)
# A tone is played followed by opening of the valve on the correct FEEDER for specified duration.
# A Signal is sent to OpenEphysGUI NetworkEvents module.

# Program can be stopped by entering 0 as duration.

from sshScripts import ssh
import zmq

class Pellets(object):
    # This class allows control of pellet feeders
    def __init__(self, n_feeder):
        # Edit these to correspond to the RPi Zeros controlling the feeders
        RPiIPs = ['192.168.0.200', '192.168.0.201', '192.168.0.202', '192.168.0.203']
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
