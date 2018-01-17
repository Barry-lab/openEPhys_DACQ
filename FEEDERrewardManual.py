# Call this program with RPiNr as an argument (1, 2, 3 or 4)
# A tone is played followed by opening of the valve on the correct FEEDER for specified duration.
# A Signal is sent to OpenEphysGUI NetworkEvents module.

# Program can be stopped by entering 0 as duration.

import pygame
from sshScripts import ssh
import sys
import zmq

n_rpi = int(sys.argv[1]) - 1

# Prepare SSH connection to FEEDER
RPiIPs = ['192.168.0.200', '192.168.0.201', '192.168.0.202', '192.168.0.203']
RPiUsername = 'pi'
RPiPassword = 'raspberry'
ssh_connection = ssh(RPiIPs[n_rpi], RPiUsername, RPiPassword)
ssh_connection.sendCommand('pkill python') # Ensure any past processes have closed
# Prepare pygame to play sound
pygame.mixer.init()
pygame.mixer.music.load('audiocheck.net_sin_2800Hz_-3dBFS_10s.wav')
# Set up ZMQ to send timestamps to OpenEphysGUI
ip = 'localhost'
port = 5556
url = "tcp://%s:%d" % (ip, port)
with zmq.Context() as context:
    with context.socket(zmq.REQ) as socket:
        socket.connect(url)
        # Run a loop until user inputs 0
        duration = 1
        while duration != 0:
            # Get user input for opening duration and timing
            duration = raw_input('Enter opening duration: ')
            duration = float(duration)
            if duration != 0:
                # Play sound from Recording PC speaker
                pygame.mixer.music.play()
                # Send timestamp to OpenEphysGUI
                socket.send('Tone: ' + str(n_rpi + 1))
                dump_response = socket.recv()
                target = str(raw_input())
                if target != '0':
                    pygame.mixer.music.stop()
                    # Open the valve on the RPi
                    ssh_connection.sendCommand('nohup python openPinchValve.py ' + str(duration) + ' &')
                    # Send timestamp to OpenEphysGUI
                    socket.send('FEEDER: ' + str(n_rpi + 1) + ' duration: ' + str(duration))
                    dump_response = socket.recv()
                else:
                    pygame.mixer.music.stop()
                    # Send timestamp to OpenEphysGUI
                    socket.send('FEEDER: ' + str(n_rpi + 1) + ' cancelled')
                    dump_response = socket.recv()
