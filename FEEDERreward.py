# Call this function with arguments RPiNr and OpeningDuration as int and float (seconds) respectively.
# A tone is played followed by opening of the valve on the correct FEEDER for specified duration.

import pygame
import time
from sshScripts import ssh
import sys

n_rpi = int(sys.argv[1]) - 1

duration = float(sys.argv[2])

RPiIPs = ['192.168.0.200', '192.168.0.201', '192.168.0.202', '192.168.0.203']

RPiUsername = 'pi'
RPiPassword = 'raspberry'

connection = ssh(RPiIPs[n_rpi], RPiUsername, RPiPassword)
connection.sendCommand('pkill python') # Ensure any past processes have closed

pygame.mixer.init()
pygame.mixer.music.load('audiocheck.net_sin_2800Hz_-3dBFS_2s.wav')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue

connection.sendCommand('nohup python openPinchValve.py ' + str(duration) + ' &')