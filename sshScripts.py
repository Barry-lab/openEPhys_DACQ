### This script is used to send commands to RPi over an SSH connection.

# Initiate connection by initating the class as connection = ssh(serverIP,sshUsername,sshPassword)
# Then send commands with connection.sendCommand('type_command_here')

from paramiko import client

class ssh:
    # This class was taken from:
    # https://daanlenaerts.com/blog/2016/01/02/python-and-ssh-sending-commands-over-ssh-using-paramiko/
    # 09/12/16
    client = None

    def __init__(self, address, username, password):
        self.client = client.SSHClient()
        self.client.set_missing_host_key_policy(client.AutoAddPolicy())
        self.client.connect(address, username=username, password=password, look_for_keys=False)

    def sendCommand(self, command):
        if(self.client):
            stdin, stdout, stderr = self.client.exec_command(command)
            while not stdout.channel.exit_status_ready():
                # Print data when available
                if stdout.channel.recv_ready():
                    alldata = stdout.channel.recv(1024)
                    prevdata = b"1"
                    while prevdata:
                        prevdata = stdout.channel.recv(1024)
                        alldata += prevdata

                    print(str(alldata))
        else:
            print("Connection not opened.")

