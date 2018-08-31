### This script is used to send commands to RPi over an SSH connection.

# Initiate connection by initating the class as connection = ssh(serverIP,sshUsername,sshPassword)
# Then send commands with connection.sendCommand('type_command_here')

from paramiko import client
from paramiko.ssh_exception import SSHException
from threading import Thread

class ssh:
    # This class was taken from:
    # https://daanlenaerts.com/blog/2016/01/02/python-and-ssh-sending-commands-over-ssh-using-paramiko/
    # 09/12/16
    client = None

    def __init__(self, address, username, password):
        self.address = address
        self.client = client.SSHClient()
        self.client.set_missing_host_key_policy(client.AutoAddPolicy())
        self.client.connect(address, username=username, password=password, look_for_keys=False)
        self.Ts_sendCommand = []

    def sendCommand(self, command, timeout=5, verbose=True):
        if(self.client):
            try:
                stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
                if verbose:
                    while not stdout.channel.exit_status_ready():
                        # Print data when available
                        if stdout.channel.recv_ready():
                            alldata = stdout.channel.recv(1024)
                            prevdata = b"1"
                            while prevdata:
                                prevdata = stdout.channel.recv(1024)
                                alldata += prevdata
                            print(str(alldata))
            except SSHException as e:
                if e.__str__() == 'Timeout opening channel.':
                    print('ERROR: Command timeout!' + '\nFailed to send command: ' + command + ' | To: ' + self.address)
                elif e.__str__() == 'SSH session not active':
                    print('ERROR: Connection lost!' + '\nFailed to send command: ' + command + ' | To: ' + self.address)
                else:
                    raise e
        else:
            print("Connection not opened.")

    def sendCommand_threading(self, command, timeout=5, verbose=True):
        T = Thread(target=self.sendCommand, args=(command, timeout, verbose))
        T.start()
        self.Ts_sendCommand.append(T)

    def testConnection(self, timeout=5):
        '''
        Returns True if connection is active. False if connection inactive until timeout or session inactive.
        '''
        if(self.client):
            try:
                stdin, stdout, stderr = self.client.exec_command('ls', timeout=timeout)
                return True
            except SSHException as e:
                if e.__str__() == 'Timeout opening channel.':
                    return False
                elif e.__str__() == 'SSH session not active':
                    return False
                else:
                    raise e
        else:
            return False

    def disconnect(self, force=False):
        if(self.client):
            if not force and len(self.Ts_sendCommand) > 0:
                for T in self.Ts_sendCommand:
                    T.join()
            self.client.close()
        else:
            print("Connection not opened.")
