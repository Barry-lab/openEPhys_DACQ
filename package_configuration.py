
import os
from shutil import rmtree
import json


package_name = 'openEPhys_DACQ'

package_path = os.path.dirname(os.path.realpath(__file__))


class PackageConfiguration(object):

    config_file_path = os.path.join(os.path.expanduser('~'), '.{}_Config.json'.format(package_name))

    default_config = {
        'root_folder': os.path.join(os.path.expanduser('~'), 'RecordingData'),
        'recording_manager_settings_subfolder': 'RecordingManagerSettings'
    }

    def __init__(self):

        if os.path.isfile(self.config_file_path):

            with open(self.config_file_path, 'r') as config_file:
                self.config = json.loads(config_file.read())

        else:

            print('Configuration file for {} not found at expected location {}'.format(package_name,
                                                                                       self.config_file_path))
            print('Assuming this is the first time {} is run on this system'.format(package_name))
            print('Entering configuration mode...\n')

            self.configure_package()

            with open(self.config_file_path, 'w') as config_file:
                json.dump(self.config, config_file, sort_keys=True, indent=4)

            input('Configuration completed. Press Enter to continue:')

    def configure_package(self):

        self.config = self.default_config

        self.configure_root_folder()

    def configure_root_folder(self):

        print('This program requires a working directory [root_folder] for:\n'
              + ' - Storing recorded data\n'
              + ' - Storing past settings of Recording Manager\n\n'
              + 'All existing data in the chosen working directory [root_folder]\n'
              + 'path will be deleted.\n'
              + 'The default path for working directory [root_folder] is {}'.format(self.config['root_folder']))

        decision = input('Use the default path for working directory [root_folder]? [y/N] ')

        if decision.lower() != 'y' and decision.lower() != 'yes':

            alternative_path = input('Please enter full path to alternative working directory [root_folder]: ')

            if os.path.isdir(os.path.dirname(alternative_path)):

                self.config['root_folder'] = alternative_path

            else:

                raise ValueError('The entered working directory path {} does not exist'.format(
                    os.path.dirname(alternative_path)
                ))

        if os.path.isdir(self.config['root_folder']):
            rmtree(self.config['root_folder'])
        os.mkdir(self.config['root_folder'])

        self.create_root_folder_subfolders()

        print('Working directory [root_folder] set as {}'.format(self.config['root_folder']))
        print('Recorded data will be found in directory {}'.format(os.path.join(self.config['root_folder'],
                                                                                'RecordingData')))

    def create_root_folder_subfolders(self):

        os.mkdir(os.path.join(self.config['root_folder'],
                              self.config['recording_manager_settings_subfolder']))


def package_config():
    return PackageConfiguration().config
