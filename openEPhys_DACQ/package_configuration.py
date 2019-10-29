
import os
from shutil import rmtree
import json


# TODO: KiloSort path currently hardcoded in StandardConfig.m. Should be done here instead.


package_name = 'openEPhys_DACQ'

package_path = os.path.dirname(os.path.realpath(__file__))


class PackageConfiguration(object):

    config_file_path = os.path.join(os.path.expanduser('~'), '.{}_Config.json'.format(package_name))

    default_config = {
        'root_folder': os.path.join(os.path.expanduser('~'), 'RecordingData'),
        'recording_manager_settings_subfolder': 'RecordingManagerSettings',
        'klustakwik_path': 
            os.path.join(os.path.expanduser('~'), 'Programs', 'klustakwik', 'KlustaKwik'),
        'kilosort_path': '',
        'npy_matlab_path': ''
    }

    def __init__(self, reconfigure=False):

        if not reconfigure and os.path.isfile(self.config_file_path):

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

            input('Configuration completed.\n'
                  + 'Press Enter to exit:')

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

            if os.path.isfile(alternative_path):

                raise ValueError('The entered working directory path {} leads to a file'.format(
                    os.path.dirname(alternative_path)
                ))

            elif os.path.isdir(os.path.dirname(alternative_path)):

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

    def configure_klustakwik_path(self):
        
        print('To use KlustaKwik for processing tetrode recordings,'
              + 'working KlustaKwik installation is required')
        
        decision = input('Would you like to use the default KlustaKwik executable path\n'
                         + self.config['klustakwik_path']
                         + '[Y/n] ')

        if decision.lower() == 'y' or decision.lower() == 'yes' or decision == '':

            klustakwik_path = self.config['klustakwik_path']

        else:

            klustakwik_path = input('Please enter full path to KlustaKwik executable: ')

        if os.path.isfile(klustakwik_path):

            self.config['klustakwik_path'] = klustakwik_path

        else:

            raise ValueError('The entered KlustaKwik executable path {} does not lead to a file'.format(
                os.path.dirname(klustakwik_path)
            ))

    def configure_kilosort_path(self):
        
        print('To use KiloSort for processing tetrode recordings,'
              + 'paths to KiloSort and npy-matlab repositories are required.')

        decision = input('Would you like to set up KiloSort by providing paths to said repositories? [y/N] ')

        if decision.lower() != 'y' and decision.lower() != 'yes':

            return

        kilosort_path = input('Please enter full path to KiloSort repository: ')

        if os.path.isdir(kilosort_path):

            self.config['kilosort_path'] = kilosort_path

        else:

            raise ValueError('Provided path {} is not a folder'.format(kilosort_path))

        npy_matlab_path = input('Please enter full path to npy-matlab repository: ')

        if os.path.isdir(npy_matlab_path):

            self.config['npy_matlab_path'] = npy_matlab_path

        else:

            raise ValueError('Provided path {} is not a folder'.format(npy_matlab_path))

    def configure_package(self):

        self.config = self.default_config

        self.configure_root_folder()

        self.configure_klustakwik_path()

        self.configure_kilosort_path()


def package_config():
    return PackageConfiguration().config


def main():
    PackageConfiguration(reconfigure=True)


if __name__ == '__main__':
    main()
