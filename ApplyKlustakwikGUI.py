### This program finds all data in a selected folder produced by DetectWaveforms.py program
### and applies Klustakwik on this data.

### By Sander Tanni, February 2017, UCL


from PyQt4 import QtGui
import sys
import ApplyKlustakwikGUIDesign
import ntpath
import numpy as np
import os
import cPickle as pickle
import ApplyKlustakwikScripts as AKS
import createWaveformGUIdata as waveGUI


class ApplyKlustakwikGUI(QtGui.QMainWindow, ApplyKlustakwikGUIDesign.Ui_MainWindow):
    def __init__(self, parent=None):
        super(ApplyKlustakwikGUI, self).__init__(parent=parent)
        self.setupUi(self)
        # Make waveformGUI button inactive by default (enabled after Klustering)
        self.pb_waveformGUI.setEnabled(False)
        # Set up GUI interaction function calls
        self.pb_load_specific.clicked.connect(lambda:self.openFilesDialog())
        self.pb_load_all.clicked.connect(lambda:self.openFileDialog())
        self.pb_cluster_selection.clicked.connect(lambda:self.cluster_selection())
        self.pb_cluster_all.clicked.connect(lambda:self.cluster_all())
        self.pb_waveformGUI.clicked.connect(lambda:self.waveformGUIdata())
        self.lw_tetrodes.setSelectionMode(3)
        
        
    def openFilesDialog(self):
        # Pops up a GUI to select files for clustering
        dialog = QtGui.QFileDialog(self, caption='Select waveform files')
        dialog.setFileMode(QtGui.QFileDialog.ExistingFiles)
        dialog.setNameFilters(['(*.waveforms)'])
        dialog.setViewMode(QtGui.QFileDialog.List) # or Detail
        if dialog.exec_():
            # Get paths and file names of selection
            filenames = dialog.selectedFiles()
            self.fpath = ntpath.dirname(filenames[0])
            self.fileNames = []
            for filename in filenames:
                # Check if file name is correct format
                numstart = filename.find('_CH') + 3
                numend = filename.find('.waveforms')
                if numstart == -1 or numend == -1:
                    print('Error: Unexpected file name')
                else:
                    self.fileNames.append(str(ntpath.basename(filename)))
            # Load waveforms to memory if any selected files are suitable
            if not not self.fileNames:
                self.load_waveforms()
        
        
    def openFileDialog(self):
        # Pops up a GUI to select a single file. All others with same prefix will be loaded
        dialog = QtGui.QFileDialog(self, caption='Select one from the set of waveform files')
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        dialog.setNameFilters(['(*.waveforms)'])
        dialog.setViewMode(QtGui.QFileDialog.List) # or Detail
        if dialog.exec_():
            # Get path and file name of selection
            tmp = dialog.selectedFiles()
            selected_file = str(tmp[0])
            self.fpath = ntpath.dirname(selected_file)
            f_s_name = str(ntpath.basename(selected_file))
            # Check if file name is correct format
            numstart = f_s_name.find('_CH') + 3
            numend = f_s_name.find('.waveforms')
            if numstart == -1 or numend == -1:
                print('Error: Unexpected file name')
            else:
                # Get prefix for this set of LFPs
                fprefix = f_s_name[:f_s_name.index('_')]
                # Get list of all other files and channel numbers from this recording
                self.fileNames = []
                for fname in os.listdir(self.fpath):
                    if fname.startswith(fprefix) and fname.endswith('.waveforms'):
                        self.fileNames.append(fname)
                # Load waveforms to memory if any selected files are suitable
                if not not self.fileNames:
                    self.load_waveforms()
    
    
    def load_waveforms(self):
        # Loads waveforms from Pickle files selected with the openFileDialog or openFilesDialog
        self.waveform_data = []
        for filename in self.fileNames:
            # Load all selected waveform files
            full_filename = self.fpath + '/' + filename
            with open(full_filename, 'rb') as file:
                tmp = pickle.load(file)
            self.waveform_data.append(tmp)
        # Extract tetrode numbers and order data by tetrode numbers
        tetrode_numbers_int = []
        for wavedat in self.waveform_data:
            tetrode_numbers_int.append(wavedat['nr_tetrode'])
        file_order = np.argsort(np.array(tetrode_numbers_int))
        self.fileNames = [self.fileNames[x] for x in file_order]
        self.waveform_data = [self.waveform_data[x] for x in file_order]
        # Update info about open files in the textboxes
        self.update_textboxes()
        # Initialize list widget for tetrodes
        self.initialize_lw_tetrodes()
        
        
    def initialize_lw_tetrodes(self):
        # Inputs tetrode selection into the listbox
        self.lw_tetrodes.clear()
        ntet = 0
        for wavedat in self.waveform_data:
            # Get channel numbers into string form
            tetrode_channels_str = map(str, list(np.array(wavedat['tetrode_channels']) + 1))
            tetrode_number_str = str(wavedat['nr_tetrode'])
            # Format the item string using tetrode number and channel numbers
            string = 'T' + tetrode_number_str + ' C ' + \
                     ' '.join(tetrode_channels_str)
            # Put String into listbox format
            item = QtGui.QListWidgetItem(string)
            # # This added variable is necessary to identify selected list items
            item.ntet = ntet
            self.lw_tetrodes.addItem(item)
            ntet += 1
            
            
    def update_textboxes(self):
        # Displays selected folder and files in the textboxes
        self.tb_fpath.setPlainText(self.fpath)
        self.tb_files.setPlainText('\n'.join(self.fileNames))
        
        
    def cluster_selection(self):
        # Identifies which tetrodes are selected and calls clustering function on them
        tetrode_rows = []
        for item in self.lw_tetrodes.selectedItems():
            tetrode_rows.append(item.ntet)
        self.cluster(tetrode_rows)
        
        
    def cluster_all(self):
        # Calls clustering function for all tetrodes
        self.cluster(range(len(self.waveform_data)))
        # Set WaveformGUI button active
        self.pb_waveformGUI.setEnabled(True)
        
        
    def cluster(self, tetrode_rows):
        # Select features to use for clustering. Refer to ApplyKlustakwikScripts.py for more info
        features2use = ['PC1', 'PC2', 'PC3', 'Amp', 'Vt']
        # For each tetrode that was selected, run KlustKwik independently
        for ntet in tetrode_rows:
            print('Applying KlustaKwik on tetrode ' + str(self.waveform_data[ntet]['nr_tetrode']))
            full_filename = self.fpath + '/' + self.fileNames[ntet]
            # Get waveforms and features selection into correct format for the KlustKwikScripts
            waveforms = np.swapaxes(self.waveform_data[ntet]['waveforms'],1,2)
            if waveforms.shape[1] < 4:
                zerovalues = np.zeros((waveforms.shape[0],4-waveforms.shape[1],waveforms.shape[2]))
                waveforms = np.concatenate((waveforms, zerovalues), axis=1)
            d = {0: features2use}
            AKS.klustakwik(waveforms, d, full_filename)
            # Cross out items in the listwidget once they have been clustered
            self.cross_out_lw_tetrode(ntet)
            
            
    def cross_out_lw_tetrode(self, ntet):
        # Cross out items in the listwidget
        item = self.lw_tetrodes.takeItem(ntet)
        f = item.font()
        f.setStrikeOut(True)
        item.setFont(f)
        self.lw_tetrodes.insertItem(ntet,item)


    def waveformGUIdata(self):
        # Convert freshly created files to waveformGUIdata
        tmpFileNames = waveGUI.getAllFiles(self.fpath, self.fileNames)
        speedcut = self.sb_speed_cut.value()
        waveGUI.createWaveformData(self.fpath, tmpFileNames, speedcut=speedcut)


# The following is the default ending for a QtGui application script
def main():
    app = QtGui.QApplication(sys.argv)
    myapp = ApplyKlustakwikGUI()
    myapp.show()
    app.exec_()
    
if __name__ == '__main__':
    main()