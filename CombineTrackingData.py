### This script combines position data from multiple cameras.
### It also corrects frame time offset errors in PosLog.csv files
### It also removes bad position data lines

### Use as follows:
### import CombineTrackingData as combPos
### combPos.combdata(Path-To-Recording-Folder)

### By Sander Tanni, May 2017, UCL

import numpy as np
import OpenEphys
import pickle
import os


def findPosLogs(rootfolder):
    # Finds the PosLog*.csv files in the recording folder
    RPi_nrs = []
    filenames = []
    # Work thorugh all files in folder
    for fname in os.listdir(rootfolder):
        if fname.startswith('PosLog') and fname.endswith('.csv') and 'Comb' not in fname:
            # Get the RPi number if PosLog file
            numstart = 6
            numend = fname.find('.csv')
            if numstart == -1 or numend == -1:
                print('Error: Unexpected file name')
            else:
                filenames.append(rootfolder + '/' + fname)
                RPi_nrs.append(int(fname[numstart:numend]))

    return RPi_nrs, filenames


def getPosData(RPi_nr, filename, OEdict, RPiSettings):
    # This function processes PosLog*.csv files.
    # Offset between actual frame time and TTL pulses are corrected.
    # If PosLog*.csv has more datapoints than TTL pulses recorded, the PosLog datapoints from the end are dropped.
    # Bad position data lines are dropped (if LED luminance drops too much or position is outside designated arena)
    RPiTime2Sec = 10 ** 6 # This values is used to convert RPi times to seconds
    # Read position data for this camera
    pos_csv = np.genfromtxt(filename, delimiter=',')
    pos_csv = np.delete(pos_csv, (0), axis=0) # Cut out the header row
    pos_csv[0,2] = 0 # Set first frametime to 0
    # Read OpenEphys frame times for this camera in seconds
    timestamps_2_use = np.logical_and(OEdict['eventId'] > 0.5, np.int8(OEdict['channel']) == RPi_nr)
    # Substract first timestamp value to align Position times to the beginning of LFP traces
    OEdict['timestamps'] = OEdict['timestamps'] - OEdict['timestamps'][0]
    OEtimes = np.float64(OEdict['timestamps'][timestamps_2_use]) / OEdict['header']['sampleRate']
    if OEtimes.size != pos_csv.shape[0]: # If PosLog*.csv has more datapoints than TTL pulses recorded 
        # Realign frame times between OpenEphys and RPi by dropping the extra datapoints in PosLog data
        offset = pos_csv.shape[0] - OEtimes.size
        pos_csv = pos_csv[:OEtimes.size,:]
        print('WARNING! Camera ' + str(RPi_nr) + ' Pos data longer than TTL pulses recorded by ' + str(offset))
        print('Assuming that OpenEphysGUI was stopped before cameras stopped.')
        print(str(offset) + ' datapoints deleted from the end of position data.')
    # Get pos_csv frametimes and TTL times in seconds
    pos_frametimes = np.float64(pos_csv[:,2]) / RPiTime2Sec
    pos_TTLtimes = np.float64(pos_csv[:,1]) / RPiTime2Sec
    # Use pos_frametimes and pos_TTLtimes differences to correct OEtimes
    RPiClockOffset = np.mean(pos_TTLtimes - pos_frametimes)
    times = OEtimes - (pos_TTLtimes - pos_frametimes - RPiClockOffset)
    # Find bad pos data lines
    idxBad = np.zeros(times.size, dtype=bool)
    primary_lum = pos_csv[:,7] # Lines with severly reduced luminance on primary LED
    idxBad = np.logical_or(idxBad, primary_lum < np.median(primary_lum) * 0.75)
    arena_size = RPiSettings['arena_size'] # Points beyond arena size
    x_too_big = np.logical_or(pos_csv[:,3] > arena_size[0] + 10, pos_csv[:,5] > arena_size[0] + 10)
    y_too_big = np.logical_or(pos_csv[:,4] > arena_size[1] + 10, pos_csv[:,6] > arena_size[1] + 10)
    idxBad = np.logical_or(idxBad, np.logical_or(x_too_big, y_too_big))
    x_too_small = np.logical_or(pos_csv[:,3] < -10, pos_csv[:,5] < -10)
    y_too_small = np.logical_or(pos_csv[:,4] < -10, pos_csv[:,6] < -10)
    idxBad = np.logical_or(idxBad, np.logical_or(x_too_small, y_too_small))
    # Combine good position data lines
    times = times[np.logical_not(idxBad)]
    pos_csv = pos_csv[np.logical_not(idxBad),:]
    posdata = np.concatenate((np.expand_dims(times, axis=1), pos_csv[:,3:7]), axis=1)

    return posdata


def savedata(posdata, rootfolder):
    # Save as posdata as .csv file
    CombFileName = rootfolder + '/PosLogComb.csv'
    with open(CombFileName, 'wb') as f:
        np.savetxt(f, posdata, delimiter=',')
    print('Position data saved as ' + CombFileName)


def combdata(rootfolder):
    # This main function utilizes all the other functions in this script
    RPi_nrs, filenames = findPosLogs(rootfolder) # Get PosLog*.csv file names
    # Get RPiSettings
    RPiSettingsFile = rootfolder + '/CameraData' + str(RPi_nrs[0]) + '/RPiSettings.p'
    with open(RPiSettingsFile, 'rb') as f:
        RPiSettings = pickle.load(f)
    # Get OpenEphysGUI events data (TTL pulse times in reference to electrophysiological signal)
    eventsfile = rootfolder + '/all_channels.events'
    OEdict = OpenEphys.loadEvents(eventsfile)
    # Process position data from all PosLog*.csv's from all cameras
    posdatas = []
    for n_rpi in range(len(RPi_nrs)):
        posdata = getPosData(RPi_nrs[n_rpi], filenames[n_rpi], OEdict, RPiSettings)
        posdatas.append(posdata)
    ######## Figure out how to combine information from multiple RPis
    savedata(posdatas[0], rootfolder)