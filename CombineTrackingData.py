### This script combines position data from multiple cameras.
### It also corrects frame time offset errors in PosLog.csv files
### It also removes bad position data lines

### Use as follows:
### import CombineTrackingData as combPos
### combPos.combdata(Path-To-Recording-Folder)

### By Sander Tanni, May 2017, UCL

import numpy as np
import pickle
import os
import NWBio


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


def getPosData(RPi_nr, filename, data_events, RPiSettings):
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
    OEtimes = data_events['timestamps'][np.array(data_events['eventID']) == RPi_nr + 1] # Use the timestamps where RPi sent pulse to OE board
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
    # Combine corrected timestamps with position data
    posdata = np.concatenate((np.expand_dims(times, axis=1), pos_csv[:,3:7]), axis=1).astype(np.float32)

    return posdata


def savedata(posdata, rootfolder):
    # Save as posdata as .csv file
    CombFileName = rootfolder + '/PosLogComb.csv'
    with open(CombFileName, 'wb') as f:
        np.savetxt(f, posdata, delimiter=',')
    print('Position data saved as ' + CombFileName)


def combineCamerasData(cameraPos, lastCombPos=None, RPiSettings=None):
    # This outputs position data based on which camera is closest to tracking target.

    # cameraPos - list of numpy vecors with 4 elements (x1,y1,x2,y2) for each camera
    # lastCombPos - Last known output from this function
    # RPiSettings - settings file saved by CameraSettingsGUI.py

    # Output - numpy vector (x1,y1,x2,y2) with data from closest camera

    # If lastCombPos is not provided, the function will attempt to locate the animal
    # simultaneously from at least 2 cameras with smaller separation than half of RPiSettings['camera_transfer_radius']
    #   If successful, closest mean coordinate is set as output
    #   If unsuccessful, output is None

    N_RPis = len(cameraPos)
    cameraPos = np.array(cameraPos, dtype=np.float32)
    camera_relative_locs = []
    for n_rpi in range(N_RPis):
        camera_relative_locs.append(np.fromstring(RPiSettings['RPi_location'][n_rpi],dtype=float,sep=','))
    camera_relative_locs = np.array(camera_relative_locs, dtype=np.float32)

    # Only work with camera data from inside the enviornment
    # Find bad pos data lines
    idxBad = np.zeros(cameraPos.shape[0], dtype=bool)
    arena_size = RPiSettings['arena_size'] # Points beyond arena size
    x_too_big = cameraPos[:,0] > arena_size[0] + 20
    y_too_big = cameraPos[:,1] > arena_size[1] + 20
    idxBad = np.logical_or(idxBad, np.logical_or(x_too_big, y_too_big))
    x_too_small = cameraPos[:,0] < -20
    y_too_small = cameraPos[:,1] < -20
    idxBad = np.logical_or(idxBad, np.logical_or(x_too_small, y_too_small))
    # Only keep camera data from within the environment
    N_RPis = np.sum(np.logical_not(idxBad))
    # Only continue if at least one RPi data remains
    if N_RPis > 0:
        cameraPos = cameraPos[np.logical_not(idxBad),:]
        camera_relative_locs = camera_relative_locs[np.logical_not(idxBad),:]
        if np.any(lastCombPos):
            # Check which cameras provide data close enough to lastCombPos
            RPi_correct = []
            for n_rpi in range(N_RPis):
                lastCombPos_distance = euclidean(cameraPos[n_rpi, :2], lastCombPos[:2])
                RPi_correct.append(lastCombPos_distance < RPiSettings['camera_transfer_radius'])
            RPi_correct = np.array(RPi_correct, dtype=bool)
            # If none were found to be withing search radius, set output to None
            if not np.any(RPi_correct):
                combPos = None
            else:
                # Use the reading from closest camera to target mean location that detects correct location
                if np.sum(RPi_correct) > 1:
                    # Only use correct cameras
                    N_RPis = np.sum(RPi_correct)
                    cameraPos = cameraPos[RPi_correct, :]
                    camera_relative_locs = camera_relative_locs[RPi_correct, :]
                    meanPos = np.mean(cameraPos[:, :2], axis=0)
                    # Find mean position distance from all cameras
                    cam_distances = []
                    for n_rpi in range(N_RPis):
                        camera_loc = camera_relative_locs[n_rpi, :] * np.array(RPiSettings['arena_size'])
                        cam_distances.append(euclidean(camera_loc, meanPos))
                    # Find closest distance camera and output its location coordinates
                    closest_camera = np.argmin(np.array(cam_distances))
                    combPos = cameraPos[closest_camera, :]
                else:
                    # If target only detected close enough to lastCombPos in a single camera, use it as output
                    combPos = cameraPos[np.where(RPi_correct)[0][0], :]
        else:
            # If no lastCombPos provided, check if position can be verified from more than one camera
            #### NOTE! This solution breaks down if more than two cameras incorrectly identify the same object
            ####       as the brightes spot, instead of the target LED.
            cameraPairs = []
            pairDistances = []
            for c in combinations(range(N_RPis), 2):
                pairDistances.append(euclidean(cameraPos[c[0], :2], cameraPos[c[1], :2]))
                cameraPairs.append(np.array(c))
            cameraPairs = np.array(cameraPairs)
            cameraPairs_Match = np.array(pairDistances) < (RPiSettings['camera_transfer_radius'] / 2)
            # If position can not be verified from multiple cameras, set output to none
            if not np.any(cameraPairs_Match):
                combPos = None
            else:
                # Otherwise, set output to mean of two cameras with best matching detected locations
                pairToUse = np.argmin(pairDistances)
                camerasToUse = np.array(cameraPairs[pairToUse, :])
                combPos = np.mean(cameraPos[camerasToUse, :2], axis=0)
                # Add NaN values for second LED
                combPos = np.append(combPos, np.empty(2) * np.nan)
    else:
        combPos = None

    return combPos


def combdata(filename):
    # filename - the full path to the raw data file
    # Get data root folder
    rootfolder = filename[:filename.rfind('/')]
    # This main function utilizes all the other functions in this script
    RPi_nrs, filenames = findPosLogs(rootfolder) # Get PosLog*.csv file names
    # Get RPiSettings
    RPiSettingsFile = rootfolder + '/CameraData' + str(RPi_nrs[0]) + '/RPiSettings.p'
    with open(RPiSettingsFile, 'rb') as f:
        RPiSettings = pickle.load(f)
    # Get OpenEphysGUI events data (TTL pulse times in reference to electrophysiological signal)
    data_events = NWBio.load_events(filename)
    # Process position data from all PosLog*.csv's from all cameras
    posdatas = []
    for n_rpi in range(len(RPi_nrs)):
        posdata = getPosData(RPi_nrs[n_rpi], filenames[n_rpi], data_events, RPiSettings)
        posdatas.append(posdata)
    if len(posdatas) > 1:
        PosDataFramesPerSecond = 20.0
        # Find first and last timepoint for position data
        first_timepoints = []
        last_timepoints = []
        for posdata in posdatas:
            first_timepoints.append(posdata[0,0])
            last_timepoints.append(posdata[-1,0])
        # Combine position data step-wise from first to last timepoint at PosDataFramesPerSecond
        # At each timepoint the closest matchin datapoints will be taken from different cameras
        timepoints = np.arange(np.array(first_timepoints).min(), np.array(last_timepoints).max(), 1.0 / PosDataFramesPerSecond)
        combPosData = [None]
        listNaNs = []
        for npoint in range(len(timepoints)):
            # Find closest matchin timepoint from all RPis
            idx_tp = np.zeros(4, dtype=np.int32)
            for n_rpi in range(len(posdatas)):
                idx_tp[n_rpi] = np.argmin(np.abs(timepoints[npoint] - posdatas[n_rpi][:,0]))
            # Convert posdatas for use in combineCamerasData function
            cameraPos = []
            for n_rpi in range(len(posdatas)):
                cameraPos.append(posdatas[n_rpi][idx_tp[n_rpi], 1:5])
            tmp_comb_data = combineCamerasData(cameraPos, combPosData[-1], RPiSettings)
            combPosData.append(tmp_comb_data)
            if tmp_comb_data is None:
                listNaNs.append(npoint)
        # Remove the extra element from combPosData
        del combPosData[0]
        # Remove all None elements
        for nanElement in listNaNs[::-1]:
            del combPosData[nanElement]
        timepoints = np.delete(timepoints, listNaNs)
        # Save timepoints and combined position data
        posdata = np.concatenate((np.expand_dims(np.array(timepoints), axis=1), np.array(combPosData)), axis=1)
        savedata(posdata, rootfolder)
        # Print info about None elements
        if len(listNaNs) > 0:
            print('Total of ' + str(len(listNaNs) * (1.0 / PosDataFramesPerSecond)) + ' seconds of position data was lost')
            if listNaNs[0] == 0:
                print('This was in the beginning and ' + str(np.sum(np.diff(np.array(listNaNs)) > 1)) + ' other epochs.')
            else:
                print('This was not in the beginning, but in ' + str(np.sum(np.diff(np.array(listNaNs)) > 1) + 1) + ' other epochs.')
    else:
        # In case of single camera being used, simply delete data outside enivornmental boundaries
        posdata = posdatas[0]
        # Find bad pos data lines
        idxBad = np.zeros(posdata.shape[0], dtype=bool)
        arena_size = RPiSettings['arena_size'] # Points beyond arena size
        x_too_big = np.logical_or(posdata[:,1] > arena_size[0] + 20, posdata[:,3] > arena_size[0] + 20)
        y_too_big = np.logical_or(posdata[:,2] > arena_size[1] + 20, posdata[:,4] > arena_size[1] + 20)
        idxBad = np.logical_or(idxBad, np.logical_or(x_too_big, y_too_big))
        x_too_small = np.logical_or(posdata[:,1] < -20, posdata[:,3] < -20)
        y_too_small = np.logical_or(posdata[:,2] < -20, posdata[:,4] < -20)
        idxBad = np.logical_or(idxBad, np.logical_or(x_too_small, y_too_small))
        # Combine good position data lines
        posdata = posdata[np.logical_not(idxBad),:]
        # Save corrected data
        savedata(posdata[:,:5], rootfolder)