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
from itertools import combinations
from scipy.spatial.distance import euclidean

def combineCamerasData(cameraPos, lastCombPos=None, TrackingSettings=None):
    # This outputs position data based on which camera is closest to tracking target.

    # cameraPos - list of numpy vecors with 4 elements (x1,y1,x2,y2) for each camera
    # lastCombPos - Last known output from this function
    # TrackingSettings - settings file saved by CameraSettingsGUI.py

    # Output - numpy vector (x1,y1,x2,y2) with data from closest camera

    # If lastCombPos is not provided, the function will attempt to locate the animal
    # simultaneously from at least 2 cameras with smaller separation than half of TrackingSettings['camera_transfer_radius']
    #   If successful, closest mean coordinate is set as output
    #   If unsuccessful, output is None

    N_RPis = len(cameraPos)
    cameraPos = np.array(cameraPos, dtype=np.float32)
    camera_relative_locs = []
    for nRPi in range(N_RPis):
        n_rpi = TrackingSettings['use_RPi_nrs'][nRPi]
        camera_relative_locs.append(TrackingSettings['RPiInfo'][str(n_rpi)]['location'])
    camera_relative_locs = np.array(camera_relative_locs, dtype=np.float32)

    # Only work with camera data from inside the enviornment
    # Find bad pos data lines
    idxBad = np.zeros(cameraPos.shape[0], dtype=bool)
    arena_size = TrackingSettings['arena_size'] # Points beyond arena size
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
            for nRPi in range(N_RPis):
                lastCombPos_distance = euclidean(cameraPos[nRPi, :2], lastCombPos[:2])
                RPi_correct.append(lastCombPos_distance < TrackingSettings['camera_transfer_radius'])
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
                    for nRPi in range(N_RPis):
                        camera_loc = camera_relative_locs[nRPi, :] * np.array(TrackingSettings['arena_size'])
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
            cameraPairs_Match = np.array(pairDistances) < (TrackingSettings['camera_transfer_radius'] / 2)
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

def remove_tracking_data_outside_boundaries(posdata, arena_size, max_error=20):
    NotNaN = np.where(np.logical_not(np.isnan(posdata[:,1])))[0]
    idxBad = np.zeros(NotNaN.size, dtype=bool)
    x_too_big = posdata[NotNaN,1] > arena_size[0] + max_error
    y_too_big = posdata[NotNaN,2] > arena_size[1] + max_error
    idxBad = np.logical_or(idxBad, np.logical_or(x_too_big, y_too_big))
    x_too_small = posdata[NotNaN,1] < -max_error
    y_too_small = posdata[NotNaN,2] < -max_error
    idxBad = np.logical_or(idxBad, np.logical_or(x_too_small, y_too_small))
    # Combine good position data lines
    posdata = np.delete(posdata, NotNaN[idxBad], axis=0)

    return posdata

def process_tracking_data(filename, save_to_file=False):
    # Get TrackingSettings
    TrackingSettings = NWBio.load_settings(filename,'/TrackingSettings/')
    # Load position data for all cameras
    posdatas = []
    for n_rpi in TrackingSettings['use_RPi_nrs']:
        posdatas.append(NWBio.load_tracking_data(filename, subset=str(n_rpi)))
    if len(posdatas) > 1:
        # If data from multiple cameras available, combine it
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
            for nRPi in range(len(posdatas)):
                idx_tp[nRPi] = np.argmin(np.abs(timepoints[npoint] - posdatas[nRPi][:,0]))
            # Convert posdatas for use in combineCamerasData function
            cameraPos = []
            for nRPi in range(len(posdatas)):
                cameraPos.append(posdatas[nRPi][idx_tp[nRPi], 1:5])
            tmp_comb_data = combineCamerasData(cameraPos, combPosData[-1], TrackingSettings)
            combPosData.append(tmp_comb_data)
            if tmp_comb_data is None:
                listNaNs.append(npoint)
        # Remove the extra element from combPosData
        del combPosData[0]
        # Remove all None elements
        for nanElement in listNaNs[::-1]:
            del combPosData[nanElement]
        timepoints = np.delete(timepoints, listNaNs)
        # Combine timepoints and position data
        posdata = np.concatenate((np.expand_dims(np.array(timepoints), axis=1), np.array(combPosData)), axis=1)
        # Print info about None elements
        if len(listNaNs) > 0:
            print('Total of ' + str(len(listNaNs) * (1.0 / PosDataFramesPerSecond)) + ' seconds of position data was lost')
            if listNaNs[0] == 0:
                print('This was in the beginning and ' + str(np.sum(np.diff(np.array(listNaNs)) > 1)) + ' other epochs.')
            else:
                print('This was not in the beginning, but in ' + str(np.sum(np.diff(np.array(listNaNs)) > 1) + 1) + ' other epochs.')
    else:
        # In case of single camera being used, just use data from that camera
        posdata = posdatas[0]
    posdata = remove_tracking_data_outside_boundaries(posdata, TrackingSettings['arena_size'], max_error=1)
    posdata = posdata.astype(np.float64)
    if save_to_file:
        # Save corrected data to file
        NWBio.save_tracking_data(filename, posdata, ProcessedPos=True, ReProcess=False)

    return posdata
