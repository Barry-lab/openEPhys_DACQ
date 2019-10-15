### This script combines position data from multiple cameras.
### It also corrects frame time offset errors in PosLog.csv files
### It also removes bad position data lines

### Use as follows:
### import CombineTrackingData as combPos
### combPos.combdata(Path-To-Recording-Folder)

### By Sander Tanni, May 2017, UCL

import os
from multiprocessing import Process
from itertools import combinations
import argparse

import numpy as np
from scipy.spatial.distance import euclidean
import h5py
from tqdm import tqdm


def combineCamerasData(cameraPos, lastCombPos, cameraIDs, CameraSettings, arena_size):
    # This outputs position data based on which camera is closest to tracking target.

    # cameraPos - list of numpy vecors with 4 elements (x1,y1,x2,y2) for each camera
    # lastCombPos - Last known output from this function. If None, the function will attempt to locate the animal.
    # cameraIDs - list of of CameraSettings.keys() in corresponding order to cameraPos and lastCombPos
    # CameraSettings - settings dictionary created by CameraSettings.CameraSettingsApp
    # arena_size - 2 element numpy array with arena height and width.

    # Output - numpy vector (x1,y1,x2,y2) with data from closest camera

    # Animal detection finding method in case lastCombPos=None
    #   simultaneously from at least 2 cameras with smaller separation than half of CameraSettings['camera_transfer_radius']
    #   If successful, closest mean coordinate is set as output
    #   If unsuccessful, output is None

    N_RPis = len(cameraPos)
    cameraPos = np.array(cameraPos, dtype=np.float32)
    camera_locations = []
    for cameraID in cameraIDs:
        camera_locations.append(CameraSettings['CameraSpecific'][cameraID]['location_xy'])
    camera_locations = np.array(camera_locations, dtype=np.float32)

    # Only work with camera data from inside the enviornment
    # Find bad pos data lines
    idxBad = np.zeros(cameraPos.shape[0], dtype=bool)
    # Points beyond arena size
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
        camera_locations = camera_locations[np.logical_not(idxBad),:]
        if np.any(lastCombPos):
            # Check which cameras provide data close enough to lastCombPos
            RPi_correct = []
            for nRPi in range(N_RPis):
                lastCombPos_distance = euclidean(cameraPos[nRPi, :2], lastCombPos[:2])
                RPi_correct.append(lastCombPos_distance < CameraSettings['General']['camera_transfer_radius'])
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
                    camera_locations = camera_locations[RPi_correct, :]
                    meanPos = np.mean(cameraPos[:, :2], axis=0)
                    # Find mean position distance from all cameras
                    cam_distances = []
                    for nRPi in range(N_RPis):
                        camera_loc = camera_locations[nRPi, :]
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
            cameraPairs_Match = np.array(pairDistances) < (CameraSettings['General']['camera_transfer_radius'] / 2.0)
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

def estimate_OpenEphys_timestamps_for_tracking_data(OE_GC_times, RPi_GC_times, RPi_frame_times, verbose=False):
    '''
    Estimates Open Ephys timestamps for each tracking data datapoint.
    OE_GC_times - Global Clock timestamps in Open Ephys time
    RPi_GC_times - Global Clock timestamps in RPi time
    RPi_frame_times - Frame timestamps in RPi time
    '''
    GlobalClock_TTL_Channel = 1
    RPiTime2Sec = 10 ** 6 # This values is used to convert RPi times to seconds
    # Crop data if more timestamps recorded on either system.
    if OE_GC_times.size > RPi_GC_times.size:
        OE_GC_times = OE_GC_times[:RPi_GC_times.size]
        print('[ Warning ] OpenEphys recorded more GlobalClock TTL pulses than RPi.\n' + 
              'Dumping extra timestamps from the end.')
    elif OE_GC_times.size < RPi_GC_times.size:
        RPi_GC_times = RPi_GC_times[:OE_GC_times.size]
        print('[ Warning ] RPi recorded more GlobalClock TTL pulses than Open Ephys.\n' + 
              'Dumping extra timestamps from the end.')
    # Find closest RPi GlobalClock timestamp to each RPi frame timestamp
    if verbose:
        print('Estimating camera timestamps in OpenEPhys time')
    RPtimes_idx = []
    for RPFtime in (tqdm(RPi_frame_times) if verbose else RPi_frame_times):
        RPtimes_idx.append(np.argmin(np.abs(RPi_GC_times - RPFtime)))
    # Compute difference from the GlobalClock timestamps for each frame
    RPtimes_full = RPi_GC_times[RPtimes_idx]
    RPF_RP_times_delta = RPi_frame_times - RPtimes_full
    # Convert this difference to seconds as Open Ephys timestamps
    RPF_RP_times_delta_in_seconds = RPF_RP_times_delta / float(RPiTime2Sec)
    # Use frame and GlobalClock timestamp diffence to estimate OpenEphys timepoints
    OEtimes_full = OE_GC_times[RPtimes_idx]
    RPi_frame_in_OE_times = OEtimes_full + RPF_RP_times_delta_in_seconds

    return RPi_frame_in_OE_times

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


def remove_tracking_data_jumps(posdata, maxjump):
    """
    Removes data with too large jumps based on euclidean distance

    posdata - numpy array with columns:
              timestamps
              LED 1 xpos
              LED 1 ypos
              LED 2 xpos
              LED 2 ypos
              , where NaN for missing LED 2 data
    maxjump - int or float specifying maximum allowed shift in euclidean distance
    """
    keepPos = []
    lastPos = posdata[0,1:3]
    for npos in range(posdata.shape[0]):
        currpos = posdata[npos,1:3]
        if euclidean(lastPos, currpos) < maxjump:
            keepPos.append(npos)
            lastPos = currpos
    keepPos = np.array(keepPos)
    print(str(posdata.shape[0] - keepPos.size) + ' of ' + 
          str(posdata.shape[0]) + ' removed in postprocessing')
    posdata = posdata[keepPos,:]

    return posdata


def iteratively_combine_multicamera_data_for_recording(
    CameraSettings, arena_size, posdatas, OE_GC_times, verbose=False):
    """Return ProcessedPos

    Combines raw tracking data from multiple cameras into a single ProcessedPos array.
    If a single camera data is provided, this will be converted into same format.
    """
    cameraIDs = sorted(CameraSettings['CameraSpecific'].keys())
    # Load position data for all cameras
    for i, posdata in enumerate(posdatas):
        if isinstance(posdata, dict):
            # This is conditional because for early recordings tracking data was immediately stored in Open Ephys time.
            # Remove datapoints where all position data is None
            idx_None = np.all(np.isnan(posdata['OnlineTrackerData']), 1)
            posdata['OnlineTrackerData'] = np.delete(posdata['OnlineTrackerData'], 
                                                     np.where(idx_None)[0], axis=0)
            posdata['OnlineTrackerData_timestamps'] = np.delete(posdata['OnlineTrackerData_timestamps'], 
                                                                np.where(idx_None)[0], axis=0)
            # Compute position data timestamps in OpenEphys time
            RPi_frame_times = posdata['OnlineTrackerData_timestamps']
            RPi_GC_times = posdata['GlobalClock_timestamps']
            RPi_frame_in_OE_times = estimate_OpenEphys_timestamps_for_tracking_data(
                OE_GC_times, RPi_GC_times, RPi_frame_times, verbose=verbose
            )
            # Combine timestamps and position data into a single array
            posdata = np.concatenate((RPi_frame_in_OE_times.astype(np.float64)[:, None], posdata['OnlineTrackerData']), axis=1)
            posdatas[i] = posdata
    if len(posdatas) > 1:
        # If data from multiple cameras available, combine it
        PosDataFramesPerSecond = 30.0
        # Find first and last timepoint for position data
        first_timepoints = []
        last_timepoints = []
        for posdata in posdatas:
            first_timepoints.append(posdata[0, 0])
            last_timepoints.append(posdata[-1, 0])
        # Combine position data step-wise from first to last timepoint at PosDataFramesPerSecond
        # At each timepoint the closest matchin datapoints will be taken from different cameras
        timepoints = np.arange(np.array(first_timepoints).min(), np.array(last_timepoints).max(), 1.0 / PosDataFramesPerSecond)
        combPosData = [None]
        listNaNs = []
        if verbose:
            print('Combining iteratively camera data for each position timepoint')
        for npoint in (tqdm(range(len(timepoints))) if verbose else range(len(timepoints))):
            # Find closest matchin timepoint from all RPis
            idx_tp = np.zeros(4, dtype=np.int32)
            for nRPi in range(len(posdatas)):
                idx_tp[nRPi] = np.argmin(np.abs(timepoints[npoint] - posdatas[nRPi][:,0]))
            # Convert posdatas for use in combineCamerasData function
            cameraPos = []
            for nRPi in range(len(posdatas)):
                cameraPos.append(posdatas[nRPi][idx_tp[nRPi], 1:5])
            tmp_comb_data = combineCamerasData(cameraPos, combPosData[-1], cameraIDs, CameraSettings, arena_size)
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
        ProcessedPos = np.concatenate((np.expand_dims(np.array(timepoints), axis=1), np.array(combPosData)), axis=1)
        # Print info about None elements
        if len(listNaNs) > 0:
            print('Total of ' + str(len(listNaNs) * (1.0 / PosDataFramesPerSecond)) + ' seconds of position data was lost')
            if listNaNs[0] == 0:
                print('This was in the beginning and ' + str(np.sum(np.diff(np.array(listNaNs)) > 1)) + ' other epochs.')
            else:
                print('This was not in the beginning, but in ' + str(np.sum(np.diff(np.array(listNaNs)) > 1) + 1) + ' other epochs.')
    else:
        # In case of single camera being used, just use data from that camera
        ProcessedPos = posdatas[0]
    ProcessedPos = remove_tracking_data_outside_boundaries(ProcessedPos, arena_size, max_error=10)
    ProcessedPos = ProcessedPos.astype(np.float64)

    return ProcessedPos
