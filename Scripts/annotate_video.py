import os
import cv2
import argparse
import numpy as np

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import NWBio
from TrackingDataProcessing import estimate_OpenEphys_timestamps_for_tracking_data

def get_video_resolution(filename):
    '''
    Return frame resolution of a video file.
    '''
    reader = cv2.VideoCapture(filename)
    ret = False
    while not ret:
        ret, frame = reader.read()
    resolution = (frame.shape[1], frame.shape[0])
    reader.release()

    return resolution

def get_video_timestamps(filename, cameraID):
    posdata = NWBio.load_raw_tracking_data(filename, cameraID)
    # Compute position data timestamps in OpenEphys time
    RPi_frame_times = posdata['VideoData_timestamps']
    RPi_GC_times = posdata['GlobalClock_timestamps']
    OE_GC_times = NWBio.load_GlobalClock_timestamps(filename)
    RPi_frame_in_OE_times = estimate_OpenEphys_timestamps_for_tracking_data(OE_GC_times, RPi_GC_times, RPi_frame_times)

    return RPi_frame_in_OE_times

def get_pos_data(filename, cameraID):
    posdata = NWBio.load_raw_tracking_data(filename, cameraID)
    idx_None = np.all(np.isnan(posdata['OnlineTrackerData']), 1)
    posdata['OnlineTrackerData'] = np.delete(posdata['OnlineTrackerData'], 
                                             np.where(idx_None)[0], axis=0)
    posdata['OnlineTrackerData_timestamps'] = np.delete(posdata['OnlineTrackerData_timestamps'], 
                                                        np.where(idx_None)[0], axis=0)
    # Compute position data timestamps in OpenEphys time
    RPi_frame_times = posdata['OnlineTrackerData_timestamps']
    RPi_GC_times = posdata['GlobalClock_timestamps']
    if not ('OE_GC_times' in locals()):
        OE_GC_times = NWBio.load_GlobalClock_timestamps(filename)
    RPi_frame_in_OE_times = estimate_OpenEphys_timestamps_for_tracking_data(OE_GC_times, RPi_GC_times, RPi_frame_times)
    # Combine timestamps and position data into a single array
    posdata = np.concatenate((RPi_frame_in_OE_times.astype(np.float64)[:, None], posdata['OnlineTrackerData']), axis=1)

    return posdata

class video_read_write(object):
    '''
    Makes video file frames sequentially available via read() method and
    can write frames sequentially into a new video MP4 file via write() method.
    '''
    def __init__(self, filename, framerate, new_file_suffix='_annotated', frame_limit=None):
        '''
        filename - full path to video file to read
        framerate - framerate of output video file in Hz
        new_file_suffix (optional) - prefix appended to new video file
        '''
        if not isinstance(new_file_suffix, str) or len(new_file_suffix) == 0:
            raise ValueError('new_file_suffix must be a string of length > 0.')
        self.frame_limit = frame_limit
        self.__frame_pos = -1
        resolution = get_video_resolution(filename)
        self.reader = cv2.VideoCapture(filename)
        out_filename = os.path.splitext(os.path.basename(filename))[0] + new_file_suffix + '.mp4'
        out_file_path = os.path.join(os.path.dirname(filename), out_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(out_file_path, fourcc, framerate, resolution)

    @property
    def frame_pos(self):
        '''
        The position of last frame returned from read() method. Count starts from 0.
        '''
        return self.__frame_pos

    def read(self):
        '''
        Returns the next frame. If no more frames available, returns None.
        '''
        ret, frame = self.reader.read()
        if ret:
            self.__frame_pos += 1
            if self.frame_limit is None or self.__frame_pos < self.frame_limit:
                return frame

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.reader.release()
        self.writer.release()


class video_annotator(object):
    '''
    Rewrites a video file, annotating each frame iteratively with annotator classes. 
    '''
    def __init__(self, annotator_classes, filename, framerate, new_file_suffix='_annotated', frame_limit=None):
        '''
        annotator_classes - iterable of classes to use for annotating each frame.
                            See annotate() method for futher description annotator_classes.
                            Note! close method is called on each class if available.
        filename - full path to video file to read
        framerate - framerate of output video file in Hz
        new_file_suffix (optional) - prefix appended to new video file
        '''
        self.video_files = video_read_write(filename, framerate, new_file_suffix, frame_limit=frame_limit)
        self.annotator_classes = annotator_classes
        self.annotate()
        self.close()

    def annotate(self):
        '''
        Iterates over all frames in the video file and each frame is modified
        by iterating over annotator_classes and calling the process() method of
        each class on the frame. Inputs to process() method are (frame ,frame counter).
        Frame counter starts from 0 on first frame.
        '''
        frame = self.video_files.read()
        while not (frame is None):
            for annotator in self.annotator_classes:
                frame = annotator.process(frame, self.video_files.frame_pos)
            self.video_files.write(frame)
            frame = self.video_files.read()

    def close(self):
        self.video_files.close()
        for annotator in self.annotator_classes:
            if hasattr(annotator, 'close'):
                annotator.close()


class annotator_position(object):
    '''
    Draws circles to locations where position was detected in tracking data
    '''
    def __init__(self, filename, cameraID, lag_signal_threshold=0.5):
        '''
        filename - str - full path to NWB file.
        cameraID - str - ID of camera data to use for annotating images.
        lag_signal_threshold - float - when closest matching position data timepoint
                               is off by more than lag_signal_threshold seconds,
                               the marker is made smaller and less bright.
        '''
        self.lag_signal_threshold = lag_signal_threshold
        resolution_option = NWBio.load_settings(filename, '/CameraSettings/General/resolution_option/')
        calibrationTmatrix = NWBio.load_settings(filename, '/CameraSettings/CameraSpecific/' + \
                                                           cameraID + '/CalibrationData/' + resolution_option + \
                                                           '/calibrationTmatrix')
        self.frame_timestamps = get_video_timestamps(filename, cameraID)
        self.calibrationTmatrix = np.linalg.inv(calibrationTmatrix)
        posdata = get_pos_data(filename, cameraID)
        self.pos_timestamps = posdata[:,0]
        self.pos_xys = (posdata[:,1:3], posdata[:,3:5])
        self.colors = ((255, 0, 0),  (0, 255, 0))
        self.widths = (4, 2)
        self.radius = 10

    def get_color_and_width(self, n_xy, distance):
        color = self.colors[n_xy]
        width = self.widths[n_xy]
        if distance > self.lag_signal_threshold:
            color = tuple(map(int, np.uint8(np.array(color) / 2.0)))
            width = int(width / 2.0)

        return color, width

    def process(self, frame, frame_pos):
        timestamp = self.frame_timestamps[frame_pos]
        idx = (np.abs(self.pos_timestamps - timestamp)).argmin()
        distance = np.abs(timestamp - self.pos_timestamps[idx])
        for n_xy in range(len(self.pos_xys)):
            xy = self.pos_xys[n_xy][idx, :]
            if not np.isnan(xy[0]):
                tmp = np.reshape(np.array([xy[0], xy[1]],dtype=np.float32),(1,1,2))
                tmp = cv2.perspectiveTransform(tmp, self.calibrationTmatrix)
                xy = tuple(list(np.int64(np.round(tmp.squeeze()))))
                color, width = self.get_color_and_width(n_xy, distance)
                frame = cv2.circle(frame, xy, self.radius, color, width)

        return frame


def annotate_with_position(folder_path, cameraID, framerate, frame_limit=None):
    filename = NWBio.get_filename(folder_path)
    pos_annotator = annotator_position(filename, cameraID)
    video_file_name = NWBio.load_raw_tracking_data(filename, cameraID, 
                                                   specific_path='VideoFile')
    video_file_path = os.path.join(folder_path, video_file_name)
    video_annotator((pos_annotator,), video_file_path, framerate, frame_limit=frame_limit)


def main(args):
    folder_path = args.folder_path[0]
    cameraID = args.cameraID[0]
    if args.framerate:
        framerate = args.framerate[0]
    else:
        framerate = 30
    if args.frame_limit:
        frame_limit = args.frame_limit[0]
    else:
        frame_limit = None
    annotate_with_position(folder_path, cameraID, framerate, frame_limit=frame_limit)

if __name__ == '__main__':
    # Input argument handling and help info
    parser = argparse.ArgumentParser(description='Convert all jpg images in folder to mp4 video.')
    parser.add_argument('folder_path', type=str, nargs=1, 
                        help='Path to recording folder.')
    parser.add_argument('cameraID', type=str, nargs=1, 
                        help='Camera ID.')
    parser.add_argument('--framerate', type=int, nargs = 1, 
                        help='Frame rate of the output video (Hz).')
    parser.add_argument('--frame_limit', type=int, nargs = 1, 
                        help='Maxium number of frames to process.')
    args = parser.parse_args()
    main(args)
