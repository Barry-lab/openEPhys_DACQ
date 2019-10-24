
import os
import cv2

from openEPhys_DACQ import NWBio


class RecordingCameraVideo(object):
    """
    Presents video recording from camera together with Open Ephys timestamps.
    """

    def __init__(self, fpath, camera_id):
        """
        :param str fpath: path to recording file or folder containing a single recording file
        :param str camera_id: camera ID value
        """

        self._fpath = NWBio.get_filename(fpath)
        self._camera_id = camera_id

        if not NWBio.check_if_open_ephys_nwb_file(self._fpath):
            raise ValueError('Path {} does not lead to a recognised recording file.'.format(self._fpath))

        camera_data = NWBio.load_raw_tracking_data(self._fpath, self._camera_id)

        self._video_fpath = os.path.join(os.path.dirname(self._fpath), camera_data['VideoFile'])

        if not os.path.isfile(self._video_fpath):
            raise ValueError('Path for video {} does not lead to a file.'.format(self._video_fpath))

        self._timestamps = NWBio.estimate_open_ephys_timestamps_from_other_timestamps(
            NWBio.load_GlobalClock_timestamps(self._fpath),
            camera_data['GlobalClock_timestamps'],
            camera_data['VideoData_timestamps'],
            other_times_divider=10 ** 6
        )

        self._current_index = -1

        self._current_frame = None

        self._capture = cv2.VideoCapture(self._video_fpath)

        self.next()

    @property
    def current_index(self):
        return self._current_index

    @property
    def current_timestamp(self):
        return self._timestamps[self.current_index]

    @property
    def next_timestamp(self):
        if self.current_index + 1 < len(self._timestamps):
            return self._timestamps[self.current_index + 1]
        else:
            return None

    @property
    def previous_timestamp(self):
        if self.current_index + 1 < len(self._timestamps) and self.current_index > 0:
            return self._timestamps[self.current_index - 1]
        else:
            return None

    @property
    def final_timestamp(self):
        return self._timestamps[-1]

    @property
    def current_frame(self):
        return self._current_frame

    @property
    def current_frame_and_timestamp(self):
        return self.current_frame, self.current_timestamp

    def next(self):
        """Moves to next frame. Returns True if next_frame_available, False otherwise.

        :return: next_frame_available
        :rtype: bool
        """
        ret, self._current_frame = self._capture.read()
        if ret:
            self._current_index += 1
        return ret

    def close(self):
        self._capture.release()
