
import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations
import cv2

from openEPhys_DACQ import NWBio
from openEPhys_DACQ.video_io import RecordingCameraVideo


def transform_pix_to_real_value(y_ind, x_ind, calibration_matrix):
    """ Transforms position on image from pixels to centimeters in real values using a transformation matrix.

    :param float y_ind: y-axis (first/vertical dimension) position on image
    :param float x_ind: x-axis (second/horizontal dimension) position on image
    :param calibration_matrix:
    :return: x_val, y_val - corresponding real values to x_ind and y_ind
    """
    led_pix_4pt = np.reshape(np.array([y_ind, x_ind]), (1, 1, 2)).astype(np.float32)
    pos_cm_from4pt = cv2.perspectiveTransform(led_pix_4pt, calibration_matrix).astype('float')
    x_val, y_val = pos_cm_from4pt.squeeze()

    return x_val, y_val


def image_peak_ind(image):
    """Returns peak y (horizontal) and x (vertical) indices for grayscale image

    :param numpy.ndarray image: grayscale image of shape (M, N)
    :return: y_ind, x_ind
    """
    (_1, _2, _3, (x_ind, y_ind)) = cv2.minMaxLoc(image)
    return y_ind, x_ind


def crop_image_around_point(image, y_ind, x_ind, min_radius):
    """Returns a rectangular crop of the image respecting image boundaries, such that it can fit a circle of min_radius.

    If y_ind and x_ind are too close to the edge, the cropped image will be smaller in the direction of the edge(s).

    :param numpy.ndarray image: grayscale image shape (M, N)
    :param int y_ind: center of cropped image relative to input image along y (vertical/first) dimension
    :param int x_ind: center of cropped image relative to input image along x (horizontal/second) dimension
    :param int min_radius: cropped image size is such that it would perfectly fit a circle with min_radius radius.
    :return: cropped_image, y_min, x_min
    """

    x_min = x_ind - min_radius
    x_max = x_ind + min_radius
    y_min = y_ind - min_radius
    y_max = y_ind + min_radius

    x_min = x_min if x_min > 0 else 0
    y_min = y_min if y_min > 0 else 0
    x_max = x_max if x_max <= image.shape[1] else image.shape[1]
    y_max = y_max if y_max <= image.shape[0] else image.shape[0]

    return image[x_min:y_max, x_min:x_max], y_min, x_min


def find_bright_circular_blobs(image, threshold):
    """Returns list of blobs in order of descending size and keypoints element for drawing blobs with opencv.

    :param numpy.ndarray image: grayscale image shape (M, N)
    :param float threshold: threshold value for blob detection
    :return: blobs, keypoints
    """
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = threshold
    params.thresholdStep = 0.01 * threshold
    params.filterByArea = True
    params.minArea = 2
    params.filterByCircularity = True
    params.minCircularity = 0.95
    params.filterByConvexity = True
    params.minConvexity = 0.95
    params.minDistBetweenBlobs = 2
    params.blobColor = 255

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)

    blobs = [{'size': keypoint.size, 'x_loc': keypoint.pt[0], 'y_loc': keypoint.pt[1]} for keypoint in keypoints]

    _, blobs, keypoints = sorted(zip([blob['size'] for blob in blobs], blobs, keypoints), reverse=True)

    return blobs, keypoints


def shift_cropped_image_blobs(blobs, keypoints, y_min, x_min):
    """Shift positional values of output from :py:func:`find_bright_circular_blobs` by x_min and y_min.
    """
    for blob in blobs:
        blob['x_loc'] = blob['x_loc'] + x_min
        blob['y_loc'] = blob['y_loc'] + y_min
    for keypoint in keypoints:
        keypoint.pt = (keypoint.pt[0] + x_min, keypoint.pt[1] + y_min)


def distance_between_adjacent_pixels(calibration_matrix, frame_shape):
    """Approximation of distance in centimeters correspond to single pixel difference at the center of field of view.

    :param numpy.ndarray calibration_matrix: matrix that can be used with :py:func:`cv2.perspectiveTransform`
    :param tuple frame_shape: shape of the frames used
    :return: distance
    """
    distance_in_pixels = 10
    # Pick two points at the center of the field of view
    tmp_loc1 = np.reshape(np.array([int(frame_shape[1] / 2), int(frame_shape[0] / 2)], dtype=np.float32), (1, 1, 2))
    tmp_loc2 = \
        np.reshape(np.array([int(frame_shape[1] / 2), int(frame_shape[0] / 2) + distance_in_pixels], dtype=np.float32),
                   (1, 1, 2))
    # Use transformation matrix to map pixel values to position in real world in centimeters
    tmp_loc1 = cv2.perspectiveTransform(tmp_loc1, calibration_matrix)
    tmp_loc2 = cv2.perspectiveTransform(tmp_loc2, calibration_matrix)
    # Compute the distance between the two points in real world centimeters
    real_distance = euclidean(np.array([tmp_loc1[0, 0, 0].astype('float'), tmp_loc1[0, 0, 1].astype('float')]),
                              np.array([tmp_loc2[0, 0, 0].astype('float'), tmp_loc2[0, 0, 1].astype('float')]))
    # Compute distance in centimeters between adjacent pixels
    distance = float(real_distance) / float(distance_in_pixels)

    return distance


def convert_centimeters_to_pixel_distance(real_distance, calibration_matrix, frame_shape):
    """Converts a distance in real values to distance in pixels at the center of frame.

    :param float real_distance: distance in real values
    :param numpy.ndarray calibration_matrix: matrix that can be used with :py:func:`cv2.perspectiveTransform`
    :param tuple frame_shape: shape of the frames used
    :return: pixel_distance
    """
    return int(np.round(float(real_distance) / distance_between_adjacent_pixels(calibration_matrix, frame_shape)))


class LedDetector(object):
    """
    Detects blobs around brightest position in images.
    """

    def __init__(self, image_shape, calibration_matrix, detection_window, search_radius,
                 threshold_multiplier=0.75, smoothing_size=5):
        """
        :param tuple image_shape: (N, M) specifying shape of images used
        :param numpy.ndarray calibration_matrix: matrix that can be used with :py:func:`cv2.perspectiveTransform`
        :param detection_window: (x_min, x_max, y_min, y_max) range of real world values where to look for led.
        :param float search_radius: size of search area (in real values) around brightest spot for blobs.
        :param float threshold_multiplier: used to multiply value at brightest spot for blob detection threshold.
        :param float smoothing_size: sigma of spatial smoothing gaussian kernel.
        """
        self._image_shape = image_shape
        self._calibration_matrix = calibration_matrix
        self._search_radius = convert_centimeters_to_pixel_distance(search_radius, calibration_matrix, image_shape)
        self._threshold_multiplier = threshold_multiplier
        self._smoothing_size = smoothing_size

        # Compute position of mapping of each pixel to real value based on calibration_matrix
        self._real_value_map = np.array((image_shape[0], image_shape[1], 2), dtype=np.float32)
        for y_ind in range(image_shape[0]):
            for x_ind in range(image_shape[1]):
                self._real_value_map[y_ind, x_ind, :] = transform_pix_to_real_value(y_ind, x_ind, calibration_matrix)

        # Create mask for masking out image pixels mapping to positions outside detection_window
        self._mask = np.zeros(image_shape, dtype=np.bool)
        for y_ind in range(image_shape[0]):
            for x_ind in range(image_shape[1]):
                if (detection_window[0] <= x_ind <= detection_window[1]
                        and detection_window[2] <= y_ind <= detection_window[3]):
                    self._mask[y_ind, x_ind] = False

    def process(self, image):
        """Returns data on bright blobs found around brightest point in sorted order of largest blobs first.

        blobs - list of dicts with elements:
            'x_loc' - x axis (horizontal/second dimension) image pixel coordinate of blob
            'y_loc' - y axis (vertical/first dimension) image pixel coordinate of blob
            'x_real' - x axis (horizontal/second dimension) real coordinate of blob based on calibration_matrix
            'y_real' - y axis (vertical/first dimension) real coordinate of blob based on calibration_matrix
            'size' - size of the blob

        keypoints - raw output from :py:func:`cv2.SimpleBlobDetector_create.detect` that can be used for
            visualising blobs with:
        >>> cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
        >>>                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        :param numpy.ndarray image: grayscale image with shape matching image_shape given during initialization
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        :return: blobs, keypoints
        """
        if image.shape != self._image_shape:
            raise Exception('Image shape does not match image_shape specified during initiation.')

        image = cv2.GaussianBlur(image, (0, 0), self._smoothing_size)
        image[self._mask] = 0

        y_ind, x_ind = image_peak_ind(image)

        cropped_image, y_min, x_min = crop_image_around_point(image, y_ind, x_ind, self._search_radius)
        blobs, keypoints = find_bright_circular_blobs(cropped_image, image[y_ind, x_ind] * self._threshold_multiplier)
        shift_cropped_image_blobs(blobs, keypoints, y_min, x_min)

        for blob in blobs:
            x, y = transform_pix_to_real_value(blob['y_loc'], blob['x_loc'], self._calibration_matrix)
            blob['x_real'] = x
            blob['y_real'] = y

        return blobs, keypoints


class LimitedMemory(object):

    def __init__(self, max_length):

        self._max_length = max_length
        self._items = []
        self._identifiers = []

    def __contains__(self, identifier):
        return identifier in self._identifiers

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        return self._items[key]

    @property
    def items(self):
        return self._items

    @property
    def identifiers(self):
        return self._identifiers

    def append(self, other, identifier=None):
        if len(self._items) >= self._max_length:
            self._items.pop(0)
            self._identifiers.pop(0)
        self._items.append(other)
        self._identifiers.append(identifier)

    def get(self, identifier):
        if identifier in self._identifiers:
            return self._items[self._identifiers.index(identifier)]
        else:
            raise ValueError('identifier {} not found in {}'.format(identifier, self))


def identify_true_position(led_positions, max_separation):
    """Returns x and y coordinates of true led_position identified and None otherwise.
    
    :param dict led_positions: dictionary with camera_id as keys and elements as dictionaries of led_position,
        as output from :py:func:`LedDetector.process`. This method uses the 'y_real' and 'x_real' values.
    :param float max_separation: maximum distance allowed between led_position from two cameras for confirmation.
    :return: x, y or None
    """

    # Compute distances between positions of all camera pairs
    camera_pairs = []
    camera_pair_distances = []
    for camera_ids in combinations(list(map(str, led_positions.keys())), 2):

        if len(led_positions[camera_ids[0]]) == 0:
            continue

        camera_pair_distances.append(euclidean(
            np.array((led_positions[camera_ids[0]][0]['x_real'], led_positions[camera_ids[0]][0]['y_real'])),
            np.array((led_positions[camera_ids[1]][0]['x_real'], led_positions[camera_ids[1]][0]['y_real']))
        ))
        camera_pairs.append(camera_ids)

    camera_pair_close_enough_idx = np.array(camera_pair_distances) < max_separation

    if not np.any(camera_pair_close_enough_idx):

        return None

    else:

        closest_camera_pair_ind = int(np.argmin(camera_pair_distances))
        camera_ids = camera_pairs[closest_camera_pair_ind]

        x, y = np.mean(np.array([(led_positions[camera_ids[0]][0]['x_real'],
                                  led_positions[camera_ids[0]][0]['y_real']),
                                 (led_positions[camera_ids[1]][0]['x_real'],
                                  led_positions[camera_ids[1]][0]['y_real'])]),
                       axis=0)

        return x, y


class ProcessedLedLocationState(object):

    unknown_position = np.array((np.nan, np.nan))

    def __init__(self):
        self._last_confirmed_position = self.unknown_position
        self._last_confirmed_timestamp = None
        self._last_timestamp_was_confirmed = False

    @property
    def last_confirmed_position(self):
        return self._last_confirmed_position

    @property
    def last_confirmed_timestamp(self):
        return self._last_confirmed_timestamp

    @property
    def last_timestamp_was_confirmed(self):
        return self._last_timestamp_was_confirmed

    def _update_prediction(self, new_position):
        if np.any(np.isnan(new_position)) or np.any(np.isnan(self._last_confirmed_position)):
            self.predicted_position = self.unknown_position
        else:
            last_movement_vector = new_position - self._last_confirmed_position
            self.predicted_position = new_position + last_movement_vector

    def update(self, position, timestamp):
        position = np.array(position)
        self._update_prediction(position)
        if ~np.any(np.isnan(position)):
            self._last_confirmed_position = position
            self._last_confirmed_timestamp = timestamp
            self._last_timestamp_was_confirmed = True
        else:
            self._last_timestamp_was_confirmed = False


class DualLedMultiCameraTracker(object):

    def __init__(self, led_1_xy, led_2_xy, timestamp, camera_positions,
                 flip_reset_speed=40, flip_reset_steps=6, flip_reset_tortuosity=0.8):

        self._led_states = (ProcessedLedLocationState(),
                            ProcessedLedLocationState())
        self._led_states[0].update(led_1_xy, timestamp)
        self._led_states[1].update(led_2_xy, timestamp)

        self._camera_positions = camera_positions

        self._flip_reset_speed = flip_reset_speed
        self._flip_reset_tortuosity = flip_reset_tortuosity

        self._position_history = LimitedMemory(flip_reset_steps)

        self._anterior_led_ind = 0
        self._posterior_led_ind = 1

    @staticmethod
    def position_vector_angle(position_vector):
        return np.angle(complex(*position_vector)) + np.pi

    def _get_movement_direction_for_led_flip(self):

        positions = np.array(self._position_history.items)

        if np.any(np.isnan(positions).flatten()):
            return None

        timestamps = np.array(self._position_history.identifiers)

        position_deltas = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))

        if np.sum(position_deltas) / euclidean(positions[0, :], positions[-1, :]) < self._flip_reset_tortuosity:
            return None

        if np.mean(position_deltas / np.diff(timestamps)) < self._flip_reset_speed:
            return None

        full_movement_vector = positions[-1, :] - positions[0, :]

        return self.position_vector_angle(full_movement_vector)

    def _attempt_led_flip(self):
        """Checks self._led_state values and compares to recent trajectory direction. If they are in
        opposite directions, the led identities are flipped.
        """

        if not all([led_state.last_timestamp_was_confirmed for led_state in self._led_states]):
            return

        movement_direction = self._get_movement_direction_for_led_flip()

        if movement_direction is None:
            return

        led_direction = self.position_vector_angle(self._led_states[self._anterior_led_ind]
                                                   - self._led_states[self._posterior_led_ind])

        if np.mod(movement_direction - led_direction, 2 * np.pi) > np.pi:

            self._anterior_led_ind = (1, 0)[self._anterior_led_ind]
            self._posterior_led_ind = (1, 0)[self._anterior_led_ind]

    def update(self, led_positions):

        # This method should use last known location to decide which camera position to use,
        # by comparing the distances to each of the cameras

        # There could be some inertia to staying with one camera beyond the mid-line between cameras.

        # It should match the input LED locations to the elements of self._led_states based
        # on proximity to the predicted locations.

        # The following function could also be used to further verify LED angle with respect to movement direction.
        self._attempt_led_flip()

        # It should then assign the led locations to correct elements in self._led_states based on
        # self._anterior_led_ind and self._posterior_led_ind.

        # If a camera switch happens, the self._anterior_led_ind and self._posterior_led_ind should default


class DualLedMultiCameraProcessor(object):
    """
    Tracks LED position in video data.
    """

    def __init__(self, video, calibration_matrix, camera_position, arena_size, led_separation,
                 threshold_multiplier, smoothing_size, camera_transfer_radius=None, init_camera_id=None):

        self._camera_ids = list(map(str, self._video.keys()))

        self._video = video
        self._calibration_matrix = calibration_matrix
        self._camera_position = camera_position

        if camera_transfer_radius is None and init_camera_id is None:
            raise ValueError('Either camera_transfer_radius or init_camera_id must be specified.')

        self._init_camera_id = init_camera_id
        self._camera_transfer_radius = camera_transfer_radius if self._init_camera_id is None else None

        self._led_detector = {}
        for camera_id in self._video:
            self._led_detector[camera_id] = LedDetector(
                self._video[camera_id].current_frame.shape,
                self._calibration_matrix[camera_id],
                (0, arena_size[0], 0, arena_size[1]),
                led_separation * 2,
                threshold_multiplier,
                smoothing_size
            )

        self._led_position_memory = {camera_id: LimitedMemory(10) for camera_id in self._video}

    def _get_led_position_in_current_frame(self, camera_id):
        """Returns led_position in current image of this camera.

        :param str camera_id: camera identifier
        :return: led_position
        """
        # If possible, use pre-existing LED positions
        if self._video[camera_id].current_timestamp in self._led_position_memory[camera_id]:
            return self._led_position_memory[camera_id].get(self._video[camera_id].current_timestamp)
        # Find LED positions and store in memory
        led_position = self._led_detector[camera_id].process(self._video[camera_id].current_frame)
        self._led_position_memory[camera_id].append(led_position, self._video[camera_id].current_timestamp)

        return led_position[:2] if len(led_position) > 2 else led_position

    def _attempt_to_identify_true_position(self):

        if len(self._video) == 1:

            led_position = self._get_led_position_in_current_frame(self._camera_ids[0])[0]

            return np.array(led_position['x_real'], led_position['y_real']), np.array((np.nan, np.nan))

        elif not (self._init_camera_id is None):

            led_position = self._get_led_position_in_current_frame(self._init_camera_id)[0]

            return np.array(led_position['x_real'], led_position['y_real']), np.array((np.nan, np.nan))

        else:

            led_positions = {camera_id: self._get_led_position_in_current_frame(camera_id)
                             for camera_id in self._camera_ids}

            xy = identify_true_position(led_positions, self._camera_transfer_radius / 2.)

            if xy is None:

                return np.array((np.nan, np.nan, np.nan, np.nan))

            else:

                return np.array(xy[0], xy[1]), np.array((np.nan, np.nan))

    def get_current_led_positions(self):
        # This method should be using class DualLedMultiCameraTracker to keep track of LED location.
        # DualLedMultiCameraTracker should notify if LED was not been tracked for a long period,
        # which should prompt this DualLedMultiCameraProcessor class to reinitialize tracking
        # using multi-camera confirmation method.


class OfflineTracker(object):
    """
    Uses video data stored on the disk to provide tracking information for a recording.
    """

    # List of supported processing methods in _initialize_processing_method and _process_for_current_timestamp
    supported_methods = ('dual_led',)

    def __init__(self, fpath, fps=30, method=None, threshold_multiplier=0.75):

        # Parse input

        self._fpath = NWBio.get_filename(fpath)

        if not NWBio.check_if_open_ephys_nwb_file(self._fpath):
            raise ValueError('Path {} does not lead to a recognised recording file.'.format(self._fpath))

        self._timestep = 1 / float(fps)

        settings = NWBio.load_settings(self._fpath)

        self._method = settings['CameraSettings']['General']['tracking_mode'] if method is None else method
        if not (self._method in self.supported_methods):
            raise ValueError('Method {} not found in support methods {}'.format(self._method, self.supported_methods))

        self._threshold_multiplier = threshold_multiplier

        # Get list of cameras_ids

        self._camera_ids = tuple(map(str, settings['CameraSettings']['CameraSpecific'].keys()))
        if len(self._camera_ids) == 0:
            raise Exception('No cameras specified in CameraSettings.')

        # Get settings for processing camera images

        resolution_setting = settings['CameraSettings']['General']['resolution_option']

        self._arena_size = settings['General']['arena_size']
        self._led_separation = settings['CameraSettings']['General']['LED_separation']
        self._camera_transfer_radius = settings['CameraSettings']['General']['camera_transfer_radius']
        self._smoothing_size = {'high': 4, 'low': 2}[resolution_setting]

        self._calibration_matrix = {}
        self._camera_position = {}

        for camera_id in self._camera_ids:
            camera_id_settings = settings['CameraSettings']['CameraSpecific'][camera_id]

            self._calibration_matrix[camera_id] = \
                camera_id_settings['CalibrationData'][resolution_setting]['calibrationTmatrix']
            self._camera_position[camera_id] = camera_id_settings['position_xy']

        # Initialise video feed

        self._video = {camera_id: RecordingCameraVideo(self._fpath, camera_id) for camera_id in self._camera_ids}

        self._align_videos()

        # Initialise tracking process

        self._current_timestamp = min([self._video[camera_id].current_timestamp
                                       for camera_id in self._camera_ids])

        self._final_timestamp = min([self._video[camera_id].final_timestamp
                                     for camera_id in self._camera_ids])

        self._timestamps = []
        self._tracking_data = []

        self._process_started = False
        self._process_finished = False

    def _align_videos(self):
        """Ensures first frames from all cameras are as closely aligned as possible.
        """
        if len(self._camera_ids) == 1:
            return

        frame_received = {camera_id: False for camera_id in self._camera_ids}
        while not all(frame_received.values()):

            earliest_frame_camera_id = self._camera_ids[int(np.argmin([self._video[camera_id].current_timestamp
                                                                       for camera_id in self._camera_ids]))]
            self._video[earliest_frame_camera_id].next()
            frame_received[earliest_frame_camera_id] = True

    def _seek_closest_frames_to_current_timestamp(self):
        """Moves camera to next video frame if it is closer to current timestamp.
        """
        camera_aligned = {camera_id: False for camera_id in self._camera_ids}
        while not all(camera_aligned.values()):

            for camera_id in self._camera_ids:

                if camera_aligned[camera_id]:
                    continue

                if self._video[camera_id].next_timestamp is None:
                    camera_aligned[camera_id] = True
                    continue

                current_frame_delta = abs(self._current_timestamp - self._video[camera_id].current_timestamp)
                next_frame_delta = abs(self._current_timestamp - self._video[camera_id].next_timestamp)

                if next_frame_delta < current_frame_delta:
                    ret = self._video[camera_id].next()
                    assert ret, ('This should be True as loop execution should not make it here \n'
                                 + 'if next frame does not exist. The video data may be corrupted.')
                else:
                    camera_aligned[camera_id] = True

    def _process_current_timestamp_with_dual_led_multi_camera_processor(self):
        pass

    def _initialize_processing_method(self):

        if self._method == 'dual_led':

            if len(self._camera_ids) > 1:

                self._processor = DualLedMultiCameraProcessor(self._video, self._calibration_matrix,
                                                              self._camera_position, self._arena_size,
                                                              self._led_separation, self._threshold_multiplier,
                                                              self._smoothing_size,
                                                              camera_transfer_radius=self._camera_transfer_radius)
                return

        raise Exception('No processor available to match method {} and n = {} camera(s).'.format(
            self._method, len(self._camera_ids)))

    def _process_for_current_timestamp(self):

        if self._method == 'dual_led':

            if len(self._camera_ids) > 1:

                return self._process_current_timestamp_with_dual_led_multi_camera_processor()

    def process(self):

        if self._process_started:
            raise Exception('Method process can be call only once for each instance of OfflineTracker.')
        else:
            self._process_started = True

        self._initialize_processing_method()

        while self._current_timestamp < self._final_timestamp:

            self._timestamps.append(self._current_timestamp)

            self._seek_closest_frames_to_current_timestamp()

            self._tracking_data.append(self._process_for_current_timestamp())

            self._current_timestamp += self._timestep

        self._process_finished = True

    def ensure_process_is_finished(self):
        if not self._process_finished:
            self.process()

    @property
    def timestamps(self):
        self.ensure_process_is_finished()
        return self._timestamps

    @property
    def led_positions(self):
        self.ensure_process_is_finished()
        return self._tracking_data

    def close(self):
        for camera_id in self._camera_ids:
            self._video[camera_id].close()
