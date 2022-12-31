# ======= imports
import cv2
import numpy as np
import json

# ======= helper functions
def extract_chessboard_frames():
    CHESSBOARD_VIDEO_PATH = r'Videos\chessboard.mp4'
    CHESSBOARD_NUM_FRAMES = 20

    print(f'Extracting {CHESSBOARD_NUM_FRAMES} chessboard frames')

    capture = cv2.VideoCapture(CHESSBOARD_VIDEO_PATH)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.array([capture.read()[1] for i in range(num_frames)])
    capture.release()

    return frames[np.arange(num_frames, step=num_frames // CHESSBOARD_NUM_FRAMES)]


def find_corner_points(chessboard_frames):
    PATTERN_SIZE = (7, 7) # According to number of internal corners - our board is 8x8 square-wise
    SQUARE_SIZE = 2.2 # cm

    pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
    pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
    pattern_points *= SQUARE_SIZE

    object_points = []
    frame_points = []

    for i, frame in enumerate(chessboard_frames):
        found, corners = cv2.findChessboardCorners(frame, PATTERN_SIZE)

        if not found:
            print("Chessboard was not found in frame #{i}")

        object_points.append(pattern_points)
        frame_points.append(corners.reshape(-1, 2))

    return object_points, frame_points

# ======= contants
CAMERA_MATRIX_KEY = 'K'
DISTORTION_COEFFICIENTS_KEY = 'dist_coeffs'
CALIB_PARAMETERS_PATH = 'calibration_parameters.json'


def find_calibration_parameters():
    print("Calculating camera intrinsics & distortion coefficients")

    # ======= extract chessboard frames from video
    chessboard_frames = extract_chessboard_frames()  

    # ======= find all corners in calibration plane
    object_points, frame_points = find_corner_points(chessboard_frames)

    # ======= get camera intrinsics + distortion coeffs
    h, w = chessboard_frames[0].shape[:2]
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(object_points, frame_points, (w, h), None, None)

    # ======= save calibration parameters
    with open(CALIB_PARAMETERS_PATH, 'w') as f:
        parameters = {CAMERA_MATRIX_KEY: camera_matrix.tolist(), DISTORTION_COEFFICIENTS_KEY: dist_coefs.ravel().tolist()}
        json.dump(json.dumps(parameters), f)



