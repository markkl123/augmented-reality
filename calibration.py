# ======= imports
import cv2
import numpy as np
import json
import math

from perspective_warping import plot_image


# ======= contants
CAMERA_MATRIX_KEY = 'K'
DISTORTION_COEFFICIENTS_KEY = 'dist_coeffs'
CALIB_PARAMETERS_PATH = 'calibration_parameters.json'
SQUARE_SIZE = 2.2


# ======= helper functions
def extract_chessboard_frames():
    CHESSBOARD_VIDEO_PATH = r'Videos\chessboard.mp4'
    CHESSBOARD_NUM_FRAMES = 20

    print(f'Extracting {CHESSBOARD_NUM_FRAMES} chessboard frames')

    capture = cv2.VideoCapture(CHESSBOARD_VIDEO_PATH)

    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.array([capture.read()[1] for _ in range(num_frames)])
    capture.release()

    return frames[np.arange(num_frames, step=math.ceil(num_frames / CHESSBOARD_NUM_FRAMES))]


def find_corner_points(chessboard_frames):
    PATTERN_SIZE = (7, 7)  # According to number of internal corners - our board is 8x8 square-wise

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


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), 255, 1)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 1)

    return img


def draw_cube(frame, r_vec, t_vec, K, dist_coeffs):
    object_points = (
            3
            * 2.2
            * np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
    )

    undistorted = cv2.undistort(frame, K, dist_coeffs)
    img_pts = cv2.projectPoints(object_points, r_vec, t_vec, K, dist_coeffs)[0]

    return draw(undistorted, img_pts)


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

    # ======= check calibration
    plot_image(draw_cube(chessboard_frames[0], _rvecs[0], _tvecs[0], camera_matrix, dist_coefs))


if __name__ == '__main__':
    find_calibration_parameters()




