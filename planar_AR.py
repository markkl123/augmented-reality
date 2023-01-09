# ======= imports
import numpy as np

from perspective_warping import *
from calibration import *
import os
from mesh_renderer import MeshRenderer


# ======= helper functions
def extract_xyz_from_keypoints(keypoints, width, height):
    TEMPLATE_WIDTH_CM = 21.2
    TEMPLATE_HEIGHT_CM = 14.1

    return np.array([(
        kp[0] * TEMPLATE_WIDTH_CM / width,
        kp[1] * TEMPLATE_HEIGHT_CM / height,
        0)
        for kp in keypoints])


# ======= contants
RENDERED_VIDEO_PATH = r'Videos\rendered.mp4'
OBJECT_PATH = r'Models\drill\drill.obj'


if __name__ == '__main__':

    # ======= loading calibration parameters
    if not os.path.exists(CALIB_PARAMETERS_PATH):
        find_calibration_parameters()

    with open(CALIB_PARAMETERS_PATH, 'r') as f:
        parameters = json.loads(json.load(f))
        K = np.array(parameters[CAMERA_MATRIX_KEY])
        dist_coeffs = np.array(parameters[DISTORTION_COEFFICIENTS_KEY])

    # ======= template image keypoint and descriptors
    template = cv2.imread(TEMPLATE_IMAGE_PATH)
    template_height, template_width = template.shape[:2]
    template_kp, template_desc = extract_keypoints_and_descriptors(template)

    # ======= video input, output and metadata
    capture = cv2.VideoCapture(ORIGINAL_VIDEO_PATH)

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    color = True
    plot_delay = 1000 // fps

    # we rotate the dimensions
    writer = cv2.VideoWriter(RENDERED_VIDEO_PATH, codec, fps, (frame_height, frame_width), color)

    renderer = MeshRenderer(K, frame_width, frame_height, OBJECT_PATH)

    print(f'Processing {num_frames} frames of size {frame_width}x{frame_height}, {num_frames / fps:.2f} seconds, {fps} f/s')

    # ======= run on all frames
    for i in range(num_frames):

        frame = capture.read()[1]
        frame = np.flip(np.transpose(frame, (1, 0, 2)), axis=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ======= find keypoints matches of frame and template
        # we saw this in the SIFT notebook
        frame_kp, frame_desc = extract_keypoints_and_descriptors(frame)
        good_matches = find_matches(frame_desc, template_desc)

        # ======= find homography
        # also in SIFT notebook
        (H, masked), good_template_kp, good_frame_kp = find_homography(template_kp, frame_kp, good_matches)

        # +++++++ take subset of keypoints that obey homography (both frame and reference)
        # this is at most 3 lines- 2 of which are really the same
        # HINT: the function from above should give you this almost completely
        transformed_points = cv2.perspectiveTransform(np.array([good_template_kp]), H)[0]
        best_keypoints = [(good_frame_kp[i], good_template_kp[i])
                          for i, (f_coord, t_coord) in enumerate(zip(good_frame_kp, transformed_points))
                          if np.linalg.norm(f_coord - t_coord) < 10]
        frame_points, template_points = zip(*best_keypoints)
        frame_points, template_points = np.array(frame_points), np.array(template_points)        

        # +++++++ solve PnP to get cam pose (r_vec and t_vec)
        # `cv2.solvePnP` is a function that receives:
        # - xyz of the template in centimeter in camera world (x,3)
        # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
        # - camera K
        # - camera dist_coeffs
        # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
        #
        # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
        # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
        # For this we just need the template width and height in cm.
        #
        # this part is 2 rows
        ret, r_vec, t_vec = cv2.solvePnP(extract_xyz_from_keypoints(template_points, template_width, template_height),
                                         frame_points,
                                         K,
                                         dist_coeffs)

        # +++++++ draw cube with r_vec and t_vec on top of rgb frame
        # We saw how to draw cubes in camera calibration. (copy and paste)
        rendered_rgb = draw_cube(frame_rgb, r_vec, t_vec, K, dist_coeffs)

        # +++++++ draw object with r_vec and t_vec on top of rgb frame
        # After drawing the cube works you can replace this with the draw function from the renderer class renderer.draw() (1 line)
        #rendered_rgb = renderer.draw(frame_rgb, r_vec, t_vec)

        # ======= plot and save frame
        rendered = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
        writer.write(rendered)

        if i % 100 == 0:
            print(f'{i}/{num_frames}')

    # ======= end all
    capture.release()
    writer.release()
