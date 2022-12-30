# ======= imports
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ======= helper functions
def print_image(image, title):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title(title)
    plt.show()


def extract_keypoints_and_descriptors(gray_image):
    feature_extractor = cv2.SIFT_create()
    return feature_extractor.detectAndCompute(gray_image, None)


def draw_keypoints(rgb_image, keypoints):
    kepoints_image = cv2.drawKeypoints(rgb_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print_image(kepoints_image, 'keypoints')


def find_matches(frame_kp, template_kp):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(frame_desc, template_desc, k=2)
    good_and_second_good_match_list = [m for m in matches if m[0].distance/m[1].distance < 0.5]
    return np.asarray(good_and_second_good_match_list)[:,0]


def find_homography(frame_kp, template_kp, good_matches):
    good_frame_kp = np.array([frame_kp[m.queryIdx].pt for m in good_matches])
    good_template_kp = np.array([template_kp[m.trainIdx].pt for m in good_matches])
    return cv2.findHomography(good_template_kp, good_frame_kp, cv2.RANSAC, 5.0)


def warp_images(back, front, H):
    height, width  = back.shape[:2]
    warped_front = cv2.warpPerspective(front, H, (width, height))
    back[warped_front > 0] = warped_front[warped_front > 0]
    return back
    

# ======= constants
INPUT_VIDEO_PATH = r'Videos/original.mp4'
OUTPUT_VIDEO_PATH = r'Videos/warped.mp4'
TEMPLATE_IMAGE_PATH = r'Images/louvre.png'
ANOTHER_IMAGE_PATH = r'Images\dog.png'

# === template image keypoint and descriptors
template_bgr = cv2.imread(TEMPLATE_IMAGE_PATH)
template_height, template_width = template_bgr.shape[:2]
template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_RGB2GRAY)
template_kp, template_desc = extract_keypoints_and_descriptors(template_gray)

# ===== video input, output and metadata
capture = cv2.VideoCapture(INPUT_VIDEO_PATH)

fps = int(capture.get(cv2.CAP_PROP_FPS))
frame_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    
codec = cv2.VideoWriter_fourcc(*"mp4v")
color = True
plot_delay = 1000 // fps

writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, codec, fps, (frame_width, frame_height), color)

print(f'Processing {num_frames} frames of size {frame_width}x{frame_height}, {num_frames / fps:.2f} seconds, {fps} f/s')

# ========== run on all frames
for i in range(100):

    frame_bgr = capture.read()[1]
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ====== find keypoints matches of frame and template
    # we saw this in the SIFT notebook
    
    frame_kp, frame_desc = extract_keypoints_and_descriptors(frame_gray)
    good_matches = find_matches(frame_kp, template_kp)

    # ======== find homography
    # also in SIFT notebook
    H, masked = find_homography(frame_kp, template_kp, good_matches)

    # ++++++++ do warping of another image on template image
    # we saw this in SIFT notebook
    another = cv2.imread(ANOTHER_IMAGE_PATH)
    another = cv2.resize(another, (template_width, template_height))

    warped = warp_images(frame_bgr, another, H)
    
    # =========== plot and save frame 
    writer.write(warped)

    if i % 100 == 0:
        print(f'{i}/{num_frames}')

# ======== end all
capture.release()
writer.release()
