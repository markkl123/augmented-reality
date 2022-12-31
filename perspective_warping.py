# ======= imports
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ======= helper functions
def plot_image(image, title=""):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.title(title)
    plt.show()


def extract_keypoints_and_descriptors(image):
    feature_extractor = cv2.SIFT_create()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return feature_extractor.detectAndCompute(gray_image, None)


def find_matches(frame_desc, template_desc):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(frame_desc, template_desc, k=2)
    good_and_second_good_match_list = [m for m in matches if m[0].distance/m[1].distance < 1]
    return np.asarray(good_and_second_good_match_list)[:,0]


def find_homography(frame_kp, template_kp, good_matches):
    good_frame_kp = np.array([frame_kp[m.queryIdx].pt for m in good_matches])
    good_template_kp = np.array([template_kp[m.trainIdx].pt for m in good_matches])
    return cv2.findHomography(good_template_kp, good_frame_kp, cv2.RANSAC, 5.0)


def warp_images(back, front, H):
    height, width = back.shape[:2]
    warped_front = cv2.warpPerspective(front, H, (width, height))
    back[warped_front > 0] = warped_front[warped_front > 0]
    return back
    

# ======= constants
ORIGINAL_VIDEO_PATH = r'Videos\original.mp4'
WARPED_VIDEO_PATH = r'Videos\warped.mp4'
TEMPLATE_IMAGE_PATH = r'Images\louvre.png'
ANOTHER_IMAGE_PATH = r'Images\dog.png'


if __name__ == '__main__':

    # ======= template image keypoint and descriptors
    template = cv2.imread(TEMPLATE_IMAGE_PATH)
    template_height, template_width = template.shape[:2]
    template_kp, template_desc = extract_keypoints_and_descriptors(template)

    # ======= video input, output and metadata
    capture = cv2.VideoCapture(ORIGINAL_VIDEO_PATH)

    fps = int(capture.get(cv2.CAP_PROP_FPS))
    frame_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))    
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    color = True
    plot_delay = 1000 // fps

    writer = cv2.VideoWriter(WARPED_VIDEO_PATH, codec, fps, (frame_width, frame_height), color)

    print(f'Processing {num_frames} frames of size {frame_width}x{frame_height}, {num_frames / fps:.2f} seconds, {fps} f/s')

    # ======= run on all frames
    for i in range(num_frames):

        frame = capture.read()[1]

        # ======= find keypoints matches of frame and template
        # we saw this in the SIFT notebook
        frame_kp, frame_desc = extract_keypoints_and_descriptors(frame)
        good_matches = find_matches(frame_desc, template_desc)

        # ======= find homography
        # also in SIFT notebook
        H, masked = find_homography(frame_kp, template_kp, good_matches)

        # +++++++ do warping of another image on template image
        # we saw this in SIFT notebook
        another = cv2.imread(ANOTHER_IMAGE_PATH)
        another = cv2.resize(another, (template_width, template_height))
        warped = warp_images(frame, another, H)

        # ======= plot and save frame 
        writer.write(warped)

        if i % 100 == 0:
            print(f'{i}/{num_frames}')

    # ======= end all
    capture.release()
    writer.release()
