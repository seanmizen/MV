import cv2 as cv
import numpy as np
import random as rng
import natsort
import pathlib
import math

# 1. greyscale
# 2. threshold
# 3. erode
# 4. contour
# 5. bounding box
# 6. label? somehow?
# 7. create global tracker variables for the ball including optimal kernel size

# return two greyscale images of the same dimensions
# accept grayscale images only
# so far only stretches 
def match_dimensions(img1, img2):
    #returns greyscale images, size matched
    print("matchdims")
    shape = np.maximum(img1.shape, img2.shape)
    img1_out = cv.resize(img1, shape)
    img2_out = cv.resize(img2, shape)
    return (img1_out, img2_out)

video_path = "./Assets/Tennis"
# frame_files = pathlib.Path(__file__).parent.joinpath("Assets/Tennis").glob("*ppm")
frame_files = sorted(pathlib.Path(video_path).glob("*ppm"))
frame_files = natsort.natsorted(frame_files, alg=natsort.PATH)
ball_reference_img = cv.imread("./Assets/BallReference.jpg", cv.IMREAD_GRAYSCALE) # Taken from frame 1

thresh = 222 # initial threshold
c_red = (240, 20, 20)
c_green = (20, 240, 20)
c_blue = (20, 20, 240)
c_yellow = (20, 200, 240)
c_black = (5, 5, 5)

ball_currently_tracked = False

for file_number, file in enumerate(frame_files):
    img = cv.imread(str(file),-1) # BGR by default
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 150 decided as a good value (given the erosion kernel is 3x3 - maybe this threshold should change?)
    # otherwise 222
    mask = cv.inRange(img_grey, 150, 255)
    img_grey_masked = cv.bitwise_and(img_grey, img_grey, mask=mask)
    cv.imshow("ImageGreyedMasked.jpg", img_grey_masked)

    # https://opencv4-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    # erode then un-erode to remove noise (potentially a better method than blurring)
    erosion_kernel = np.ones((3, 3), np.uint8) # accept nxn white areas only
    img_opened = cv.morphologyEx(img_grey_masked, cv.MORPH_OPEN, erosion_kernel)
    cv.imshow("First Image, Masked, opened", img_opened)
    # cv.waitKey(0)
    # up until now is good

    contours, hierarchy = cv.findContours(img_opened, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # f = cv.FONT_HERSHEY_PLAIN
    # best_matching_ball_id = -1
    best_matching_ball = ()
    best_image = np.zeros([1,1])
    best_similarity = - math.inf
    ball_captured = False
    
    for i, c in enumerate(contours):
        # Creates a rectangle for the potential object
        x, y, w, h = cv.boundingRect(c)
        if (w < 4) or (h < 4): continue
        sub_image = img_opened[y:y + h, x:x + w]

        (resized_image, resized_reference) = match_dimensions(sub_image, ball_reference_img)
        error_l2 = cv.norm(resized_image, resized_reference, cv.NORM_L2)
        similarity = 1 - error_l2 / ( h * w )
        print('Similarity = ', similarity)
        cv.rectangle(img, (x, y), (x + w, y + h), c_yellow, 1)

        # if similarity > best_similarity:
        if similarity > best_similarity:
            # ball_captured = True
            # ball_currently_tracked = True
            best_image = sub_image
            # cv.imshow(str(similarity), sub_image)
            # cv.waitKey(0)
            best_matching_ball = (x, y, w, h, similarity)

    # if ball_captured:
        
    
    # TODO account for no good ball found
    # TODO create "Searchbox" which expands when the ball disappears
    (x, y, w, h, similarity) = best_matching_ball
    cv.putText(img, "Frame " + str(file_number) , (2, 14), cv.FONT_HERSHEY_PLAIN, 1, c_black, 1, cv.LINE_AA)
    cv.putText(img, "Ball", (x - 2, y - 12), cv.FONT_HERSHEY_PLAIN, 1, c_green, 1, cv.LINE_AA)
    cv.putText(img, "Norm_L2: " + str(round(similarity,2)), (x - 2, y - 2), cv.FONT_HERSHEY_PLAIN, 0.8, c_green, 1, cv.LINE_AA)
    cv.rectangle(img, (x, y), (x + w, y + h), c_green, 2)
    
    cv.imshow("ImageWLabels.jpg", img)
    # if file_number == 6 or file_number == 26 or file_number == 40:
    #     cv.imwrite("ImageWithLabels(early) " + str(file_number) + ".png", img)
    # https://www.delftstack.com/howto/python/opencv-compare-images/

    cv.waitKey(0)
    # exit()
