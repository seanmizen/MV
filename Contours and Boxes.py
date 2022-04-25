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
    shape = np.maximum(img1.shape, img2.shape)
    img1_out = cv.resize(img1, shape)
    img2_out = cv.resize(img2, shape)
    return (img1_out, img2_out)

# accept grayscale images only
def image_compare(img1, img2):
    (resized_img1, resized_img2) = match_dimensions(img1, img2)
    error_l2 = cv.norm(resized_img1, resized_img2, cv.NORM_L2)
    (h, w) = resized_img1.shape
    similarity = 1 - error_l2 / ( h * w )
    return similarity

video_path = "./Assets/Tennis"
# frame_files = pathlib.Path(__file__).parent.joinpath("Assets/Tennis").glob("*ppm")
frame_files = sorted(pathlib.Path(video_path).glob("*ppm"))
frame_files = natsort.natsorted(frame_files, alg=natsort.PATH)
ball_reference_img = cv.imread("./Assets/BallReference.jpg", cv.IMREAD_GRAYSCALE) # Taken from frame 1

thresh = 222 # initial threshold
c_red = (20, 20, 240)
c_green = (20, 240, 20)
c_blue = (255, 10, 10)
c_yellow = (20, 200, 240)
c_black = (5, 5, 5)
c_white = (250, 250, 250)
c_search = c_white

# ball_currently_tracked = False
capture_mode = "capture" # capture, tracking, search
captured_balls = []
search_box = (0,0,1,1)

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

    contours, hierarchy = cv.findContours(img_opened, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    best_matching_ball = (0, 0, 1, 1, -math.inf, np.zeros([1,1])) #(x, y, w, h, similarity, img) 
    best_similarity = - math.inf

    candidate_ball_found = False
    for i, c in enumerate(contours):
        # Creates a rectangle for the potential object
        x, y, w, h = cv.boundingRect(c)
        if (w < 4) or (h < 4): continue
        sub_image = img_opened[y:y + h, x:x + w]
        similarity = image_compare(sub_image, ball_reference_img)
        # cv.rectangle(img, (x, y), (x + w, y + h), c_yellow, 1)

        #3 scenarios: initial capture, tracking, search
        if capture_mode == "capture": # initial capture
            # match the best "ball" on the screen, regardless of location
            if similarity > best_similarity:
                candidate_ball_found = True
                best_matching_ball = (x, y, w, h, similarity, sub_image)

        elif capture_mode == "tracking" or capture_mode == "search":
            # if this contour's centroid is within the search box, consider it
            # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
            contour_moment = cv.moments(c)
            cx = int(contour_moment['m10']/contour_moment['m00'])
            cy = int(contour_moment['m01']/contour_moment['m00'])
            within_search_x = search_box[0] <= cx and cx <= (search_box[0] + search_box[2])
            within_search_y = search_box[1] <= cy and cy <= (search_box[1] + search_box[3])
            cv.drawMarker(img, (cx,cy), color=c_yellow, markerType=cv.MARKER_CROSS, thickness=1)
            if within_search_x and within_search_y:
                candidate_ball_found = True
                if similarity > best_similarity:
                    best_matching_ball = (x, y, w, h, similarity, sub_image)
        
    (x, y, w, h, similarity, best_image) = best_matching_ball

    top_speed = 0
    search_box_expansion = 1

    if capture_mode == "search" and candidate_ball_found:
        capture_mode = "capture"

    if ((capture_mode == "tracking") and (not candidate_ball_found)) or capture_mode == "search":
        capture_mode = "search"
        # create search box from last known ball location
        x, y, w, h = search_box
        search_box = (x - int(w/2), y - int(h/2), w * 2, h * 2) #double size of last known ball position
        similarity = 0

    if capture_mode == "capture":
        capture_color = c_blue
        capture_mode = "tracking"
    elif capture_mode == "search":
        capture_color = c_red
    else:
        # implied capture_mode == "tracking"
        capture_color = c_green
    
    cv.putText(img, "Frame " + str(file_number) , (2, 14), cv.FONT_HERSHEY_PLAIN, 1, c_black, 1, cv.LINE_AA)

    if capture_mode == "search":
        (img_h, img_w, _) = img.shape
        if search_box[2] > (img_h/2) or search_box[3] > (img_w/2):
            capture_mode == "capture"
        cv.putText(img, "SEARCH", (x - 2, y - 2), cv.FONT_HERSHEY_PLAIN, 0.8, capture_color, 1, cv.LINE_AA)
    else:
        search_box = (x - int(w/2), y - int(h/2), w * 2, h * 2) #double size of last known ball position
        captured_balls.append(best_matching_ball)
        cv.putText(img, "Ball", (x - 2, y - 12), cv.FONT_HERSHEY_PLAIN, 1, capture_color, 1, cv.LINE_AA)
        cv.putText(img, "Norm_L2: " + str(round(similarity,2)), (x - 2, y - 2), cv.FONT_HERSHEY_PLAIN, 0.8, capture_color, 1, cv.LINE_AA)

    cv.rectangle(img, (x, y), (x + w, y + h), capture_color, 2)
    
    cv.imshow("ImageWLabels.jpg", img)
    # if file_number == 6 or file_number == 26 or file_number == 40:
    # cv.imwrite("SearchBox" + str(file_number) + ".png", img)

    cv.waitKey(0)
    # exit()
