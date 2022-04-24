import cv2
import numpy as np
import math
import os
import pathlib
import natsort
import glob


video_path = "./Assets/Tennis"
# frame_files = pathlib.Path(__file__).parent.joinpath("Assets/Tennis").glob("*ppm")
frame_files = sorted(pathlib.Path(video_path).glob("*ppm"))
frame_files = natsort.natsorted(frame_files, alg=natsort.PATH)
for file in frame_files:
    img = cv2.imread(str(file),-1)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower = np.array([210, 230, 230])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower, upper)

    kernel_size = 1 # powers of 3 only
    img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    mask_blurred = cv2.inRange(img_blurred, lower, upper)

    img_masked = cv2.bitwise_and(img_blurred, img_blurred, mask=mask_blurred)
    img_masked_grey = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(img_grey, 50, 150, apertureSize=3)

    cv2.imshow("img", img)
    # cv2.imshow("img masked", img_masked)
    # cv2.imshow("img blurred", img_blurred)
    # cv2.imshow("img grey", img_grey)
    cv2.imshow("img masked grey", img_masked_grey)




# further work:

video_path = "./Assets/Tennis"
# frame_files = pathlib.Path(__file__).parent.joinpath("Assets/Tennis").glob("*ppm")
frame_files = sorted(pathlib.Path(video_path).glob("*ppm"))
frame_files = natsort.natsorted(frame_files, alg=natsort.PATH)
for file in frame_files:
    img = cv2.imread(str(file),-1) # BGR by default
    # img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower = np.array([200, 220, 220])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(img, lower, upper)

    kernel_size = 3 # powers of 3 only
    img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    mask_blurred = cv2.inRange(img_blurred, lower, upper)

    # img_masked = cv2.bitwise_and(img_blurred, img_blurred, mask=mask)
    img_masked = cv2.bitwise_and(img, img, mask=mask)
    img_masked_grey = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)

    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    # erode then un-erode to remove noise (potentially a better method than blurring)
    # kernel = np.ones((2, 2), np.uint8) # accept 2x2 white areas only
    kernel = np.ones((2, 2), np.uint8)

    img_opened = cv2.morphologyEx(img_masked_grey, cv2.MORPH_OPEN, kernel)
    # img_closed = cv2.morphologyEx(img_masked, cv2.MORPH_CLOSE, kernel)    

    edges = cv2.Canny(img_masked_grey, 50, 150, apertureSize=3)

    cv2.imshow("img", img)
    # cv2.imshow("img masked", img_masked)
    # cv2.imshow("img blurred", img_blurred)
    # cv2.imshow("img grey", img_grey)
    # cv2.imshow("img masked grey", img_masked_grey)
    cv2.imshow("img morphology opened", img_opened)
