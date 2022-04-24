import cv2 as cv
import numpy as np
import random as rng
import natsort
import pathlib

video_path = "./Assets/Tennis"
# frame_files = pathlib.Path(__file__).parent.joinpath("Assets/Tennis").glob("*ppm")
frame_files = sorted(pathlib.Path(video_path).glob("*ppm"))
frame_files = natsort.natsorted(frame_files, alg=natsort.PATH)
thresh = 222 # initial threshold
for file in frame_files:
    img = cv.imread(str(file),-1) # BGR by default
    # cv.imshow("First Image", img)

    # cv.imshow("First Image Grey.jpg", img_grey)

    ### noise filter 0: greyscale, threshold ###
    img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur_kernel_size = 9 # powers of 3 only
    lower = np.array([222, 222, 222])
    upper = np.array([255, 255, 255])
    mask = cv.inRange(img_grey, 222, 255)
    img_grey_masked = cv.bitwise_and(img_grey, img_grey, mask=mask)
    cv.imshow("ImageGreyedMasked.jpg", img_grey_masked)

    ### noise filter 1: blur ###
    blur_kernel_size = 9 # powers of 3 only
    img_blurred = cv.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    # cv.imshow("First Image, blurred", img_blurred)

    lower = np.array([200, 220, 220])
    upper = np.array([255, 255, 255])
    mask = cv.inRange(img, lower, upper)
    mask_blurred = cv.inRange(img_blurred, lower, upper)
    img_masked = cv.bitwise_and(img, img, mask=mask)
    img_masked_grey = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)
    # cv.imshow("First Image, masked", img_masked)
    # cv.imshow("First Image, masked, grey", img_masked_grey)
    img_blurmasked = cv.bitwise_and(img_blurred, img_blurred, mask=mask)
    img_blurmasked_grey = cv.cvtColor(img_blurmasked, cv.COLOR_BGR2GRAY)
    # cv.imshow("First Image, blurmasked, grey", img_blurmasked_grey)

    ### noise filter 2: erosion ###
    # https://opencv4-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    # erode then un-erode to remove noise (potentially a better method than blurring)
    erosion_kernel = np.ones((3, 3), np.uint8) # accept nxn white areas only
    img_opened = cv.morphologyEx(img_masked_grey, cv.MORPH_OPEN, erosion_kernel)
    # cv.imshow("First Image, Masked, opened", img_opened)

    # cv.waitKey(0)

    # testing zone:
    # https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
    # CV trackbar is incredible for figuring out correct thresholds

    def mask_callback(val):
        threshold = val
        lower = np.array([val, val, val])
        upper = np.array([255, 255, 255])

        mask = cv.inRange(img, lower, upper)
        mask_blurred = cv.inRange(img_blurred, lower, upper)
        img_masked = cv.bitwise_and(img, img, mask=mask)
        img_masked_grey = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)
        cv.imshow("First Image, masked", img_masked)
        # cv.imshow("First Image, masked, grey", img_masked_grey)
        img_blurmasked = cv.bitwise_and(img_blurred, img_blurred, mask=mask)
        img_blurmasked_grey = cv.cvtColor(img_blurmasked, cv.COLOR_BGR2GRAY)
        # cv.imshow("First Image, blurmasked, grey", img_blurmasked_grey)

        erosion_kernel = np.ones((2, 2), np.uint8) # accept nxn white areas only
        img_opened = cv.morphologyEx(img_masked, cv.MORPH_OPEN, erosion_kernel)
        cv.imshow("First Image, Masked, opened", img_opened)
        
        canny_output = cv.Canny(thresh_img, threshold, threshold * 2)
        
        aa, contours = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        contours_poly = [None]*len(aa)
        boundRect = [None]*len(aa)
        centers = [None]*len(aa)
        radius = [None]*len(aa)
        for i, c in enumerate(aa):
            contours_poly[i] = cv.approxPolyDP(c, 3, True)
            boundRect[i] = cv.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
        
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        
        for i in range(len(contours)):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours_poly, i, color)
            cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
            (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        
        cv.imshow('Contours', canny_output)

        print(val)

    thresh_img = img_opened

    thresh_window = 'Threshold checker'
    cv.namedWindow(thresh_window)
    cv.imshow(thresh_window, img)

    max_thresh = 255
    cv.createTrackbar('Canny thresh:', thresh_window, thresh, max_thresh, mask_callback)
    mask_callback(thresh)

    cv.waitKey(0)
exit()

# 120 is a good thresh val - follow this up with a 3x3(?) MORPH_OPEN
