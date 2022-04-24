import cv2
import numpy as np
import math
import os
import pathlib
import natsort
import glob

# file renaming script:
# frame_files = pathlib.Path(__file__).parent.joinpath("Assets/Tennis").glob("*ppm")
# frame_files = natsort.natsorted(frame_files, alg=natsort.PATH)
# for file in frame_files:
#     start = str(file)[:str(file).find(".")]
#     nums = str(file)[str(file).find(".") + 1 :-str(file)[::-1].find(".") - 1]
#     end = str(file)[-str(file)[::-1].find("."):]
#     nums = "000" + nums
#     nums = nums[-3:]
#     newPath = start + "." + nums + "." + end
#     os.rename(str(file), newPath)

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

    if False:
        kernel = np.ones((13, 13), np.uint8)
        thresh = cv2.threshold(img_masked_grey, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Contours are found in the image to show the edges of each object so identiying individual players is easier. It stores the contours it finds ina  list contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contours:
            # Draws a rectangle around the potential object
            x, y, w, h = cv2.boundingRect(c)

            if ((h >= 1 and w >= 1) and (h <= 30 and w <= 30)): # limit size
                object_img = img[y:y + h, x:x + w]

                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                # white ball  detection
                mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)

                if (nzCount >= 3):
                    # detect football
                    cv2.putText(image, "football", (x - 2, y - 2), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.waitKey(0)

# vidcap method
# vidcap = cv2.VideoCapture("Assets/Tennis/.*%03d.ppm", cv2.CAP_IMAGES)
# print(vidcap._asdict())

exit()

while True:
    success, image = vidcap.read()
    print(type(image))
    # cv2.imshow("Tennis Detection", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cv2.destroyAllWindows()

exit()

# //-------------------------------------
# Defining Colours
# Green
# Green
lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

# Blue
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# Red
lower_red = np.array([0, 31, 255])
upper_red = np.array([176, 255, 255])

# White
lower_white = np.array([0, 0, 0])
upper_white = np.array([0, 0, 255])
# //-------------------------------------

vidcap = cv2.VideoCapture("./Assets/cutvideo.mp4")

success, image = vidcap.read()
count = 0
success = True
idx = 0

# cv2.imread(str(i) + '.png')

while success:
    # //-----------------------------------------------------------------------------------------------------------------
    mask = cv2.inRange(image, lower_green, upper_green)
    img = cv2.bitwise_and(image, image, mask=mask)

    # Blur Image
    kernel_size = 9
    blur_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Use canny edge detection
    gray = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=70,  # Min number of votes for valid line
        minLineLength=40,  # Min allowed length of line
        maxLineGap=20  # Max allowed gap between line for joining them
    )

    if lines is not None:
        new_lines = []
        longest = [[0, 0, 0, 0]]
        for i in range(len(lines)):
            l = lines[i]
            x1, y1, x2, y2 = l[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
            if (angle < 45) and (angle > -45):
                pass
            elif (angle == 45):
                pass
            else:
                new_lines.append(l)

        for i in range(len(new_lines)):
            l = lines[i]
            x1, y1, x2, y2 = l[0]
            c1, v1, c2, v2 = longest[0]
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            storeddist = math.sqrt((c2 - c1) ** 2 + (v2 - v1) ** 2)
            if dist > storeddist:
                longest = None
                longest = [l[0]]
        # //-----------------------------------------------------------------------------------------------------------------
        # Converting image from RGB to HSV wjhich gives it a hue value 0 and 180
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Creating a mask which only selects pixels that are within a certain range in this case the colour range of green
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Apply the mask to the image
        res = cv2.bitwise_and(image, image, mask=mask)
        # Converting the image from HSV back to RGB
        res_bgr = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        # Converting the image to grayscale
        res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        # ?
        kernel = np.ones((13, 13), np.uint8)
        # ?
        thresh = cv2.threshold(res_gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # ?
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Contours are found in the image to show the edges of each object so identiying individual players is easier. It stores the contours it finds ina  list contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        prev = 0
        # Defining a font to be used later
        font = cv2.FONT_HERSHEY_SIMPLEX
        playerpoints=[]
        # analysing all ther contours it found
        for c in contours:
            # Draws a rectangle around the potential object
            x, y, w, h = cv2.boundingRect(c)

            # Detect if the object is a player
            if (h >= (1.5) * w):
                if (w > 15 and h >= 15):
                    idx = idx + 1
                    player_img = image[y:y + h, x:x + w]
                    player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                    # If player has blue jersy
                    mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                    res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                    nzCount = cv2.countNonZero(res1)
                    # If player has red jersy
                    mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                    res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2BGR)
                    res2 = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
                    nzCountred = cv2.countNonZero(res2)

                    if (nzCount >= 20):
                        # Mark blue jersy players as france
                        cv2.putText(image, "France", (x - 2, y - 2), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        x, y, w, h = cv2.boundingRect(c)
                        middlex = (x + (w/2))
                        point = (middlex, y+h)
                        playerpoints.append(point)

                    else:
                        pass
                    if (nzCountred >= 20):
                        # Mark red jersy players as belgium
                        cv2.putText(image, "Belgium", (x - 2, y - 2), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        x, y, w, h = cv2.boundingRect(c)
                        middlex = (x + (w/2))
                        point = (middlex, y+h)
                        playerpoints.append(point)
                        
                    else:
                        pass
            if ((h >= 1 and w >= 1) and (h <= 30 and w <= 30)):
                player_img = image[y:y + h, x:x + w]

                player_hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
                # white ball  detection
                mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2GRAY)
                nzCount = cv2.countNonZero(res1)

                if (nzCount >= 3):
                    # detect football
                    cv2.putText(image, "football", (x - 2, y - 2), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            if len(playerpoints) == 0:
                continue

# //-----------------------------------------------------------------------------------------------------------------
# Line Mover
        x1, y1, x2, y2 = longest[0]
        height, width = image.shape[:2]
        show_all_lines = False

        playerlines=[]
        cs=[]

        if y1 and y2 == 0:
            gradient = 0
        elif (y1 == y2):
            gradient = 0
        else:
            gradient = (y2-y1)/(x2-x1)
            
        if gradient != 0:
            #largest y
            for points in playerpoints:
                x3,y3=points
                if x2 > x1:
                    diffx = x3 - x1
                    diffy = y3 - y1
                    x1 += diffx
                    x2 += diffx
                    y1 += diffy
                    y2 += diffy
                    playerline = (x1,y1,x2,y2)
                    playerlines.append(playerline)
                if x1 > x2:
                    diffx = x3 - x2
                    diffy = y3 - y2
                    x1 += diffx
                    x2 += diffx
                    y1 += diffy
                    y2 += diffy
                    playerline = (x1,y1,x2,y2)
                    playerlines.append(playerline)
            for line in playerlines:
                x1, y1, x2, y2 = line
                c = -((gradient * x1) - y1)
                cs.append(c)
                if show_all_lines:
                    cv2.line(image, (0, int(c)), (int(width), int((gradient * width) + c)), (255, 0, 0), 2)
            if len(cs) > 0:
                target_c = min(cs)
                cv2.line(image, (0, int(target_c)), (int(width), int((gradient * width) + target_c)), (0, 255, 0), 2)
               
        count += 1
        #cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imshow("Offside Detection", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        success, image = vidcap.read()

vidcap.release()
cv2.destroyAllWindows()
