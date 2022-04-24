#! /usr/bin/python
# -*- coding: utf-8 -*-
"""Isolate Green from an image"""
import cv2
import numpy as np

# to get local dirname, use:
# os.path.dirname(os.path.abspath(__file__)

if __name__ == "__main__":
    img = cv2.imread("./Assets/testimage4.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect Ground
    lower_green = np.array([1, 40, 1])
    upper_green = np.array([70, 255, 200])

    mask = cv2.inRange(img, lower_green, upper_green)
    img = cv2.bitwise_and(img, img, mask=mask)  # why?
    # --------------------------------------------

    cv2.imshow("Masked", img)
    cv2.waitKey(0)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((13, 13), np.uint8)
    thresh = cv2.threshold(
        img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Masked", thresh)
    cv2.waitKey(0)
