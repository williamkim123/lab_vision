'''
Need to see if we can set the threshold for the background using through the hardware.
'''

import cv2
import numpy as np
import imutils
from PIL import Image, ImageStat


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    '''
    Maintaining aspect ratio for images
    '''
    dim = None
    (h, w) = image.shape[:2]

    # print(h, w)

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def background(source):
    ''''
    Used the resource: https://stackoverflow.com/questions/35968598/how-to-find-and-measure-arc-length-within-an-image
    '''
    img = cv2.imread(source)

    img = resize_with_aspect_ratio(img, width=800)
    # if img.shape[1] < 600:
    #     img = imutils.resize(img, width=800)

    # light yellow and dark red -> BGR, setting the threshold from the faintest yellow to that of darket red.
    lower = np.array([102, 255, 255], dtype="uint8")
    upper = np.array([0, 0, 255], dtype="uint8")

    # resize the frame, convert it to the HSV color space and determine the HSV pixel intensities that fall into the lower and upper bound
    frame = imutils.resize(img, width=800)
    HSV_coversion = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(HSV_coversion, lower, upper)

    # apply the a series of erosions/dilations to the mask with the elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

    # show the skin in the image along with the mask
    cv2.imwrite('output.jpg', np.hstack([skin]))

    image = cv2.imread('output.jpg')
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    # cv2.imshow("Input", image)

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresh", thresh)
    cv2.imwrite("../../Images/black_white.jpg", thresh)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]

    # *** Return back an error message if there is more than 1 contour found.
    print("[INFO] {} unique contours found".format(len(cnts)))

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the contour
        ((x, y), _) = cv2.minEnclosingCircle(c)
        cv2.putText(image, "#{}".format(i), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Image - Outline", image)
    cv2.waitKey(0)


background('..\..\Images\\image_t2.jpg')