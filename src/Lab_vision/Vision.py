'''
1) Find the arbitrary center - know it approximately as I will position the camera
    a) We know the center since we are setting the location of the camera
    b) Alternative method I am selecting it by moments
2) Change the image into grayscale
3) Using a line equation draw a line straight up and extract the all the pixel values -matplotlib
4) Use the Gaussian derivative to find the initial and ending point of the parameters of the coil - save it into a separate array
5) Do this for every 5 degrees of the circle itself.
6) Connect the inlier and outlier using Ransac (3points for sphere, 5 points for ellipse)
7) Most accurate parameter of the shape.
'''
# TODO: Need to add third angle of ignorance for both inner and outer, might have to separate the angle of interest for inner and outer points
# Computer Vision libary
import cv2
import numpy as np
import imutils

import matplotlib.pyplot as plt
import math
from scipy.signal import argrelextrema
from bresenham import bresenham

# Saving points into a file for the user to manually save
import csv

class Vision:
    '''
    Input the source of the image getting inspected and the angle(s) of rays to be drawn to collect data points.
    '''

    def __init__(self, image_source: str, angle_of_contact: float, black_white_threshold, angle_of_detection):
        self.image_source = image_source
        self.angle_of_contact = angle_of_contact
        self.black_white_threshold = black_white_threshold
        self.angle_of_detection = angle_of_detection

    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        '''
        Maintaining aspect ratio for images
        https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
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

    def first_center_position(self):
        '''
        This function returns the dimensions of the resized image with the maintained aspect ratio.
        '''
        # First import the image in grayscale
        self.img = cv2.imread(self.image_source, 0)

        # Calling the resized image
        self.resized_img = self.resize_with_aspect_ratio(self.img, width=800)

        print("dimensions of the resized image - same aspect ratio", self.resized_img.shape)
        # cv2.imshow('resize', self.resized_img)
        # cv2.waitKey()

        self.height = self.resized_img.shape[0]
        self.width = self.resized_img.shape[1]
        # print(self.height, self.width)

        # User always needs to input the location of the center points.
        X = 322
        Y = 323
        #self.moments = ((X, Y))
        self.x0 = X
        self.y0 = Y
        return X, Y

    def inner_outer_points(self):
        '''
        Drawing the inner and outer points, with the option to exclude
        certain polar angles if it is not fit. Need to write a shell script that
        connects the user input to this module
        '''
        # The x0, y0 needs to be the moments of the circle and x1, y1 needs to be the second point of the angle
        # length = np.sqrt((self.width - self.x0) ** 2 + (self.height - self.y0) ** 2)
        # length = 300

        # Hardcoded the ray values to be 75 percent of the total height of the image.
        length = self.height * 3 / 4
        points = []
        self.inner_points = []
        self.outer_points = []
        for angle in np.arange(360, 0, -self.angle_of_contact):
            '''
            Note that that for some of the images the angle of ignorance should be hardcoded.
            '''
            if angle > 55 and angle < 75:
                continue
            # elif angle < 270:
            #     continue

            # Input angles from computer_vision main script.
            if len(self.angle_of_detection) ==1:
                if angle >= self.angle_of_detection[0] and angle <= self.angle_of_detection[1]:
                    endy = self.y0 + length * math.sin(math.radians(angle))
                    endx = self.x0 + length * math.cos(math.radians(angle))
                    points.append((endx, endy, angle))
            else:
                if angle >= self.angle_of_detection[0][0] and angle <= self.angle_of_detection[0][1] \
                        or angle >= self.angle_of_detection[1][0] and angle <= self.angle_of_detection[1][1]:
                    endy = self.y0 + length * math.sin(math.radians(angle))
                    endx = self.x0 + length * math.cos(math.radians(angle))
                    points.append((endx, endy, angle))


        for i, point in enumerate(points[:]):
            p = list(bresenham(self.x0, self.y0, int(point[0]), int(point[1])))
            # Make a list to store the intensity for each of the points along the points that the rays are drawn over.
            intensity = []
            for pnt in p:
                # Setting the boundaries for where the rays should reach
                if pnt[0] >= self.width or pnt[1] >= self.height or pnt[0] < 0 or pnt[1] < 0:
                    continue
                else:
                    intensity.append([pnt[0], pnt[1], self.resized_img[pnt[1], pnt[0]]])

            a = np.array(intensity)
            # print(a)
            plt.figure("Rays Drawing from Class Vision")
            plt.imshow(self.resized_img)
            plt.plot([self.x0, point[0]], [self.y0, point[1]], 'r')

            # Plot the intensity to see the the changes
            # plt.figure()
            # plt.plot(a[:,2])
            # plt.plot(x1)

            a[a[:, 2] < self.black_white_threshold, 2] = 0
            a[a[:, 2] > self.black_white_threshold, 2] = 255

            x1 = np.diff(a[:, 2])
            '''
            Finding the outlier and inlier points using local min/max
            Only storing the data points that we want to append onto the list
            '''
            # self.inner_points.append(a[np.argmax(x1), :2])
            self.inner_points.append(a[np.argmax(x1) + 1, :2])
            x2 = x1.copy()
            # Difference of intensity.
            # x2[x2 > -5] = 0

            # self.outer_points.append(a[np.argmin(x1), :2])
            # Flipping the array and going backwards for the outer (blue) points and subtracting it from the real shape to obtain the actual index
            index = x2.shape[0] - np.argmin(np.flip(x2))


            # Adding the points onto the rays diagram
            self.outer_points.append(a[index, :2])
            plt.plot(self.inner_points[-1][0], self.inner_points[-1][1], 'ro')
            plt.plot(self.outer_points[-1][0], self.outer_points[-1][1], 'bo')
            # plt.show()

        plt.figure("Class Vision")
        inner_points = np.array(self.inner_points)
        outer_points = np.array(self.outer_points)
        # Sort the points in the list of differentiation
        # a1 = outer_points.sort()
        # print(a1)

        plt.imshow(self.resized_img)
        plt.plot(inner_points[:, 0], inner_points[:, 1], 'ro')
        plt.plot(outer_points[:, 0], outer_points[:, 1], 'bo')

        # plt.plot(x0,y0, 'g+')
        # plt.plot(np.mean(inner_points[:, 0]), np.mean(inner_points[:, 1]), 'go')

        return np.array(self.inner_points), np.array(self.outer_points)

    def file_creator(self, intensity):
        '''
        Import the X, Y coordinates and the intensities to an excel/csv file
        Also should
        Export the the matplolib image to the folder.
        https://stackoverflow.com/questions/17068966/how-do-you-write-several-arrays-into-an-excel-file
         *** Need to look into how to do edit excel sheets in python
        '''
        # print(inner_points)
        import pandas as pd

        # data = pd.DataFrame(
        #         {'All the Points': intensity,
        #         })
        data1 = np.array(intensity[:, 0])
        data2 = np.array(intensity[:, 1])
        data3 = np.array(intensity[:, 2])

        data = pd.DataFrame(
            {'x_coordinate': data1,
             'y_coordinate': data2,
             'intensity value': data3
             })

        # Writing to Excel
        data_to_excel = pd.ExcelWriter("From_python.xlsx", engine='xlsxwriter')

        data.to_excel(data_to_excel, sheet_name="Sheet1")

        data_to_excel.save()
