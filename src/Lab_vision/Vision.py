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

# Computer Vision libary
import cv2
import numpy as np
import imutils
from numpy import asarray

import scipy.ndimage
import matplotlib.pyplot as plt
import math
from scipy.signal import argrelextrema
from bresenham import bresenham

# RANSAC algorithm
from numpy.linalg import inv

# Saving points into a file for the user to manually save
import csv


class Vision:
    '''
    Input the source of the image getting inspected and the angle(s) of rays to be drawn to collect data points.
    '''

    def __init__(self, image_source: str, angle_of_contact: float):
        self.image_source = image_source
        self.angle_of_contact = angle_of_contact

    def first_center_position(self):

        # First import the image in grayscale
        self.img = cv2.imread(self.image_source, 0)

        dimensions = self.img.shape
        # print(f'Actual dimension of the image', {dimensions})
        '''
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        '''
        self.width = 800
        self.height = 600
        dimension = (self.width, self.height)
        # print(dimension)
        self.resized_img = cv2.resize(self.img, dimension, interpolation=cv2.INTER_LINEAR)

        # Get the moments of the image and set that point as the initial point of the array
        array_data = asarray(self.resized_img)
        moment = cv2.moments(self.resized_img)
        X = int(moment['m10'] / moment["m00"])
        Y = int(moment["m01"] / moment["m00"])
        # TODO this needs my attention
        X = 427
        Y = 308
        # cv2.circle(self.resized_img, (X,Y), 2, (205,114,101), 5)
        # print(f' Moments Coordinate: ({X},{Y})')

        # make moments a member
        self.moments = ((X, Y))
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
        '''
        Is this because it is not finding the local minima?
        '''
        points = []
        self.inner_points = []
        self.outer_points = []
        for angle in np.arange(360, 0, -self.angle_of_contact):
            # TODO Look into why the angles are not working.
            if angle > 275:
                continue
            # # elif 170 < angle < 190:
            # #     continue
            elif angle < 65:
                continue
            endy = self.y0 + length * math.sin(math.radians(angle))
            endx = self.x0 + length * math.cos(math.radians(angle))
            points.append((endx, endy, angle))

        for i, point in enumerate(points[:]):
            p = list(bresenham(self.x0, self.y0, int(point[0]), int(point[1])))
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
            x1 = np.diff(a[:, 2])

            # Run the file creator here, and import the unfiltered inner_points, outer_points and intensity s
            # self.file_creator(np.array(intensity))

            # for i, intensity_value in enumerate(intensity[:]):
            #     print("Intensity values", intensity_value[2])
            #     d_measurement = np.hypot(self.x0, intensity_value[0])

            # plt.figure("Just the intensities")
            # plt.plot(x1[:-1])

            # Finding the outlier and inlier points using local min/max
            # Only storing the data points that we want to append onto the list
            # self.inner_points.append(a[np.argmax(x1), :2])
            self.inner_points.append(a[np.argmax(x1), :2])
            x2 = x1.copy()
            x2[x2 > -10] = 0
            # self.outer_points.append(a[np.argmin(x1), :2])
            index = x2.shape[0] - np.argmin(np.flip(x2))
            # if index < 200:
            #     continue
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
