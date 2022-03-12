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
        print(f'Actual dimension of the image', {dimensions})
        '''
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        '''
        self.width = 800
        self.height = 600
        dimension = (self.width, self.height)
        print(dimension)
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
        print(f' Moments Coordinate: ({X},{Y})')

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
        # length = np.sqrt((Obj.width - x0) ** 2 + (Obj.height - y0) ** 2)/2
        length = self.height / 2
        '''
        Is this because it is not finding the local minima?
        '''
        points = []
        self.inner_points = []
        self.outer_points = []
        for angle in np.arange(0, 360, self.angle_of_contact):
            if angle < 65:
                continue
            elif angle > 270:
                continue
            endy = self.y0 + length * math.sin(math.radians(angle))
            endx = self.x0 + length * math.cos(math.radians(angle))
            points.append((endx, endy, angle))

        for i, point in enumerate(points[:]):
            p = list(bresenham(self.x0, self.y0, int(point[0]), int(point[1])))
            intensity = []
            for pnt in p:
                if pnt[0] >= self.width or pnt[1] >= self.height or pnt[0] < 0 or pnt[1] < 0:
                    continue
                else:
                    intensity.append([pnt[0], pnt[1], self.resized_img[pnt[1], pnt[0]]])

            a = np.array(intensity)
            plt.imshow(self.resized_img)
            plt.plot([self.x0, point[0]], [self.y0,  point[1]], 'r')
            x1 = np.diff(a[:, 2])
            # yo = np.array(x1)
            # print(y0)
            # print(type(yo))
            '''
            Storing the differentiation values sorted in the order of smallest x distance to greatest.
            '''
            # plt.figure("Just the intensities")
            # plt.plot(x1[:-1])

            '''
            Finding the outlier and inlier points using local min/max
            self.inner_points.append(a[np.argmax(x1), :2])
            '''
            if 0 < point[2] < 90:
                pass

            elif 275 < point[2] < 360:
                pass
            else:
                self.outer_points.append(a[np.argmin(x1), :2])
                self.inner_points.append(a[np.argmax(x1), :2])

        plt.figure("Image Plot")
        inner_points = np.array(self.inner_points)
        outer_points = np.array(self.outer_points)
        # Sort the points in the list of differentiation
        # a1 = outer_points.sort()
        # print(a1)

        plt.imshow(self.resized_img)
        plt.plot(inner_points[:, 0], inner_points[:, 1], 'ro')
        plt.plot(outer_points[:, 0], outer_points[:, 1], 'bo')

        # plt.plot(x0,y0, 'g+')
        plt.plot(np.mean(inner_points[:, 0]), np.mean(inner_points[:, 1]), 'go')

        return np.array(self.inner_points), np.array(self.outer_points)

    def file_creator(self, input):
        '''
        Import the X, Y coordinates and the intensities to an excel/csv file
        Also should
        Export the the matplolib image to the folder.
        https://stackoverflow.com/questions/17068966/how-do-you-write-several-arrays-into-an-excel-file
         *** Need to look into how to do edit excel sheets in python
        '''
        import xlwt
        from tempfile import TemporaryFile
        book = xlwt.Workbook()
        sheet1 = book.add_sheet('sheet1')

        a = [1, 2, 3, 4, 5]
        b = [6, 7, 8, 9, 10]
        c = [2, 3, 4, 5, 6]

        data = [a, b, c]

        for row, array in enumerate(data):
            for col, value in enumerate(array):
                sheet1.write(row, col, value)

        name = "this.xls"
        book.save(name)
        book.save(TemporaryFile())

