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
        # cv2.circle(self.resized_img, (X,Y), 2, (205,114,101), 5)
        print(f' Moments Coordinate: ({X},{Y})')

        # make moments a member
        self.moments = ((X, Y))
        return X, Y

    def inner_outer_points(self):
        '''
        Drawing the inner and outer points, with the option to exclude
        certain polar angles if it is not fit. Need to write a shell script that
        connects the user input to this module
        '''
        # The x0, y0 needs to be the moments of the circle and x1, y1 needs to be the second point of the angle
        # length = np.sqrt((Obj.width - x0) ** 2 + (Obj.height - y0) ** 2)/2
        length = Obj.height / 2
        '''
        Is this because it is not finding the local minima?
        '''
        points = []
        self.inner_points = []
        self.outer_points = []
        for angle in np.arange(0, 360, Obj.angle_of_contact):
            # if 45<angle<90:
            #     continue
            endy = y0 + length * math.sin(math.radians(angle))
            endx = x0 + length * math.cos(math.radians(angle))
            points.append((endx, endy, angle))

        for i, point in enumerate(points[:]):
            # a = Obj.createLineIterator((x0, y0), point, Obj.resized_img)
            p = list(bresenham(x0, y0, int(point[0]), int(point[1])))
            intensity = []
            for pnt in p:
                if pnt[0] >= self.width or pnt[1] >= self.height or pnt[0] < 0 or pnt[1] < 0:
                    continue
                else:
                    intensity.append([pnt[0], pnt[1], Obj.resized_img[pnt[1], pnt[0]]])

            a = np.array(intensity)
            # print(a)

            # plt.plot([x0, point[0]], [y0,  point[1]], 'r')
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
            if 0 < point[2] < 0:
                pass

            elif 0 < point[2] < 0:
                pass
            # include all
            # elif == 0:
            #     self.outer_points.append(a[np.argmin(x1), :2])
            #     self.inner_points.append(a[np.argmax(x1), :2])
            else:
                self.outer_points.append(a[np.argmin(x1), :2])
                self.inner_points.append(a[np.argmax(x1), :2])

        plt.figure("Image Plot")
        inner_points = np.array(self.inner_points)
        outer_points = np.array(self.outer_points)
        # Sort the points in the list of differentiation
        # a1 = outer_points.sort()
        # print(a1)

        plt.imshow(Obj.resized_img)
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


class RANSAC(Vision):
    '''
    Inspired from: https://github.com/SeongHyunBae
    '''

    def __init__(self, innerpoints, outerpoints, n_iteration):
        # trial run
        self.inner_points = innerpoints
        self.outer_points = outerpoints

        self.x1_inner = innerpoints[:, 0]
        self.y1_inner = innerpoints[:, 1]
        self.x2_outer = outerpoints[:, 0]
        self.y2_outer = outerpoints[:, 1]
        self.n = n_iteration
        self.d_min = 99999
        self.best_model = None

    def random_sampling(self, sample):
        sampling_points = []
        saved_points = []
        count = 0
        if sample =='outer':
            sample_dataX = self.x2_outer
            sample_dataY = self.y2_outer
        elif sample == 'inner':
            sample_dataX = self.x1_inner
            sample_dataY = self.y1_inner
        else:
            print('Error')

        # getting three data points from the sample
        while True:
            ran_samples = np.random.randint(len(sample_dataX))

            if ran_samples not in saved_points:
                sampling_points.append((sample_dataX[ran_samples], sample_dataY[ran_samples]))
                saved_points.append(ran_samples)
                count += 1

            if count == 3:
                break

        return sampling_points

    def model_create(self, sampling_points):
        '''
        :param sampling_points:
        :return: Values for the three (a,b,c) sampling points through matrix calculation
        '''

        pt1 = sampling_points[0]
        pt2 = sampling_points[1]
        pt3 = sampling_points[2]

        # Need to understand this code
        a = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]], [pt3[0] - pt2[0], pt3[1] - pt2[1]]])
        b = np.array([[pt2[0] ** 2 - pt1[0] ** 2 + pt2[1] ** 2 - pt1[1] ** 2],
                      [pt3[0] ** 2 - pt2[0] ** 2 + pt3[1] ** 2 - pt2[1] ** 2]])
        # There is problem with the inverse operation on the code
        inv_A = inv(a)

        c_x, c_y = np.dot(inv_A, b) / 2
        c_x, c_y = c_x[0], c_y[0]
        r = np.sqrt((c_x - pt1[0]) ** 2 + (c_y - pt1[1]) ** 2)

        return c_x, c_y, r

    def evaluate_model(self, model, sample):
        d = 0
        c_x, c_y, r = model

        '''
        Need to go over the this RANSAC algorithm
        '''
        if sample =='outer':
            sample_dataX = self.x2_outer
            sample_dataY = self.y2_outer
        elif sample == 'inner':
            sample_dataX = self.x1_inner
            sample_dataY = self.y1_inner
        else:
            print('Error')

        for i in range(len(sample_dataX)):
            for i in range(len(sample_dataX)):
                dis = np.sqrt((sample_dataX[i] - c_x) ** 2 + (sample_dataY[i] - c_y) ** 2)

                if dis >= r:
                    d += dis - r
                else:
                    d += r - dis

            return d

    def execute_ransac(self, sample):
        '''
        Finding the optimal model
        '''
        for i in range(self.n):
            model = self.model_create(self.random_sampling(sample))
            d_temp = self.evaluate_model(model, sample)

            if self.d_min > d_temp:
                if sample == 'inner':
                    self.best_model_inner = model
                    self.d_min = d_temp
                elif sample == 'outer':
                    self.best_model_outer = model
                    self.d_min = d_temp
                else:
                    print("error")
    def plot_points(self, point1,point2, rad1, point3, point4,rad2):

        plt.scatter(self.x1_inner, self.y1_inner, c='red', marker='o', label='data')
        plt.scatter(self.x2_outer, self.y2_outer, c='blue', marker='o', label='data')

        # show result
        circle = plt.Circle((point1, point2), radius=rad1, color='r', fc='y', fill=False)
        plt.gca().add_patch(circle)

        # show result
        circle = plt.Circle((point3, point4), radius=rad2, color='r', fc='y', fill=False)
        plt.gca().add_patch(circle)

        plt.figure()
        outer_points = np.array(self.final_outer_points)
        inner_points = np.array(self.final_inner_points)
        # tail_inner_points = np.array(self.final_tail_points)

        plt.imshow(Obj.resized_img)
        plt.plot(point1, point2)
        plt.plot(point3, point4)
        plt.plot(outer_points[:, 0], outer_points[:, 1], 'bo')
        plt.plot(inner_points[:, 0], inner_points[:, 1], 'ro')
        # Tail position
        # plt.plot(tail_inner_points[:, 0], tail_inner_points[:, 1], 'ro')

        plt.axis('scaled')
        plt.show()

    def outlier_remover(self):
        '''
        Need to set the threshold distance so that any point(s) out of that range should be gone
        :return: A new list of inner and outer points after running the RANSAC.
        '''
        # Call the execute function
        self.execute_ransac('outer')
        point1, point2, rad1 = self.best_model_outer

        self.execute_ransac('inner')
        point3, point4, rad2 = self.best_model_inner

        test_list = np.array(self.outer_points)
        final_outer_points = []
        # Delete x2 outliers if the points are 5-10 increments outside radius(r) in the x-axis.
        for point in test_list:
            dist = np.hypot(point[0] - point1, point[1] - point2)
            # above 3% of the radius and below 5% of original radius is the percentage offset.
            if dist < 1.03 * rad1 and dist > 0.95 * rad1:
                final_outer_points.append(point)
        self.final_outer_points = final_outer_points

        test_list = np.array(self.inner_points)
        final_inner_points = []
        # Delete x2 outliers if the points are 5-10 increments outside radius(r) in the x-axis.
        for point in test_list:
            dist = np.hypot(point[0] - point3, point[1] - point4)
            # 0.05 is the percentage offset.
            if dist < 1.1 * rad2 and dist > 0.9 * rad2:
                final_inner_points.append(point)
        self.final_inner_points = final_inner_points

        # '''
        # Tail position.
        # '''
        # # tail position
        # test_list = np.array(self.inner_points)
        # final_tail_points = []
        # for point in test_list:
        #     dist = np.hypot(point[0] - point3, point[1] - point4)
        #     if dist > rad1:
        #         final_tail_points.append(point)
        #
        # self.final_tail_points = final_tail_points

        self.plot_points(point1,point2,rad1,point3,point4,rad2)

        return self.final_inner_points, self.final_outer_points

    def point_connector(self, final_inner, final_outer):
        '''
        Connects the points of the inlier and the outlier points after the ransac has finished running. The inlier
        points and outlier points should the newly updated lists from outlier_remover()

        Note the graph looks inverted when printed.
        '''
        # printing the final result of the inner points after ransac algorithm
        x = [point[0] for point in final_inner]
        y = [point[1] for point in final_inner]

        # printing the final result of outer points after ransac algorithm
        x1 = [point[0] for point in final_outer]
        y1 = [point[1] for point in final_outer]

        plt.imshow(Obj.resized_img)
        plt.plot(x, y, c='g', linewidth=7.0)
        plt.plot(x1, y1, c= 'r', linewidth=7.0)
        plt.show()
        # print(type(self.inner_points))

Obj = Vision('new.jpg', 1)
x0, y0 = Obj.first_center_position()
print(x0, y0)

# Need to return data separately into x_data and y_data
inner_points, outer_points = Obj.inner_outer_points()

ransac = RANSAC(inner_points, outer_points, 50)
final_inner, final_outer = ransac.outlier_remover()

# Connecting all of the filtered inner and outer points of the coil, excluding the tail points.
# ransac.point_connector(final_inner, final_outer)

# ransac2 = RANSAC(np.array(ransac.final_inner_points), np.array(ransac.final_outer_points), 50)
# ransac2.outlier_remover()
