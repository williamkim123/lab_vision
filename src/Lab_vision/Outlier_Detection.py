'''
This is a class to detect outliers and the data onto a separate file.
# Tail/Outlier Detection
TODO: Need to draw the rays again and detect the defect of the circle.
'''
from Vision import Vision
from Ransac import RANSAC
import matplotlib.pyplot as plt
import numpy as np
import math
from bresenham import bresenham

class OUTILER_DEFECT(RANSAC):
    def __init__(self, Obj, ransac, final_outer_points):
        self.obj = Obj
        self.ransac = ransac
        self.final_defect_points = final_outer_points

    def circle_defect(self, final_outer_radius, final_outer_points):
        '''
        Functions to print our defect points:
        - Should return unfiltered points including the tail detection.
        '''

        print(self.outer_points, "These are the outer points")
        plt.figure('circle defection figure')
        plt.imshow(self.obj.resized_img)

        plt.plot(self.outer_points[:, 0], self.outer_points[:, 1], 'bo')
        plt.show()

        defect_points = []
        for point in self.outer_points:

            distance_point = np.sqrt((point[0]-self.x0)**2 + (point[1]-self.y0)**2)
            if distance_point > final_outer_radius * 1.05:
                defect_points.append(point)
        print("Defect points", defect_points)

        final_plot_points = np.array(defect_points)
        '''
        Finding the average of the points
        '''
        self.point_of_defect = np.array(final_plot_points).mean(axis=1)
        print(self.point_of_defect.shape)

        print("Defect Points", self.point_of_defect,)

        plt.figure("tail position")
        plt.imshow(self.obj.resized_img)
        plt.plot(final_plot_points[:, 0], final_plot_points[:, 1], 'bo')
        plt.show()

        return np.array(self.point_of_defect)

    def angle_between_points(self):
        '''
        Printing the angle between two points the average coordinate of where the defect(tail) is to that
        of the initial center of the coil.
        '''
        # Need to see where the center of mass compared to the image is
        print(type(self.final_defect_points))
        center_coord = np.array([self.obj.x0, self.obj.y0])
        print(center_coord)

        dist = np.hypot(self.final_defect_points[0] - self.obj.x0, self.final_defect_points[1] - self.obj.y0)
        deltaY = (self.final_defect_points[1] - self.obj.y0)**2
        deltaX = (self.final_defect_points[0] - self.obj.x0)
        # Finding the angle of defect
        print(dist)
        radian = math.atan2(deltaY, deltaX)
        degree = radian*(180/math.pi)
        print(f"The defect is approximately located at {round(degree,2)} degrees")

    def four_quadrants(self):
        '''
        Devide the whole image into four quadrant upper-right, upper-left, lower-right and lower-left.
        '''

        # height = self.height
        # width = self.width

        # get coordinates (x,y)
        xy_coords = np.flip(np.column_stack(np.where(self.obj.resized_img >= 0)), axis=1)
        print("These are the x,y coordinates of the image", xy_coords)

        # print(type(xy_coords))

        # upper_left = []
        # upper_right = []
        # lower_right = []
        # lower_left = []
        #
        # for i in xy_coords:
        #     # print("hulo" ,xy_coords[:, 0])
        #     if xy_coords[0] <
