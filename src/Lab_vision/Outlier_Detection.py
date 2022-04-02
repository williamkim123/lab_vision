'''
This is a class to detect outliers and the data onto a separate file.
'''
from Vision import Vision
from Ransac import RANSAC

import numpy as np
import math

class OUTILER_DEFECT(RANSAC):
    def __init__(self, Obj, ransac, final_defect_points):
        self.obj = Obj
        self.ransac = ransac
        self.final_defect_points = final_defect_points

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
