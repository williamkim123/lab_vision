'''
This script is the final step to check for circularity of the object.
1) Takes the inner and the outer points that are drawn over the ransac and calculates the RANSAC.
'''
from Vision import Vision
from Ransac import RANSAC
import matplotlib.pyplot as plt
import numpy as np


class CIRCULARITY_MEASUREMENT(Vision):
    def __init__(self, Obj, inner_points, outer_points, new_center_points_outer, new_center_points_inner):
        self.obj = Obj
        # self.ransac = ransac
        self.final_inner = np.array(inner_points)
        self.final_outer = np.array(outer_points)
        self.new_center_outer = new_center_points_outer
        self.new_center_inner = new_center_points_inner

        # ransac radius for the outer circle

    def line_drawing(self):
        '''
        Calling the bresenham's function and drawing a vertical line up and horizontal.
        '''
        # print("Hullo", self.final_inner, "Outer Ya", self.final_outer)


        # plt.figure("circularity data")
        # plt.imshow(self.obj.resized_img)
        # plt.plot(self.final_outer[:, 0], self.final_outer[:, 1], 'bo')
        # plt.plot(self.final_inner[:, 0], self.final_inner[:, 1], 'ro')
        # plt.show()

        # Taking the two points vertical, and horizontal plus at an angle for the inner(red) and outer (blue) points
        '''
        Getting the first point in the array and finding the distance of the point from the new center point found
        through the ransac.
        '''
        # # outer center coords
        # print(self.new_center_outer)
        # # inner center coords
        # print(self.new_center_inner)
        # # upper point
        # print(self.final_inner[0])

        right_horizontal = Vision(self.obj.image_source, 1, 40, [[0, 1], [0, 0]])
        inner_right, outer_right = right_horizontal.inner_outer_points()

        left_horizontal = Vision(self.obj.image_source, 1, 40, [[180, 180], [0, 0]])
        inner_left, outer_left = left_horizontal.inner_outer_points()

        print("these are the circularity points", inner_left, outer_left)