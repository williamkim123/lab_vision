'''
1) Defection and outlier remover.
2) Tells the user where the outlier is.
'''

from Vision import Vision
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import math

class Outlier_Detection(Vision):
    '''
    For now RANSAC_TRIALS is only for the outer points.
    Need to make a separate class to get the data points in Excel file.
    '''

    def __init__(self, Obj,innerpoints, outerpoints: list, x0, y0):
        self.outer_points = outerpoints
        self.inner_points = innerpoints
        self.x0 = x0
        self.y0 = y0
        self.obj = Obj


    def circle_detection(self):

        # Adding points outside of the circle for the outer points
        # TODO: Look into this
        points_outside = []

        # Calculating the distance to the coordinates of the outer points
        outer_points = np.array(self.outer_points)
        inner_points = np.array(self.inner_points)

        final_outer_radius, final_outer_points = self.clean_points(outer_points)
        final_inner_radius, final_inner_points = self.clean_points(inner_points)


        plt.figure("Diagram of Outlier points from Class Outlier_Detection")
        plt.imshow(self.obj.resized_img)
        circle = plt.Circle((self.x0, self.y0), radius=final_outer_radius, color='r', fc='y', fill=False)
        circle1 = plt.Circle((self.x0, self.y0), radius=final_inner_radius, color='r', fc='y', fill=False)

        plt.gca().add_patch(circle)
        plt.gca().add_patch(circle1)

        plt.plot(final_outer_points[:, 0], final_outer_points[:, 1], 'bo')
        plt.plot(final_inner_points[:, 0], final_inner_points[:, 1], 'ro')
        plt.show()

        # TODO: Need to return the new center of the circle of the outer points as well

        return final_inner_points, final_outer_points, final_outer_radius

    def clean_points(self, points):

        distance_radius = []
        for point in points:
            distance_radius.append(np.hypot(point[0] - self.x0, point[1] - self.y0))

        # guess radius should never exceed self.height//2
        guess_radius = []
        # Drawing the line from 0 all the way up
        guess_radius.append(0)
        scores = []

        while guess_radius[-1] <= self.obj.height // 2:
            # Taking out all the pixels 20 down and 20 up from the distance of the radius
            # scores.append(np.where(np.logical_and(np.array(distance_radius) > guess_radius[-1] -20 , np.array(distance_radius) < guess_radius[-1] + 22))[0].shape[0])
            scores.append(np.where(np.logical_and(np.array(distance_radius) > guess_radius[-1] * 0.95, np.array(distance_radius) < guess_radius[-1] * 1.1))[0].shape[0])
            guess_radius.append(guess_radius[-1] + 1)

        # final_radius = guess_radius[np.argmax(scores)]
        # TODO: Is this final radius good enough.
        #Taking the average radius of the scores as there are some numbers.
        final_radius = np.array(guess_radius)[np.where(np.array(scores) == np.max(scores))[0]].mean()

        #final_points = points[np.where(np.logical_and(np.array(distance_radius) > final_radius - 10, np.array(distance_radius) < final_radius + 20))[0]]
        final_points = points[np.where(np.logical_and(np.array(distance_radius) > final_radius * 0.95,np.array(distance_radius) < final_radius * 1.2))[0]]

        return final_radius, final_points

    '''
    *** After taking seval images should be able to compare differences between the coil through here ***
    '''
    def circle_defect(self, final_outer_radius, final_outer_points):

        # print(final_outer_radius)
        # print("These are the final outer points from outlier remover for outer points", final_outer_points)

        # 1) Finding all the points outside of the outer circle
        # 2) Categorizing them as defects, min 10 points with distance of each being < 10, using convex hull!
        # 3) Finding the center of mass of the defect points and using trig seeing where it is approximately to origin of point

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
        1) To do this draw the point on the image itself
        '''
        # self.center_of_mass = np.array(ndimage.center_of_mass(final_plot_points))
        # print(self.center_of_mass, "center of mass coord for tail")
        '''
        Finding the average of the points
        '''
        # TODO: Look into shortening this code
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
        print(type(self.point_of_defect))
        center_coord = np.array([self.x0, self.y0])
        print(center_coord)

        dist = np.hypot(self.point_of_defect[0] - self.x0, self.point_of_defect[1] - self.y0)
        deltaY = (self.point_of_defect[1] - self.y0)**2
        deltaX = (self.point_of_defect[0] - self.x0)
        # Finding the angle of defect
        print(dist)
        radian = math.atan2(deltaY, deltaX)
        degree = radian*(180/math.pi)
        print(f"The defect is approximately located at {round(degree,2)} degrees")






