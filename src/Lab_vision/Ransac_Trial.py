'''
Prestep:
1) Need to filter out the points inbetween the inner and the outer coil
using the concept of filtering the points by amplitude by the distance.

Steps:
1) Pick two random points (x1, y1)-> (h,k), (x2,y2)
2) Set your h and k values to be the center of the circle
3) Set the r to be the distance of the random points from 1) to the center h, k value
4) Use the (x-h)^2 + (y-k)^2 = r^2 to draw the circle
5) Put all the points in the following equation to find distance of each
point with it d = (x-h)^2 + (y-k)^2 - r^2
6) Set a specific threshold for distance (5-95% outer to inner proximity)
7) Count all the points whose distance is under the threshold
8) Voting, score of the random circle.
9) Repeat the process from step 1, iteration 80-100 times
10) After repetitions find/return the circle whose voting score is largest.
Display it's h,k,r
'''

from Vision import Vision
import matplotlib.pyplot as plt
import numpy as np

class RANSAC_TRIALS(Vision):
    '''
    For now RANSAC_TRIALS is only for the outer points.
    '''

    def __init__(self, Obj,innerpoints, outerpoints: list, x0, y0, n_iteration: int):
        self.outer_points = outerpoints
        self.inner_points = innerpoints
        self.n = n_iteration
        self.x0 = x0
        self.y0 = y0
        self.obj = Obj

    def circle_detection(self):
        # Calculating the distance to the coordinates of the outer points
        outer_points = np.array(self.outer_points)
        inner_points = np.array(self.inner_points)

        final_outer_radius, final_outer_points = self.clean_points(outer_points)
        final_inner_radius, final_inner_points = self.clean_points(inner_points)


        plt.figure()
        plt.imshow(self.obj.resized_img)
        circle = plt.Circle((self.x0, self.y0), radius=final_outer_radius, color='r', fc='y', fill=False)
        circle1 = plt.Circle((self.x0, self.y0), radius=final_inner_radius, color='r', fc='y', fill=False)
        plt.gca().add_patch(circle)

        plt.gca().add_patch(circle1)

        plt.plot(final_outer_points[:, 0], final_outer_points[:, 1], 'bo')
        plt.plot(final_inner_points[:, 0], final_inner_points[:, 1], 'bo')
        plt.show()


    def clean_points(self, points):
        # This for outer points
        # TODO need to reduce the computational road on this.
        distance_radius = []
        for point in points:
            distance_radius.append(np.hypot(point[0] - self.x0, point[1] - self.y0))

        # guess radius should never exceed self.height//2
        guess_radius = []
        guess_radius.append(0)
        scores = []
        while guess_radius[-1] <= self.obj.height//2:

            scores.append(np.where(np.logical_and(np.array(distance_radius) > guess_radius[-1] -20 , np.array(distance_radius) < guess_radius[-1] + 20))[0].shape[0])
            guess_radius.append(guess_radius[-1] + 1)

        # final_radius = guess_radius[np.argmax(scores)]
        final_radius = np.array(guess_radius)[np.where(np.array(scores) == np.max(scores))[0]].mean()

        final_points = points[np.where(np.logical_and(np.array(distance_radius) > final_radius -20, np.array(distance_radius) < final_radius + 20))[0]]

        return final_radius, final_points







