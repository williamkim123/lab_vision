'''
Explanation of the calculation can be found here. https://darkpgmr.tistory.com/60

True Ransac algorithm to run the points
'''

from Vision import Vision
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

from Outlier_Circle_Remover import Outlier_Detection

class RANSAC(Vision):
    '''
    Inspired from: https://github.com/SeongHyunBae
    '''

    def __init__(self, Obj, OUTLIER, innerpoints, outerpoints, n_iteration):
        # trial run
        self.inner_points = innerpoints
        self.outer_points = outerpoints
        self.obj = Obj
        self.obj1 = OUTLIER

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

        # print("length of sample data going into RANSAC", len(sample_dataX))

        # getting three data points from the sample
        # TODO: Need to make a conditional that makes sure the three random points are not the same.
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

        # calculating A, B, C value from three points by using matrix
        '''
        https://darkpgmr.tistory.com/60, explanation for the calculation for the code
        '''
        a = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]], [pt3[0] - pt2[0], pt3[1] - pt2[1]]])
        b = np.array([[pt2[0] ** 2 - pt1[0] ** 2 + pt2[1] ** 2 - pt1[1] ** 2],
                      [pt3[0] ** 2 - pt2[0] ** 2 + pt3[1] ** 2 - pt2[1] ** 2]])
        # Catch the error if two or more points are the same or the triangle is very small for it to take the inverse of the three random points.
        try:
            inv_A = inv(a)
            c_x, c_y = np.dot(inv_A, b) / 2
            c_x, c_y = c_x[0], c_y[0]
            r = np.sqrt((c_x - pt1[0]) ** 2 + (c_y - pt1[1]) ** 2)
            return c_x, c_y, r
        except:
            return None


    def evaluate_model(self, model, sample):
        d = 0
        c_x, c_y, r = model

        # print(c_y, c_x, r)
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
            if model is not None:
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

        # plt.scatter(self.x1_inner, self.y1_inner, c='red', marker='o', label='data')
        # plt.scatter(self.x2_outer, self.y2_outer, c='blue', marker='o', label='data')

        plt.figure("Final Diagram for RANSAC!")
        # show result
        circle = plt.Circle((point1, point2), radius=rad1, color='r', fc='y', fill=False)
        plt.gca().add_patch(circle)

        # show result
        circle = plt.Circle((point3, point4), radius=rad2, color='r', fc='y', fill=False)
        plt.gca().add_patch(circle)


        outer_points = np.array(self.final_outer_points)
        inner_points = np.array(self.final_inner_points)
        # tail_inner_points = np.array(self.final_tail_points)

        plt.imshow(self.obj.resized_img)
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
            if dist < 1.05 * rad2 and dist > 0.98 * rad2:
                final_inner_points.append(point)
        self.final_inner_points = final_inner_points

        self.plot_points(point1,point2,rad1,point3,point4,rad2)

        return self.final_inner_points, self.final_outer_points

    def point_connector(self, final_inner, final_outer):
        '''
        Connects the points of the inlier and the outlier points after the ransac has finished running. The inlier
        points and outlier points should the newly updated lists from outlier_remover()
        '''
        # printing the final result of the inner points after ransac algorithm
        x = [point[0] for point in final_inner]
        y = [point[1] for point in final_inner]

        # printing the final result of outer points after ransac algorithm
        x1 = [point[0] for point in final_outer]
        y1 = [point[1] for point in final_outer]

        plt.imshow(self.obj.resized_img)
        plt.plot(x, y, c='g', linewidth=7.0)
        plt.plot(x1, y1, c= 'r', linewidth=7.0)
        plt.show()

    def change_of_circle(self):
        '''
        After final circle is drawn
        '''