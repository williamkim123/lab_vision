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
from Vision import Vision
from Ransac import RANSAC
from Outlier_Circle_Remover import Outlier_Detection
from Outlier_Detection import OUTILER_DEFECT
import numpy as np

# What is the range of angles you want to ignore?
# input_angle1, input_angle2 = input("Enter the range (two angles) to be excluded: ").split()
#if input_angle1 and input_angle2 == int:

final_inner_outlier =  []
final_outer_outlier = []
final_outlier_radius = []

for black_white_threshold in range(40, 100, 100):
    Obj = Vision('..\..\Images\\image_t2.jpg', 1, black_white_threshold, [[270, 360], [0, 90]])
    x0, y0 = Obj.first_center_position()
    # print(x0, y0)

    # Need to return data separately into x_data and y_data
    inner_points, outer_points = Obj.inner_outer_points()

    # Obj.file_creator()

    # Removing outliers purely
    OUTLIER = Outlier_Detection(Obj,inner_points, outer_points, x0, y0)
    final_inner_points, final_outer_points, final_outer_radius = OUTLIER.circle_detection()
    final_inner_outlier.append(final_inner_points)
    final_outer_outlier.append(final_outer_points)
    final_outlier_radius.append(final_outer_radius)

# averaging
final_outer_radius = np.mean(final_outlier_radius)

for i, item in enumerate(final_outer_outlier):
    if i == 0 :
        output = item
    else:
        output = np.vstack([output, item])

for i, item in enumerate(final_inner_outlier):
    if i == 0 :
        inner_output = item
    else:
        inner_output = np.vstack([inner_output, item])


final_circle_defect = OUTLIER.circle_defect(final_outer_radius, output)

# OUTLIER.angle_between_points(final_circle_defect)

ransac = RANSAC(Obj, OUTLIER, inner_output, output, 100)
final_inner, final_outer = ransac.outlier_remover()

# ransac.point_connector(final_inner, final_outer)


outlier_defect = OUTILER_DEFECT(Obj, ransac, final_circle_defect)
outlier_defect.angle_between_points()
outlier_defect.four_quadrants()



