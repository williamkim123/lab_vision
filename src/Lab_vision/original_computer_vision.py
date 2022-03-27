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

# What is the range of angles you want to ignore?
# input_angle1, input_angle2 = input("Enter the range (two angles) to be excluded: ").split()
#if input_angle1 and input_angle2 == int:

Obj = Vision('..\..\Images\\new.jpg', 1)
x0, y0 = Obj.first_center_position()
# print(x0, y0)

# Need to return data separately into x_data and y_data
inner_points, outer_points = Obj.inner_outer_points()

# Obj.file_creator()

# Removing outliers purely
OUTLIER = Outlier_Detection(Obj,inner_points, outer_points, x0, y0)
final_inner_points, final_outer_points, final_outer_radius = OUTLIER.circle_detection()
OUTLIER.circle_defection(final_outer_radius, final_outer_points)

ransac = RANSAC(Obj, OUTLIER, final_inner_points, final_outer_points, 100)
final_inner, final_outer = ransac.outlier_remover()



