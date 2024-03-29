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

Obj = Vision('..\..\Images\\new.jpg', 1)
x0, y0 = Obj.first_center_position()
print(x0, y0)

# Need to return data separately into x_data and y_data
inner_points, outer_points = Obj.inner_outer_points()

ransac = RANSAC(Obj, inner_points, outer_points, 50)
final_inner, final_outer = ransac.outlier_remover()

# Connecting all of the filtered inner and outer points of the coil, excluding the tail points.
# ransac.point_connector(final_inner, final_outer)

# ransac2 = RANSAC(np.array(ransac.final_inner_points), np.array(ransac.final_outer_points), 50)
# ransac2.outlier_remover()
