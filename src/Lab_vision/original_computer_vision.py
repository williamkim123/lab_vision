'''
Main script to run the pipeline
'''
# Computer Vision libary
from Vision import Vision
from Ransac import RANSAC
from Outlier_Circle_Remover import Outlier_Detection
from Outlier_Detection import OUTILER_DEFECT
from circularity import CIRCULARITY_MEASUREMENT
import numpy as np

# To scan all the images in the folder
import os
import cv2
import glob

# Go within the directory of images and start analyzing all the images
def analyze_file_images():
    '''
    Import the function image_analysis above and return the final image for measurement
    - Import a list of images to image_analysis
    return: Final image that needs to be analyzed
    '''
    path = glob.glob("..\..\\video_analyzer\\data\\*.jpg")

    list_of_files = [] # make a list to store all the files intensities of each frame.
    for file in path:
        # Reading the grayscale image(s) of all the files in the directory reading the image
        img_gray = cv2.imread(file,0)
        length = os.stat(file).st_size
        print(f"{img_gray} has been uploaded with length {length} bytes")

        # Print the intensity value at the harcoded x and y_coordinates intensity value from 0-255
        intensity_yx = img_gray[180][380]
        print(intensity_yx, "This is the intensity at 380, 180")

        # Set a threshold for the intensity
        # TODO: Need to look into planning the right method for this algorithm
        if intensity_yx > 20:
            list_of_files.append(file)

    print(list_of_files)
    return list_of_files

# Input img
def vision_system():
    # Making a list to find the average points of all the inner and outer points when running the outlier/detection script(s)
    final_inner_outlier = []
    final_outer_outlier = []
    final_outlier_radius = []

    # Running a for loop that sets the threshold for intensity from  40 to 100 of the image, specifically for RGB and grayscale images.
    for black_white_threshold in range(40, 100, 100):
        '''
        Arguments for the main class vision is the image source, angle between each rays, range for threshold of the image - not needed for black and white,
        Angle of interest for the rays.
        '''
        # Obj = Vision(img_file, 1, black_white_threshold, [[270, 390], [0, 100]])
        Obj = Vision('..\..\Images\\frame2155.jpg', 1, black_white_threshold, [[0, 181], [0, 0]])
        x0 = Obj.x0
        y0 = Obj.y0
        # print(x0, y0)

        # Need to return data separately into x_data and y_data
        inner_points, outer_points = Obj.inner_outer_points()

        # Creating excel file with intensity of inner and outer circle
        # Obj.file_creator()

        # Removing outliers purely
        OUTLIER = Outlier_Detection(Obj,inner_points, outer_points, x0, y0)
        final_inner_points, final_outer_points, final_outer_radius = OUTLIER.circle_detection()
        final_inner_outlier.append(final_inner_points)
        final_outer_outlier.append(final_outer_points)
        final_outlier_radius.append(final_outer_radius)

    # Taking the average of both final radius and outer/outlier points.
    final_outer_radius = np.mean(final_outlier_radius)

    # Concatenating all the points, sending the combined points to function to be analyzed.
    for i, item in enumerate(final_outer_outlier):
        if i == 0 :
            outer_points_output = item
        else:
            # Stack arrays in sequence vertically (row wise)
            outer_points_output = np.vstack([outer_points_output, item])

    for i, item in enumerate(final_inner_outlier):
        if i == 0 :
            inner_output = item
        else:
            inner_output = np.vstack([inner_output, item])

    # Getting the outlier points of the coil.
    final_circle_defect = OUTLIER.circle_defect(final_outer_radius, outer_points_output)

    # Setting the ransac iteration number to be 100
    ransac = RANSAC(Obj, OUTLIER, inner_output, outer_points_output, 100)
    # These are the parameters of the new ransac circle.
    final_inner, final_outer, new_center_outer, new_center_inner= ransac.outlier_remover()
    # ransac.point_connector(final_inner, final_outer)

    '''
    Class for circularity test
    '''
    circularity_measurement = CIRCULARITY_MEASUREMENT(Obj, final_inner, final_outer, new_center_outer, new_center_inner)
    circularity_measurement.line_drawing()

    # TODO: Need to find a better parameter to find the defect position.
    # output for detecting the defect.
    tail_defect = OUTILER_DEFECT(Obj, ransac, final_circle_defect)
    tail_defect.angle_between_points()
    # tail_defect.four_quadrants()

'''
Should return the final image for detection.
'''
if __name__ == '__main__':
    # files = analyze_file_images()
    # for file in files:
    #     vision_system(file)

    vision_system()