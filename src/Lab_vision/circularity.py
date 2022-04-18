'''
This script is the final step to check for circularity of the object.
1) Takes the inner and the outer points that are drawn over the ransac and calculates the RANSAC.
'''
from Vision import Vision
from Ransac import RANSAC


class CIRCULARITY_MEASUREMENT(Vision):
    def __init__(self, Obj, ransac):
        self.obj = Obj
        self.ransac = ransac

    def line_drawing(self):
        '''
        Calling the bresenham's function and drawing a vertical line up and horizontal.
        '''
