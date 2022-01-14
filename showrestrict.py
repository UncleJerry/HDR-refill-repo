import numpy as np
import cv2
from scipy import io
from scipy import ndimage
import math
import json

if __name__ == "__main__":

    seriesNum = '007'
    testDate = '0523'
    cropNum = 1

    constraint_rgb = cv2.imread('./'+ testDate +'/pieces/const' + seriesNum + 'c' + str(cropNum) + '.png')
    constraint = cv2.imread('./'+ testDate +'/pieces/const' + seriesNum + 'c' + str(cropNum) + '.png', cv2.IMREAD_GRAYSCALE)
    

    for row in range(constraint.shape[0]):
        for col in range(constraint.shape[1]):
            if constraint[row, col] == 4:
                constraint_rgb[row, col] = [255, 0, 0]
            
            if constraint[row, col] == 2:
                constraint_rgb[row, col] = [0, 255, 0]

            if constraint[row, col] == 1:
                constraint_rgb[row, col] = [0, 0, 255]


    cv2.imshow("c", constraint_rgb)
    cv2.waitKey(0)