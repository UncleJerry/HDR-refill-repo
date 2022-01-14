import numpy as np
import cv2
from skimage import morphology

def SaturationDetection(img, threshold, seriesNum):
    saturatedMap = np.zeros(img.shape, dtype=np.bool)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if img[row, col] > threshold:
                saturatedMap[row, col] = 1

    invert_saturatedMap = np.invert(saturatedMap)
    invert_saturatedMap = morphology.remove_small_objects(invert_saturatedMap, min_size=512, connectivity=1, in_place=False)
    saturatedMap = np.invert(invert_saturatedMap)
    saturatedMap = morphology.remove_small_objects(saturatedMap, min_size=128, connectivity=1, in_place=False)
    np.save('saturatedMap' + seriesNum + '.npy', saturatedMap)

if __name__ == "__main__":
    seriesNum = "002"

    img = cv2.imread('adjusted' + seriesNum + '2a.png', cv2.IMREAD_GRAYSCALE)
    SaturationDetection(img, 247, seriesNum)