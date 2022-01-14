import numpy as np
import cv2
import json
from skimage import morphology


def get_flo(file_name):
    with open(file_name, 'rb') as f:
        magic, = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w, h = np.fromfile(f, np.int32, count=2)
            # print(f'Reading {w} x {h} flo file')
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))
            return data2D

if __name__ == "__main__":

    seriesNum = "002"
    testDate = '0414'
    saturatedMap = np.load('./Experimental\ Data/Saturatation\ Maps/saturatedMap' + seriesNum + '.npy')
    example = cv2.imread('./Scenes/' + seriesNum + 'a/input_1_aligned.tif')

    flow = get_flo('./Experimental\ Data/flow/flow-' + seriesNum + 'a.flo')
    flow = cv2.resize(flow, (saturatedMap.shape[1], saturatedMap.shape[0]), interpolation=cv2.INTER_CUBIC)

    flow_r = get_flo('./Experimental\ Data/flow/flow-' + seriesNum + 'r.flo')
    flow_r = cv2.resize(flow_r, (saturatedMap.shape[1], saturatedMap.shape[0]), interpolation=cv2.INTER_CUBIC)

    f = open('./Experimental\ Data/flow/flow-' + seriesNum + 'a.json', 'r')

    flow_statistics = json.loads(f.read())
    
    oriSatMap = saturatedMap
    saturatedMap = saturatedMap.astype(np.uint8) * 255

    lostInfoMap = np.zeros_like(saturatedMap)

    # binary hole filling
    kernel = np.ones((5, 5), np.uint8)

    
    mask = np.zeros_like(example, dtype=np.bool)
    for row in range(saturatedMap.shape[0]):
        for col in range(saturatedMap.shape[1]):
            shift = np.sqrt(flow[row, col, 0] ** 2 + flow[row, col, 1] ** 2)
            shift_r = np.sqrt(flow_r[row, col, 0] ** 2 + flow_r[row, col, 1] ** 2)
            if shift > 20 and (saturatedMap[row, col] == 255 and shift_r < 20):
                mask[row, col] = [1, 1, 1]


    invert_mask = np.invert(mask)
    invert_mask = morphology.remove_small_objects(invert_mask, min_size=512, connectivity=1, in_place=False)
    mask = np.invert(invert_mask)
    mask = morphology.remove_small_objects(mask, min_size=1024, connectivity=1, in_place=False)
    cv2.imwrite('./'+ testDate +'/mask' + seriesNum + '.png', mask * 255)
    cv2.imwrite('./'+ testDate +'/example_' + seriesNum + 'o.jpg', example * mask)
    cv2.imwrite('./'+ testDate +'/const' + seriesNum + '.png', saturatedMap)