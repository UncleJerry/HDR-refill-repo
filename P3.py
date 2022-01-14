import numpy as np
import cv2
import math
import json


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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def GenerateBoundingBox(img, bounder_width):
    h = np.size(img, 0)
    w = np.size(img, 1)

    bounder_left = w
    bounder_right = 0
    bounder_top = h
    bounder_bottom = 0

    for i in range(h):
        for j in range(w):
            
            if img[i, j] == 255:
                bounder_top = min(bounder_top, i - bounder_width)
                bounder_bottom = max(bounder_bottom, i + bounder_width)
                bounder_left = min(bounder_left, j - bounder_width)
                bounder_right = max(bounder_right, j + bounder_width)

    bounder = np.array([bounder_left, bounder_top, bounder_right, bounder_bottom], dtype=np.int32)
    return bounder

def rgb2gray(rgb):
    b, g, r = rgb[0], rgb[1], rgb[2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return gray

def getAverageEv(img, mask=None, general=False):

    counter = 0
    sumOfLuminance = 0
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if (mask is None and general) or mask[row, col] != 0:
                sumOfLuminance += rgb2gray(img[row, col])
                counter += 1

        
    sumOfLuminance /= counter
    return sumOfLuminance

def similar(refilled, ori):
    #print("Refilled: " + str(math.log2(refilled)))
    #print("Ori: " + str(math.log2(ori)))
    return math.log2(refilled) < math.log2(ori) * 1.0747

def pixelDifference(lhs, rhs):
    dif_r = lhs[2] - rhs[2]
    dif_g = lhs[1] - rhs[1]
    dif_b = lhs[0] - rhs[0]

    differenceSum = math.pow(dif_r, 2) + math.pow(dif_g, 2) + math.pow(dif_b, 2);
    sigma = 50;
    numerator = math.sqrt(differenceSum) / math.sqrt(3);

    return numerator #math.exp(-(numerator / (2 * math.pow(sigma, 2))));

def getAverageDifference(refilled, ori, mask=None, general=False):

    counter = 0
    sumOfDifference = 0
    for row in range(refilled.shape[0]):
        for col in range(refilled.shape[1]):
            if (mask is None and general) or mask[row, col] != 0:
                lhs = refilled[row, col].astype(int)
                rhs = ori[row, col].astype(int)
                sumOfDifference += pixelDifference(lhs, rhs)
                counter += 1

        
    sumOfDifference /= counter
    return sumOfDifference

def isBoundary(mask, row, col):
    if mask[row, col] != 0:
        return False
    
    distance = 21
    if mask[row, col + distance] != 0 or mask[row, col - distance] != 0 or mask[row + distance, col] != 0 or mask[row - distance, col] != 0:
        return True

    return False

def searchBackground(img, mask, flow):
    kernel = np.ones((7, 7), np.float32) / 49
    distance = 10
    averaged = cv2.filter2D(img,-1,kernel)
    #height, width, _ = 
    markMap = np.zeros(img.shape[0: 2])
    patches = []

    for row in range(distance * 2 + 1, img.shape[0] - (distance * 2 + 1)):
        for col in range(distance * 2 + 1, img.shape[1] - (distance * 2 + 1)):
            if markMap[row, col]:
                continue

            shift = np.sqrt(flow[row, col, 0] ** 2 + flow[row, col, 1] ** 2)
            if shift < 15 and isBoundary(mask, row, col):
                
                patch = averaged[row - distance: row + distance + 1, col - distance: col + distance + 1]
                markMap[row - distance: row + distance + 1, col - distance: col + distance + 1] = True

                patches.append(patch)
    return patches

def refillAgain(img, patches):
    height, width = img.shape[0: 2]
    maxEV = -999.9
    maxIndex = 0
    distance = 21

    for index in range(len(patches)):
        thisEV = getAverageEv(patches[index], general=True)
        if thisEV > maxEV:
            maxEV = thisEV
            maxIndex = index


    tile = np.tile(patches[maxIndex], (math.ceil(height / (distance)), math.ceil(width / (distance),), 1))
    tile = cv2.resize(tile, (width, height), interpolation=cv2.INTER_CUBIC)

    return tile



if __name__ == "__main__":

    seriesNum = '010'
    testDate = '0107'
    crops = 3
    print("Series: " + seriesNum)
    for cropNum in range(1, crops + 1):
        #if cropNum == 5:
        #    continue

        ori = cv2.imread('./'+ testDate +'/pieces/example' + seriesNum + 'c' + str(cropNum) + '.png')
        refilled = cv2.imread('./'+ testDate +'/pieces/result' + seriesNum + 'c' + str(cropNum) + '.png')
        mask = cv2.imread('./'+ testDate +'/pieces/mask' + seriesNum + 'c' + str(cropNum) + '.png', cv2.IMREAD_GRAYSCALE)
        fullsize = cv2.imread('./Scenes/' + seriesNum + 'a/input_1_aligned.tif')
        flow = get_flo('./Experimental\ Data/flow/flow-' + seriesNum + 'a.flo')
        f = open('./'+ testDate +'/pieces/bounder' + seriesNum + 'c' + str(cropNum) + '.json', 'r')
        
        b_json = json.loads(f.read())
        flow = cv2.resize(flow, (fullsize.shape[1], fullsize.shape[0]), interpolation=cv2.INTER_CUBIC)
        bounder = [b_json['leftBounder'], b_json['topBounder'], b_json['rightBounder'], b_json['bottomBounder']]
        flow = flow[bounder[1]:bounder[3], bounder[0]:bounder[2], :]

        patches = searchBackground(refilled, mask, flow)
        tile = refillAgain(refilled, patches)


        #similarMap = scanSimilarPatch(refilled, ori, mask)

        #refilled[similarMap!=0] = tile[similarMap!=0]
        #cv2.imwrite('./'+ testDate +'/pieces/result' + seriesNum + 'c' + str(cropNum) + '_p3.png', refilled)

        refilledEv = getAverageEv(refilled,mask=mask)
        oriEv = getAverageEv(ori, mask=mask)

        if similar(refilledEv, oriEv) and getAverageDifference(refilled, tile, mask=mask) > 100 or cropNum == 5:
            print('crop: ' + str(cropNum))
            refilled[mask!=0] = tile[mask!=0]
            cv2.imwrite('./'+ testDate +'/pieces/result' + seriesNum + 'c' + str(cropNum) + '.png', refilled)