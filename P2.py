import numpy as np
import cv2
from scipy import ndimage
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

def MarkLabel(labels, label_index):
    label_map = np.zeros_like(labels, dtype=np.uint8)
    h = np.size(labels, 0)
    w = np.size(labels, 1)

    for i in range(h):
        for j in range(w):
            if labels[i, j] == label_index:
                label_map[i, j] = 255
    
    return label_map

if __name__ == "__main__":

    seriesNum = '010'
    testDate = '0107'
    img = cv2.imread('./Scenes/' + seriesNum + 'a/input_1_aligned.tif')
    saturatedMap = np.load('./Experimental\ Data/Saturatation\ Maps/saturatedMap' + seriesNum + '.npy')
    flow = get_flo('./Experimental\ Data/flow/flow-' + seriesNum + 'a.flo')

    flow = cv2.resize(flow, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    mask = cv2.imread('./'+ testDate +'/mask' + seriesNum + '.png', cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    cv2.imwrite('./'+ testDate +'/mask' + seriesNum + '_dilated.png', mask)


    #ref = cv2.imread('ref' + seriesNum + '.jpg')

    #img[mask!=0] = np.array([0, 0, 0], np.uint8)
    connection = [[0,1,0], [1,1,1], [0,1,0]]
    
    labels, num_labels = ndimage.measurements.label(mask, connection)
    
    for label in range(1, num_labels+1):
        label_map = MarkLabel(labels, label)
        bounder = GenerateBoundingBox(label_map, 150)
        
        for index in range(4):
            bounder[index] = np.clip(bounder[index],0, max(img.shape))
        
        label_map = label_map[bounder[1]:bounder[3], bounder[0]:bounder[2]]
        img_piece = img[bounder[1]:bounder[3], bounder[0]:bounder[2], :]
        flow_piece = flow[bounder[1]:bounder[3], bounder[0]:bounder[2], :]
        gray_piece = cv2.cvtColor(img_piece, cv2.COLOR_BGR2GRAY)
        con_piece = saturatedMap[bounder[1]:bounder[3], bounder[0]:bounder[2]]
        con_piece = np.logical_and(con_piece, np.logical_not(np.logical_and(np.logical_not(gray_piece), np.logical_not(label_map))))
        gradient = 0.0
        label_map_bool = label_map / 255
        mass_of_center = ndimage.measurements.center_of_mass(label_map_bool.astype(dtype=np.bool))
        spatial_map = np.zeros_like(con_piece)
        const_map = np.zeros_like(con_piece, dtype=np.uint8)
        p = 0.0
        h = 0.0

        # Summing up motionvector
        for row in range(label_map.shape[0]):
            for col in range(label_map.shape[1]):
                if label_map[row, col] == 255:
                    h += flow_piece[row, col, 0]
                    p += flow_piece[row, col, 1]
        gradient = p / h
        
        target_g = -1 / gradient
        
        b = mass_of_center[0] - target_g * mass_of_center[1]

        for row in range(con_piece.shape[0]):
            for col in range(con_piece.shape[1]):
                y = target_g*col + b 
                shift = np.sqrt(flow_piece[row, col, 0] ** 2 + flow_piece[row, col, 1] ** 2)
                if p > 0 and h > 0 and y > row - 40:
                    spatial_map[row, col] = True
                    if con_piece[row, col]:
                        const_map[row, col] = 4
                    else:
                        const_map[row, col] = 2
                if p < 0 and h > 0 and y < row + 40:
                    spatial_map[row, col] = True
                    if con_piece[row, col]:
                        const_map[row, col] = 4
                    else:
                        const_map[row, col] = 2
                if p > 0 and h < 0 and y > row - 40:
                    spatial_map[row, col] = True
                    if con_piece[row, col]:
                        const_map[row, col] = 4
                    else:
                        const_map[row, col] = 2
                if p < 0 and h < 0 and y < row + 40:
                    spatial_map[row, col] = True
                    if con_piece[row, col]:
                        const_map[row, col] = 4
                    else:
                        const_map[row, col] = 2

                if con_piece[row, col] and not spatial_map[row, col]:
                    const_map[row, col] = 1


        con_piece = np.logical_or(con_piece, label_map)
        cv2.imwrite('./'+ testDate +'/pieces/example' + seriesNum + 'c' + str(label) + '.png', img_piece)
        cv2.imwrite('./'+ testDate +'/pieces/mask' + seriesNum + 'c' + str(label) + '.png', label_map)
        cv2.imwrite('./'+ testDate +'/pieces/const' + seriesNum + 'c' + str(label) + '.png', const_map)
        #cv2.imwrite('./'+ testDate +'/pieces/ref' + seriesNum + 'c' + str(label) + '.jpg', refcop)

        bounder_json = {
            'seriesNum' : seriesNum,
            'leftBounder' : bounder[0],
            'topBounder' : bounder[1],
            'rightBounder' : bounder[2],
            'bottomBounder' : bounder[3],
        }

        json_str = json.dumps(bounder_json, cls=NpEncoder)
        f = open('./'+ testDate +'/pieces/bounder' + seriesNum + 'c' + str(label) + '.json', "w")
        f.write(json_str)
        f.close()