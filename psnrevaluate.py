import cv2
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    seriesNums = ['007', '008', '009', '010', '037', '061', '073']
    my_psnrs = np.zeros(len(seriesNums))
    com_psnrs = np.zeros(len(seriesNums))
    counter = 0
    testDate = '0530'
    for series in seriesNums:

        ground_truth = cv2.imread('./groundtruth/512/' + series + '.hdr', -1)
        my_work = cv2.imread('./'+ testDate +'/' + series + 'p.hdr', -1)
        competitor = cv2.imread('./competitor/' + series + 'a.hdr', -1)
        mark_my = tf.image.psnr(ground_truth, my_work, max_val=1.0)
        mark_competitor = tf.image.psnr(ground_truth, competitor, max_val=1.0)

        print('original ' + series + ': ' + str(mark_competitor))
        print('my ' + series + ': ' + str(mark_my))

        com_psnrs[counter] = mark_competitor
        my_psnrs[counter] = mark_my
        counter += 1

    ave_my = np.average(my_psnrs)
    ave_com = np.average(com_psnrs)

    print('Average my: ' + str(ave_my))
    print('Average com: ' + str(ave_com))