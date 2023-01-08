import numpy as np
import cv2


def specify_histogram_on_hsv(hsv_img, cum_dist1, cum_dist2):
    ''' taking values '''
    values = np.copy(hsv_img[:, :, 2])
    ''' for each i , finding the best j to change the intensity of all pixels with intensity i to j'''
    for i in range(256):
        ''' position of pixels with intensity i '''
        pixels = np.where(values == i)
        if pixels[0].size > 0:
            ''' list of all points where the second cumulative distribution is smaller than the first one'''
            a = np.where(cum_dist2 <= cum_dist1[i])
            ''' changing the intensities '''
            hsv_img[pixels[0], pixels[1], 2] = a[0][-1]
