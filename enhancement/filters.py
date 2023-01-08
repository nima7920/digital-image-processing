import numpy as np
import cv2
from scipy import signal


def apply_excluded_boxFilter_lib(img, filter_size, exclude_mask):
    '''  copying pixels that are not going to be filtered '''
    excluded_pixels = img[exclude_mask].copy()
    '''  creating the box filter '''
    filter = np.ones(filter_size) / (filter_size[0] * filter_size[1])
    '''  applying filter on image '''
    result = cv2.filter2D(img, -1, filter)
    ''' changing back the pixels we have copied '''
    result[exclude_mask] = excluded_pixels
    return result


def convolve_colors(b, g, r):
    kernel = np.array([[-1, 1], [1, -1]])
    b2 = np.abs(signal.convolve(b, kernel))
    g2 = np.abs(signal.convolve(g, kernel))
    r2 = np.abs(signal.convolve(r, kernel))
    return [b2, g2, r2]
