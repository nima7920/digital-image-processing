import numpy as np


def change_color(hsv_img, input_color_range, output_color):
    h = hsv_img[:, :, 0]
    ''' taking the position of all pixels having the given color '''
    color_mask = np.logical_and(h >= input_color_range[0], h <= input_color_range[1])
    ''' changing color '''
    h[color_mask] = output_color
    ''' returning positions '''
    return color_mask
