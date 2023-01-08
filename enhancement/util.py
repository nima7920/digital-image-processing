import numpy as np
import cv2


def convert_from_float32_to_uint8(img):
    max_index, min_index = np.max(img), np.min(img)
    a = 255 / (max_index - min_index)
    b = 255 - a * max_index
    result = (a * img + b).astype(np.uint8)
    return result
