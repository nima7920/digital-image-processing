import numpy as np
import cv2
from enhancement import util

def apply_log_transform_on_hsv(hsv_img, alpha):
    value = np.float32(hsv_img[:, :, 2] / 255)
    value = (1 / np.log(alpha + 1)) * np.log(1 + alpha * value)
    hsv_img[:, :, 2] = util.convert_from_float32_to_uint8(value)


def apply_gamma_transform_on_hsv(hsv_img, gamma):
    value = np.float32(hsv_img[:, :, 2] / 255)
    value = np.power(value, gamma)
    hsv_img[:, :, 2] = util.convert_from_float32_to_uint8(value)
