import numpy as np
import cv2
import matplotlib.pyplot as plt
from enhancement import colors, filters

'''  input and output paths  '''
input_path = 'data/enhancement/Flowers.jpg'
output_path = 'outputs/color_changer_res.jpg'

''' Defining Color ranges in HSV space '''
pink_range = np.array([130, 180])
yellow_color = 25

'''   Reading image from local drive  '''
img = cv2.imread(input_path)

'''  Converting image from BGR to HSV , in order to work on H channel  '''
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

'''  Changing pink colors to yellow  '''
color_mask = colors.change_color(hsv_img, pink_range, yellow_color)

'''  Applying filter  '''
filter_size = (31, 31)
result = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
result = filters.apply_excluded_boxFilter_lib(result, filter_size, color_mask)
cv2.imwrite(output_path,result)

