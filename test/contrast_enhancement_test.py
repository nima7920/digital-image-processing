import matplotlib.pyplot as plt
import numpy as np
import cv2
from enhancement import contrast_enhancement as ce

'''  input and output paths  '''
input_path = 'data/enhancement/Enhance1.jpg'
output_path = 'outputs/ce_result.jpg'

'''   Reading image from local drive  '''
img = cv2.imread(input_path)

'''    Converting Color space from BGR to RGB and then to HSV  '''
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

'''  Applying transforms on V channel , in order to increase contrast and intensity '''
ce.apply_gamma_transform_on_hsv(hsv_img, 0.6)
# ce.apply_log_transform_on_hsv(hsv_img, 15)
result = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
plt.imshow(result)
plt.show()

''' Converting image to BGR and Saving  '''
result = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
cv2.imwrite(output_path, result)
