import numpy as np
import cv2
import matplotlib.pyplot as plt
from enhancement import histogram

''' input directories '''
input_path1 = 'data/enhancement/Dark.jpg'
input_path2 = 'data/enhancement/Pink.jpg'

''' output directories '''
output_path1 = 'outputs/histogram_res1.jpg'
output_path2 = 'outputs/histogram_res2.jpg'

'''  loading images '''
img1 = cv2.imread(input_path1)
img2 = cv2.imread(input_path2)

'''  converting images to hsv space '''
hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

'''  generating the normalized histogram of v channels of the images '''
h1, bins1 = np.histogram(hsv_img1[:, :, 2], bins=256, range=(0, 256), density=True)
h2, bins2 = np.histogram(hsv_img2[:, :, 2], bins=256, range=(0, 256), density=True)

'''  calculating the cumulative function of histograms '''
cum_dist1 = np.cumsum(h1) * (bins1[1] - bins1[0])
cum_dist2 = np.cumsum(h2) * (bins2[1] - bins2[0])

''' specifying the histograms based on their cumulative distribution '''
histogram.specify_histogram_on_hsv(hsv_img1, cum_dist1, cum_dist2)

''' histogram and cumulative distribution after applying histogram specification '''
h, bins = np.histogram(hsv_img1[:, :, 2], bins=256, range=(0, 256), density=True)
cum_result = np.cumsum(h) * (bins[1] - bins[0])
''' saving histogram '''
plt.plot(bins[1:],h)
plt.savefig(output_path1)

''' saving result image '''
img = cv2.cvtColor(hsv_img1, cv2.COLOR_HSV2BGR)
cv2.imwrite(output_path2, img)
