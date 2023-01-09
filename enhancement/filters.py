import numpy as np
import cv2
from scipy import signal
import math


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

def gaussianKernel(sigma, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    k = int(kernel_size / 2)
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = np.exp(-((i ** 2 + j ** 2) / (2 * (sigma ** 2))))
            kernel[i + k, j + k] = x
    kernel = kernel / (np.sum(kernel))
    return kernel


def kernel_img(kernel):
    shape = kernel.shape
    result = np.zeros((shape[0] * 100, shape[1] * 100))
    for i in range(shape[0]):
        for j in range(shape[1]):
            result[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = kernel[i, j]
    return result



def laplacian(img):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    result = signal.convolve2d(img,kernel, mode='valid')
    return result

def ideal_highpass_filter(shape, diameter):
    high_pass = np.zeros(shape)
    d = diameter ** 2
    x0, y0 = int(shape[0] / 2), int(shape[1] / 2)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if ((i - x0) ** 2 + (j - y0) ** 2 > d):
                high_pass[i, j] = 1

    return high_pass

def apply_fourier_mul(img, kernel):
    img_fourier = np.fft.fft2(img, axes=(0, 1))
    img_fourier = np.fft.fftshift(img_fourier)
    r, g, b = img_fourier[:, :, 0] * kernel, img_fourier[:, :, 1] * kernel, img_fourier[:, :, 2] * kernel
    result = np.zeros(img_fourier.shape, dtype=np.complex)
    result[:, :, 0] = r
    result[:, :, 1] = g
    result[:, :, 2] = b
    return result

def apply_laplacian_fourier(img):
    img_fourier = np.fft.fft2(img, axes=(0, 1))
    img_fourier = np.fft.fftshift(img_fourier)
    width, height = img_fourier.shape[0], img_fourier.shape[1]
    result = np.zeros(img_fourier.shape, dtype=np.complex)
    for i in range(width):
        for j in range(height):
            coeff = 4 * math.pi * math.pi * ((i - int(width / 2)) ** 2 + (j - int(height / 2)) ** 2)
            result[i, j] = img_fourier[i, j] * coeff

    return result


