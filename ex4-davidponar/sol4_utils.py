from scipy.signal import convolve2d
import numpy as np
from imageio import imread, imwrite
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve as imconv


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img



REPRESENTATION_RGB = 2
REPRESENTATION_GRY = 1

PIXSIZE = 255

MAP_RGB_TO_YIQ = np.matrix([
        [   0.299,  0.587,  0.114   ],
        [   0.596, -0.276, -0.321   ],
        [   0.212, -0.523,  0.311   ]
])

MAP_YIQ_TO_RGB = MAP_RGB_TO_YIQ.I

def read_image(filename, representation):
    image = imread(filename)
    return {
        REPRESENTATION_RGB : image,
        REPRESENTATION_GRY : rgb2gray(image)
     }.get(representation).astype(np.float64) / 256


IM_SIZE_TRESHOLD = 4
def gaussian_filter(filter_size):
    ret = imconv( np.ones(filter_size), np.ones(filter_size), mode="wrap")
    ret.shape = ( 1, filter_size) 
    return ret / sum(ret[0])

def build_pyramid(im, max_levels, _filter, ret_list = [ ]):
    if max_levels == 0 or len(im) <= IM_SIZE_TRESHOLD:
        return ret_list , _filter
    ret_list.append(im)
    blured_image = imconv( imconv(im, _filter), _filter.transpose())
    reduced_image = blured_image[::2, ::2]
    return build_pyramid(reduced_image, max_levels -1, _filter, ret_list )

def build_gaussian_pyramid(im, max_levels, filter_size, ret_list = [] ):
    return build_pyramid( im, max_levels, gaussian_filter(filter_size), [])