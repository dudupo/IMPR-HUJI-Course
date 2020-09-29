
from imageio import imread, imwrite
from skimage.color import rgb2gray

import os, random
import numpy as np
from skimage.draw import line


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





def relpath(path):
    """Returns the relative path to the script's location

    Arguments:
    path -- a string representation of a path.
    """
    return os.path.join(os.getcwd(), path)


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """

    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']

    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images


def images_for_denoising():
    """Returns a list of image paths to be used for image denoising in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


def images_for_deblurring():
    """Returns a list of image paths to be used for text deblurring in Ex5"""
    return list_images(relpath("current/text_dataset/train"), True)


# For those who wish to experiment...
def images_for_super_resolution():
    """Returns a list of image paths to be used for image super-resolution in Ex5"""
    return list_images(relpath("current/image_dataset/train"), True)


# def motion_blur_kernel(kernel_size, angle):
#     """Returns a 2D image kernel for motion blur effect.

#     Arguments:
#     kernel_size -- the height and width of the kernel. Controls strength of blur.
#     angle -- angle in the range [0, np.pi) for the direction of the motion.
#     """
#     if kernel_size % 2 == 0:
#         raise ValueError('kernel_size must be an odd number!')
#     if angle < 0 or angle > np.pi:
#         raise ValueError('angle must be between 0 (including) and pi (not including)')
#     norm_angle = 2.0 * angle / np.pi
#     if norm_angle > 1:
#         norm_angle = 1 - norm_angle
#     half_size = kernel_size // 2
#     if abs(norm_angle) == 1:
#         p1 = (half_size, 0)
#         p2 = (half_size, kernel_size - 1)
#     else:
#         alpha = np.tan(np.pi * 0.5 * norm_angle)
#         if abs(norm_angle) <= 0.5:
#             p1 = (2 * half_size, half_size - int(round(alpha * half_size)))
#             p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
#         else:
#             alpha = np.tan(np.pi * 0.5 * (1 - norm_angle))
#             p1 = (half_size - int(round(alpha * half_size)), 2 * half_size)
#             p2 = (kernel_size - 1 - p1[0], kernel_size - 1 - p1[1])
#     rr, cc = line(p1[0], p1[1], p2[0], p2[1])
#     kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
#     kernel[rr, cc] = 1.0
#     kernel /= kernel.sum()
#     return kernel
