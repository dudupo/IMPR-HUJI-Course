
import numpy as np
from scipy.ndimage.filters import convolve as imconv
from imageio import imread, imwrite
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from matplotlib import gridspec

from operator import __add__ , __sub__

IM_SIZE_TRESHOLD = 4
#mode='same'
def gaussian_filter(filter_size):
    ret = imconv( np.ones(filter_size), np.ones(filter_size), mode="wrap")


    ret.shape = ( 1, filter_size) 
    return ret / sum(ret[0])

def build_pyramid(im, max_levels, _filter, ret_list = [ ]):
    if max_levels == 0 or len(im) <= IM_SIZE_TRESHOLD:
        return ret_list
    ret_list.append(im)
    blured_image = imconv( imconv(im, _filter), _filter.transpose())
    reduced_image = blured_image[::2, ::2]
    # reduced_image.shape =  
    return build_pyramid(reduced_image, max_levels -1, _filter, ret_list )

def build_gaussian_pyramid(im, max_levels, filter_size, ret_list = [] ):
    return build_pyramid( im, max_levels, gaussian_filter(filter_size), [])

def expend (image, _filter):
        ret_image = np.zeros( (image.shape[0]*2, image.shape[1]*2) )
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ret_image[2*i][2*j] = image[i][j] 
        return imconv( imconv(ret_image, _filter) * 2, _filter.transpose()) * 2

def gen_up_down( _list ):
    __ = _list[0]
    for i, level_g in enumerate ( _list ) :
        if level_g.shape[0] < __.shape[0] :
            yield i-1, i, level_g

def gen_down_up( _list ):
        __ = _list[-1]
        for i, level_g in list(enumerate ( _list ))[-2::-1]:
            yield i, i+1, level_g

            # print(_list[i+1].shape)
            # print(level_g.shape)
          #if level_g.shape[0] < __.shape[0] :


from copy import deepcopy
def restore_or_build_laplacian_pyramid(pyramid, _filter, coeff, _operator_, gen):
    laplacian_pyramid = []
    print("len : {}".format( len(pyramid)))
    for j, i, level_g in gen( pyramid ):
        print(j)
        pyramid[j] *= coeff[j] 
        # laplacian_pyramid.append(
        pyramid[j] = _operator_ (pyramid[j], expend(pyramid[i], _filter)) 
    laplacian_pyramid = deepcopy(pyramid)
    return laplacian_pyramid
    

def build_laplacian_pyramid(im, max_levels, filter_size):
    _filter = gaussian_filter(filter_size)
    pyramid = build_gaussian_pyramid(im, max_levels, filter_size, [])
    laplacian_pyramid = restore_or_build_laplacian_pyramid(pyramid, _filter, np.ones(max_levels), __sub__, gen_up_down)
    laplacian_pyramid.append( pyramid[-1] )
    return laplacian_pyramid

def laplacian_to_image(lpyr, filter_vec, coeff):    
    return restore_or_build_laplacian_pyramid(lpyr, filter_vec , coeff , lambda x, y:  x+y, gen_down_up)

def render_pyramid(pyr, levels):
    pass
def display_pyramid(pyr, levels):
    pass

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    # assert im1.shape == im2.shape == mask.shape
    L_1 =  build_laplacian_pyramid( im1, max_levels, filter_size_im)  
    L_2 =  build_laplacian_pyramid( im2, max_levels, filter_size_im)     
    G_m = build_gaussian_pyramid(mask, max_levels, filter_size_mask, [])
    L_out = list(map(lambda k : G_m[k] * L_1[k] + ( 1 - G_m[k] ) * L_2[k], list(range(max_levels)) ))
    _filter = gaussian_filter(filter_size_im)
    return laplacian_to_image(L_out, _filter, np.ones(max_levels) )

def heart_beat_test():
    # gs = gridspec.GridSpec(4, 4, width_ratios=[8, 4 ,2, 1],height_ratios=[8, 4 ,2, 1] )
    # for _p in [  build_laplacian_pyramid]:    #build_gaussian_pyramid
    #     for j,im in zip( [0,5, 10, 15 ], _p(image, 6, 8) ):
    #         ax0 = plt.subplot(gs[j])
    #         ax0.imshow( im.astype(np.float64), cmap = "gray")
    #     plt.show()
    
    # _filter = gaussian_filter(8)
    #  laplacian_to_image(pyramid, _filter, np.ones(8) )
    image = rgb2gray(imread("yuv2.jpg")) 
    # plt.subplot(2,2,1)
    # plt.axis('off')
    # plt.imshow( image, cmap = "gray" )
    print(image.shape)
    image2 = rgb2gray(imread("astro.jpg"))
    # plt.subplot(2,2,2)
    # plt.axis('off')
    # plt.imshow( image2, cmap = "gray" )  
    mask = rgb2gray(imread("mask.jpg")) 
    # plt.subplot(2,2,3)
    # plt.axis('off')
    # plt.imshow( mask, cmap = "gray" )
    print(mask)
    print(image)
    ind = mask < 0.2
    ind2 = mask >= 0.2
    mask[ind] = False
    mask[ind2] = True
    print(len(ind))
    print(len(ind2))
    res = pyramid_blending(image2.astype(np.float64), image.astype(np.float64), mask, 5, 12 , 12)
    # plt.subplot(2,2,4)
    plt.axis('off')

    plt.imshow( res[0].astype(np.float64), cmap = "gray" )
    plt.show()

if __name__ == "__main__" :
    # print( gaussian_filter(3) )
    heart_beat_test()
    # pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    # pyr, filter_vec = build_laplacian_pyramid(im, max_levels, filter_size)
