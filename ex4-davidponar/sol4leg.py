import sys
import os

PACKAGE_PARENT = '../ex3-davidponar'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))




from sol3 import * 
from scipy.ndimage.filters import convolve as imconv 
import numpy as np

def get_lap_derivate(image):
    derivate_vev = np.array([1, 0, -1])
    Ix, Iy = *( imcov(image,vec) for vec in [ derivate_vev, derivate_vev.transpose() ] ) 
    Ix = np.tensordot(  Ix , np.array( [[1 , 0 ], [0 , 0 ]]) ,axes=0)
    Iy = np.tensordot(  Iy , np.array( [[0 , 0 ], [0 , 1 ]]) ,axes=0)
    return np.einsum('ijk,ijn->ijkn', Ix, Iy)
    
def respone( _matrix ):
    return np.linalg.det( _matrix ) - 0.04 * np.matrix.trace( _matrix )

def create_respone(Intensity):
    return np.array(list( map(
         lambda raw : list( map(
              lambda cell : respone (cell) , raw)) ,  Intensity)))


def harris_corner_detector(im):
    return np.argwhere(
        non_maximum_suppression(
            create_respone(
                get_lap_derivate(im))))


    

def non_maximum_suppression(im):
    pass

def sample_descriptor(im, pos, desc_rad):
    pass

def find_features(pyr):
    #build_gaussian_pyramid(  )
    desc_rad = []
    pos = harris_corner_detector( pyr[0] )
    pyr = sample_descriptor(pyr[0], pos, desc_rad) 
    