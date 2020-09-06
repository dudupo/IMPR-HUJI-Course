from functools import reduce
from math import exp
import numpy as np




def create_DFT_func(phase):

    def _DFT_func( signal ):
        return np.vectorized( lambda u :
                reduce ( lambda x, fx :
                 fx*exp(  phase * x * u * np.pi * 2 / len(signal)  ),
                  enumerate(signal)) ) ( np.linspace( 0, len(signal)  ) ) 
        
    return _DFT_func

_DFT_private_signature = create_DFT_func( -1.j )

def DFT( signal ):
    return _DFT_private_signature(signal)
    
_IDFT_private_signature = create_DFT_func( 1.j )

def IDFT (fourier_signal):
    return _IDFT_private_signature(fourier_signal) / len(fourier_signal) 


def DFT2(image):
    return np.vectorized(DFT) ( map(DFT ,image) )

def IDFT2(image):
    return np.vectorized(IDFT) ( map(IDFT ,image) )

def change_rate(filename, ratio):
    pass 

def change_samples(filename, ratio):
    pass