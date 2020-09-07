from functools import reduce
from math import exp
import numpy as np
import scipy.io.wavfile as wav



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

# class array_warpper( np.array ):
#     def __init__(self ):
#         pass 
    
def _FFT( signal, N, inv=False ):

    # if len(signal) == 0:
    #     return np.array([0])

    if len(signal) == 1:
        return signal

    even = _FFT( signal[0::2], N, inv)
    odd  = _FFT( signal[1::2], N, inv)
    phase = 1.j * 2 * np.pi/len(signal)

    if inv : 
        phase *= -1

    ret = np.zeros( len(signal ))
    hist = int(len(signal)/ 2)
    for i in range(hist) : 
        ret[i] = even[i] + np.exp(-phase * i) * odd[i]
        ret[i + hist] =  even[i] - np.exp(-phase * i) * odd[i]
    return  ret  

def IFFT(signal):
    return _FFT(signal, len(signal), inv=True)/ len(signal)

def FFT(signal):
    return _FFT(signal, len(signal))

def SFFT(signal, win_func, nperseg = 6, noverlap=None):
    if noverlap is None:
        noverlap = 40
    

    fourier_signal = FFT(signal)
    fourier_window = FFT( win_func( np.linspace(-2,2,nperseg))) * 100
    ret = np.zeros( ( noverlap, len(signal) ) )

    for i  in range(noverlap):
        print(i)
        win_start = int( i * len(signal) / noverlap) 
        ret[i][win_start:win_start+nperseg] = fourier_signal[win_start:win_start+nperseg] * fourier_window
    return ret    

def ISFFT(ssft_signal):
    return IFFT(np.array( np.sum(ssft_signal.transpose())))

def change_rate(filename, ratio):
    rate, wav_signal = wav.read(filename)
    wav.write( filename, ratio * rate, wav_signal )

def resize(data, ratio):
    return IDFT( DFT( data )[:- (1- ratio) * len(data)] )

def change_samples(filename, ratio):
    rate, wav_signal = wav.read(filename)
    compress_signal = np.real(resize(wav_signal, ratio))
    wav.write(filename, rate, compress_signal)
    return compress_signal

def resize_spectrogram(data, ratio):
    pass



from matplotlib import pyplot as plt
if __name__ == "__main__":    
    rect = np.vectorize( lambda x : 1 if abs(x) < 1 else 0 )
    signal = rect(np.linspace(0, 200, 2**8))
    ret =  SFFT(signal ,rect)
    # print( ret[2][10:30])
    plt.imshow( SFFT(signal ,rect)) 
    plt.show()
