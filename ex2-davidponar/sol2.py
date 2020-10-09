from functools import reduce
from math import exp
import numpy as np
import scipy.io.wavfile as wav

from tqdm import tqdm




def create_DFT_func(phase):

    def _DFT_func( signal ):
        indices = np.arange(len(signal)) 
        indices = np.tensordot(indices, indices, axes=0 )  
        phases = np.exp( 2*np.pi*phase  * indices / len(signal)  )
        return np.einsum( 'i,ij->j', signal, phases)
    
    return _DFT_func

DFT, IDFT = ( create_DFT_func( phase ) for phase in [ -1.j, 1.j ] ) 

DFT2, IDFT2 = ( lambda image :
 np.apply_along_axis(_fun, 0, np.apply_along_axis(_fun, -1, image))
  for _fun in  [ DFT, IDFT ] ) 

def change_rate(filename, ratio):
    rate, wav_signal = wav.read(filename)
    wav.write( filename, ratio * rate, wav_signal )

def resize(data, ratio):    
    res = DFT( data )
    cut = int((1- (1/ratio))/2 *len(data))
    res = np.fft.ifftshift(np.fft.fftshift(res)[cut:-cut])
    ret = IDFT(res) 
    return ret

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


def change_samples(filename, ratio):
    rate, wav_signal = wav.read(filename)
    compress_signal = np.real(resize(wav_signal, ratio))
    wav.write(filename, rate, compress_signal)
    return compress_signal

from matplotlib import pyplot as plt
from scipy.signal import stft, istft
from math import cos

def resize_spectrogram(data, ratio):
    def resize_win( data ):
        cut = int((1- (1/ratio))/2 *len(data))
        return data[cut : -cut]
    return istft(np.apply_along_axis(resize_win, -1 , stft( data )[-1] ))[-1]

def resize_spectrogram_file(filename, ratio):
    rate, wav_signal = wav.read(filename)
    compress_signal = np.real(resize_spectrogram(wav_signal, ratio))
    wav.write(filename, rate, compress_signal)
    return compress_signal

def resize_vocoder(data, ratio):
    pass

from scipy.ndimage.filters import convolve as imconv 

def conv_der(im):
    derivate_vec = np.array([[1, 0, -1]], dtype=np.float64 )
    dx, dy = ( imconv(im ,vec, mode='constant') for vec in [ derivate_vec, derivate_vec.copy().transpose() ] ) 
    
    # for test
    #dx, dy = DFT2(dx) , DFT2(dy) 
    
    return  np.sqrt( np.abs(dx) ** 2 +  np.abs(dy) ** 2 )

def fourior_der(im):
    
    def derivate_phase( _shape ):
        ret =  np.zeros( _shape )
        derivate_vec = np.array([1, 0, -1], dtype=np.float64 )
        middle_X = ret.shape[0]//2 -1 
        middle_Y = ret.shape[1]//2 -1
        for j in range(derivate_vec.shape[0]):
            ret[middle_Y][middle_X + j] = derivate_vec[j]
        return DFT2(ret)
    
    der_phase_x = derivate_phase(im.shape)
    der_phase_y = derivate_phase(im.shape[::-1]).transpose()    
    image_phase = DFT2(im)

    dx , dy = IDFT2(image_phase * der_phase_x), IDFT2(image_phase * der_phase_y)  
    return np.sqrt( np.abs(dx) ** 2 +  np.abs(dy) ** 2 )
 

def test_rects():
    rect = np.vectorize( lambda x : 1 if abs(x) < 4 else 0 )
    signal = rect(np.linspace(0, 2**5, 2**7 ))
    fig = plt.figure()
    result = np.fft.fftshift( DFT( signal))
    plt.imshow( np.real( np.fft.fftshift(DFT2( np.tensordot(signal, signal, axes=0 ) ) ) ))
    plt.subplot(211).plot(  result )
    plt.subplot(212).plot(  np.fft.fftshift( np.fft.fft(signal)) )
    plt.show()

def test_2d():
    rect = np.vectorize( lambda x : 1 if abs(x) < 4 else 0 )
    signal = rect(np.linspace(0, 2**5, 2**7 ))
    fig = plt.figure()
    plt.imshow( np.real( np.fft.fftshift(DFT2( np.tensordot(signal, signal, axes=0 )) ) ) )
    plt.show()
    plt.imshow( np.real( np.fft.fftshift(IDFT2(DFT2( np.tensordot(signal, signal, axes=0 )) ) ) ))
    plt.show()

def test_resize_spect():
    rect = np.vectorize( lambda x : 1 if abs(x) < 4 else 0 )
    signal = rect(np.linspace(0, 2**5, 2**12 ))
    fig = plt.figure()
    result = resize_spectrogram( signal, 1.2 )
    plt.subplot(211).plot(  result )
    plt.show()

def test_der():
    rect = np.vectorize( lambda x : 1 if abs(x) < 1 else 0 )
    signal = rect(np.linspace(0, 2**3, 2**7 ))
    fig = plt.figure()
    plt.imshow( np.fft.fftshift(conv_der( np.tensordot(signal, signal, axes=0 ))))
    plt.show() 
    

def test_fourior_der():
    rect = np.vectorize( lambda x : 1 if abs(x) < 1 else 0 )
    signal = rect(np.linspace(0, 2**3, 2**7 ))
    fig = plt.figure()
    # plt.imshow( np.fft.fftshift(fourior_der( np.tensordot(signal, signal, axes=0 ))))
    plt.imshow(fourior_der( np.tensordot(signal, signal, axes=0 )))
    plt.show() 
    

if __name__ == "__main__":  
    # test_g()
    # test_rects()  
    # test_2d()
    # test_resize_spect()
    test_der()
    test_fourior_der()
    # from scipy.signal.windows import hann
    
    # rect = np.vectorize( lambda x : 1 if abs(x) < 4 else 0 )
    # signal = rect(np.linspace(0, 2**12, 2**12))
    # han_array =  hann( 156 )
    # plt.imshow(SFFT(signal , lambda vec : han_array  , nperseg = 156, noverlap=300))
    # plt.show()

    # resize_spectrogram_file("AUD_brief_test.wav", 1.2)

    # ret =  SFFT(signal ,rect)
    # print( ret[2][10:30])
    #plt.imshow( SFFT(signal ,rect)) 
    #plt.plot( ISFFT(  SFFT(signal ,rect) ))