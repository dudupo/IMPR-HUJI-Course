from keras import layers, Input, Model
import keras
import numpy as np
from random import choice, randint
from sol5_utils import *

def load_dataset(filenames, batch_size, corruption_func, crop_size):

    cache = {}

    while True:
        batch =  [[ ], []]
        for _ in range( batch_size ):
            _filename = choice(filenames)
            if _filename not in cache:
                image = read_image( _filename ,REPRESENTATION_GRY) 
                image = np.array( [ [ [ image[y][x] ] for x in range( image.shape[1] )] for y in range(image.shape[0])] )
                
                cache[ _filename ] = ( image - 0.5*np.ones(image.shape),
                 corruption_func(image) - 0.5*np.ones(image.shape) ) 

            for i in range(2):                
                batch[i].append( cache[ _filename ][i])

            _shape = batch[0][-1].shape
            s_width, s_height = randint(0,  _shape[0] - crop_size), randint(0,  _shape[1] - crop_size)  
            
            for i in range(2):
                batch[i][-1] = batch[i][-1][ s_width:s_width+crop_size, s_height:s_height+crop_size ]

        yield np.array(batch)


def resblock(input_tensor, num_channels):
    x = layers.Conv2D( num_channels , (3 , 3), padding ='same')(input_tensor)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D( num_channels , (3 , 3), padding ='same')(x)
    layers.Add()([x, input_tensor])
    x = layers.Activation("relu")(x)
    return x

def build_nn_model(height, width, num_channels, num_res_blocks):
    input_tensor = Input(shape=(height, width, 1))
    x = layers.Conv2D( num_channels , (3 , 3), padding ='same' )(input_tensor)
    x = layers.Activation("relu")(x)
    for _ in range(num_res_blocks):
        x = resblock(x, num_channels)
    x = layers.Conv2D( 1 , (3 , 3), padding ='same' )(x)
    layers.Add()([x, input_tensor])
    x = layers.Activation("relu")(x) 
    return Model(input_tensor, x)

def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    
    crop_size = 5
    data_generator = load_dataset(images, batch_size, corruption_func, crop_size)
    validation_data = [ next(data_generator) for _ in range(num_valid_samples)]
    model.compile(optimizer=keras.optimizers.Adam(beta_2=0.9),loss="mean_squared_error")
    model.fit(data_generator, steps_per_epoch=steps_per_epoch, epochs=num_epochs, validation_data=validation_data)

def restore_image(corrupted_image, base_model):
    input_tensor = Input(shape=corrupted_image.shape)
    
    #
    #   clipping
    #  
    
    return Model(inputs=input_tensor, outputs=base_model(input_tensor) ).predict() 
    

def add_gaussian_noise(image, min_sigma, max_sigma):
    sigma = np.random.uniform(low=min_sigma, high=max_sigma, size=1)[0]
    image += np.random.normal( scale=sigma, size=image.shape )
    return np.around(image * 255)/255

def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    images = images_for_denoising()
    num_channels = 48
    height, width = 24 ,24 
    batch_size, steps_per_epoch, num_epochs, num_valid_samples =\
         { False : (100, 100, 5, 1000), True : (10, 3, 2, 30) }[quick_mode]
    model = build_nn_model(height, width,num_channels ,num_res_blocks)
    train_model(model, images, lambda image : add_gaussian_noise(image, 0, 0.2),
     batch_size, steps_per_epoch, num_epochs, num_valid_samples)
    return model    

from math import tan, radians
from  scipy.ndimage.filters import convolve 
def add_motion_blur(image, kernel_size, angle):
    conv_kernel = np.zeros( (kernel_size,kernel_size))
    for i in range( kernel_size ):
        conv_kernel[int( tan(angle)* i )][i] = 1
    return convolve(image, conv_kernel, 'full')
    
def random_motion_blur(image, list_of_kernel_sizes):
    for kernel_size in list_of_kernel_sizes:
        image = add_motion_blur( image,  np.random.uniform(low=0, high=np.pi ))
    return image

def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    images = images_for_deblurring()
    num_channels = 32
    height, width = 16 ,16 
    batch_size, steps_per_epoch, num_epochs, num_valid_samples =\
         { False : (100, 100, 10, 1000), True : (10, 3, 2, 30) }[quick_mode]
    model = build_nn_model(height, width,num_channels ,num_res_blocks)
    train_model(model, images, lambda image : random_motion_blur(image, [7]),
     batch_size, steps_per_epoch, num_epochs, num_valid_samples)
    return model    



