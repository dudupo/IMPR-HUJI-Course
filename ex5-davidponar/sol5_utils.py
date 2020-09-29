
from imageio import imread, imwrite
from skimage.color import rgb2gray

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
