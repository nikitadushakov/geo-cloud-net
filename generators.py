import random
from skimage.io import imread
from skimage.transform import resize
import numpy as np

"""
Some lines borrowed from https://www.kaggle.com/petrosgk/keras-vgg19-0-93028-private-lb
"""


def get_image_from_file(file, img_rows, img_cols, max_possible_input_value):
    image_red = imread(file[0])
    image_green = imread(file[1])
    image_blue = imread(file[2])
    image_nir = imread(file[3])

    image = np.stack((image_red, image_green, image_blue, image_nir), axis=-1)

    image = resize (image, ( img_rows, img_cols), preserve_range=True, mode='symmetric')


    image /= max_possible_input_value
    return image



