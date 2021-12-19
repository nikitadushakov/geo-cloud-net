
from __future__ import print_function
import os
import numpy as np
import cloud_net_model
from generators import get_image_from_file
import tifffile as tiff
import pandas as pd



def get_input_image_names(list_names, directory_name):
    list_img = []
    list_msk = []
    list_test_ids = []

    for filenames in list_names['name']:
        nred = 'red_' + filenames
        nblue = 'blue_' + filenames
        ngreen = 'green_' + filenames
        nnir = 'nir_' + filenames

        dir_type_name = "test"
        fl_img = []
        fl_id = '{}.TIF'.format(filenames)
        list_test_ids.append(fl_id)

        fl_img_red = directory_name + '/' + dir_type_name + '_red/' + '{}.TIF'.format(nred)
        fl_img_green = directory_name + '/' + dir_type_name + '_green/' + '{}.TIF'.format(ngreen)
        fl_img_blue = directory_name + '/' + dir_type_name + '_blue/' + '{}.TIF'.format(nblue)
        fl_img_nir = directory_name + '/' + dir_type_name + '_nir/' + '{}.TIF'.format(nnir)
        fl_img.append(fl_img_red)
        fl_img.append(fl_img_green)
        fl_img.append(fl_img_blue)
        fl_img.append(fl_img_nir)

        list_img.append(fl_img)

    
    
    return list_img, list_test_ids



# path to dataset
GLOBAL_PATH = '/Users/nikitadushakov/Downloads/archive'
# TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
TEST_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_test')
PRED_FOLDER = './Predictions'
if not os.path.exists(PRED_FOLDER):
    os.mkdir(PRED_FOLDER)



in_rows = 384
in_cols = 384
num_of_channels = 4
num_of_classes = 1
batch_sz = 10
max_bit = 65535  # maximum gray level in landsat 8 images
weights_path = 'model_weights.h5'


# getting input images names
test_patches_csv_name = 'test_patches_38-Cloud.csv'
df_test_img = pd.read_csv(os.path.join(TEST_FOLDER, test_patches_csv_name)).iloc[1500:1510]
test_img, test_ids = get_input_image_names(df_test_img, TEST_FOLDER)


def test():
    model = cloud_net_model.model_arch(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.load_weights(weights_path)
    for k, i in enumerate(test_img):
        test_image = np.array([get_image_from_file(i, in_rows, in_cols, max_bit)])

        mask = model.predict(test_image)
        print(k)
        tiff.imsave(os.path.join(PRED_FOLDER,test_ids[k]), mask)
test()