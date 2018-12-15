import warnings
import numpy as np
from scipy.misc import imread, imsave
from skimage import data
from skimage.transform import resize
from keras.models import load_model

import sys

warnings.filterwarnings('ignore')

def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def float_image_to_uint8(im):
    return (im * 255).round().astype('uint8')

def predict_custom_image(image=None, model=None):
    if isinstance(image, str):
        im = imread(image)
    else:
        im = image

    if len(im.shape) == 2:
        w, h = im.shape
        im = np.empty((w, h, 3), dtype=np.uint8)
        im[:, :, 0] = im
        im[:, :, 1] = im
        im[:, :, 2] = im

    target_size = model.input.__dict__['_keras_shape'][1:-1]

    im_resize = resize(im, target_size)
    im = np.expand_dims(im_resize, 0)
    preds = model.predict(im)
    pred = preds[:, :, :, 0][0]

    return pred

if __name__ == '__main__':
    file_name = 'SNR-12-Data.jpg'
    unet = load_model('unet-original.keras')
    c = predict_custom_image(file_name, unet)
    imsave('edges.jpg', c)

