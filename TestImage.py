from PreprocessImage import *
from keras.models import load_model
import matplotlib.pyplot as plt

unet = load_model('unet-original.keras')

CustomImage = 'SNR-12-Data.jpg'
c=predict_custom_image(CustomImage, model=unet)
imsave('edges.jpg', c)
