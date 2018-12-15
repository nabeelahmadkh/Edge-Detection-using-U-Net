import glob
import numpy as np
import cv2
from PIL import Image
from PIL.ImageOps import grayscale

# Loading Data from locally saved files
imageList_1 = glob.glob('C:\\Users\\Subham\\Documents\\EdgeDetectionVGG16\\True Image\\*.jpg')
imageList_2 = glob.glob('C:\\Users\\Subham\\Documents\\EdgeDetectionVGG16\\Computed Image\\*.jpg')

True_Image_array = []
Computed_Image_array = []

'''
for image in imageList_1:
    img = cv2.imread(image)
    True_Image_array.append(img)

for image in imageList_2:
    img = cv2.imread(image)
    Computed_Image_array.append(img)
'''
cimg1=Image.open(imageList_1[4])
gimg1=grayscale(cimg1)
gimg1=gimg1.resize((256,256),resample=0)
gimg1.save('cropped.jpg')
timg=np.asarray(gimg1)

cimg2=Image.open(imageList_2[4])
gimg2=grayscale(cimg2)
gimg2.save('computed.jpg')
img=np.asarray(gimg2)

'''
True_Image_np_array = np.array(True_Image_array[0])
Computed_Image_np_array = np.array(Computed_Image_array[0])

print (True_Image_np_array.shape)
'''
w,h=gimg2.size

numerator=0
denominator=0


for i in range(0,w):
    for j in range(0,h):
        if img[i][j]>0:
            inten=1
        else:
            inten=0
        if timg[i][j]>100:
            tinten=0
        else:
            tinten=1
        if inten==tinten:
            numerator=numerator+1
        denominator=denominator+1
print(numerator, denominator)
accuracy=numerator/denominator

#accuracy=(np.abs(img-timg)<1e-10).all(axis=(0,2)).mean()

print ("accuracy: ",accuracy)

#print(Computed_Image_np_array.shape)
