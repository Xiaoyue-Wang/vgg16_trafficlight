#!/usr/bin/python3.5
import numpy as np
#import tensorflow as tf
import cv2

def getImageData(filename):
    image=cv2.imread(filename)
    size=image.shape
    print(size)
    #grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    npdata=np.array(image,dtype='float32')/255.0
    newdata=np.reshape(npdata,size)

    return newdata

#this is a test
#print(getImageData('/home/xiaoyue/PycharmProjects/vgg16_learn/image/taffic_light/010_20180927/4539_red.jpg'))

