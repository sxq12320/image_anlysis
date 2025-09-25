import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('class1/img/pic06.tif')

if len(img.shaper)==3:
    img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

gramma1 = 1.5
c = 1

