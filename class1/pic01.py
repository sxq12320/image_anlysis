import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('class1\img\pic01.jpg')
median_img = cv.medianBlur(img , 3)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('before')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(median_img, cmap='gray')
plt.title('after')
plt.axis('off')
plt.show()