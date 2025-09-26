import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

Threshold =50

gamma1 = 1.2
c1 = 2

gamma2 = 0.6
c2 = 3.0

img = cv.imread('class1/img/pic06.tif')

if len(img.shape)==3:
    img = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

def enhance(image , gamma , c):
    enhanced = image.astype(np.float32)/255.0
    enhanced = c * np.power(enhanced , gamma)
    enhanced = (enhanced*255.0).astype(np.uint8)
    return enhanced
mask = img>Threshold

enhanced1 = img.copy()
enhanced1[mask] =(img[mask] * 0.8).astype(np.uint8)

enhanced2 = enhance(enhanced1 , gamma2 , c2)



plt.subplot(1 , 3 , 1)
plt.imshow(img , cmap='gray')
plt.title('before')
plt.axis('off')

plt.subplot(1 , 3 , 2)
plt.imshow(enhanced1 , cmap='gray')
plt.title('after_first')
plt.axis('off')

plt.subplot(1 , 3 , 3)
plt.imshow(enhanced2 , cmap='gray')
plt.title('after_second')
plt.axis('off')

plt.tight_layout()
plt.show()
