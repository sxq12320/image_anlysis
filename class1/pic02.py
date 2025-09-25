import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('class1\img\pic02.tif')

gamma = 0.3
c = 1

gray_norm = img.astype(np.float32) / 255.0

enhanced = c * np.power(gray_norm, gamma)

enhanced = np.clip(enhanced, 0, 1)
enhanced = (enhanced * 255).astype(np.uint8)

plt.subplot(1,2,1)
plt.imshow(img , cmap= 'gray')
plt.title('before')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(enhanced,cmap= 'gray')
plt.title('after')
plt.axis('off')

plt.tight_layout()
plt.show()