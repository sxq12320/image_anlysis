import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

sample_rate = 1000    # 采样率
duration = 10         # 持续时间
frequency = 20         # 震动频率

t = np.arange(0 , duration , 1/sample_rate) # 生成时间序列
u = np.sin(2*np.pi*frequency*t) # 生成正弦波sin(2 pi f t)

noise_amplitude = 1
noise = np.random.normal(0 , noise_amplitude , len(u))
u = noise + u # 加一些高斯白噪声





# plt.plot(t,u,'b-',label = '噪声', alpha = 0.7)
# plt.show()
