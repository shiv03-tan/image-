# -*- coding: utf-8 -*-
"""apple_sam.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19i5PzDaizo04asElaMmGi71J-qf4SSUi
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as image
import urllib
import numpy as np

# load image from url
f = urllib.request.urlopen("https://thumbs.dreamstime.com/b/%E7%BA%A2%E8%8B%B9%E6%9E%9C%E4%B8%8E%E7%BB%BF%E8%8B%B9%E6%9E%9C%E6%9E%9C%E5%AE%9E%E7%89%B9%E5%86%99%E5%9B%BE%E5%83%8F-%E7%BE%8E%E4%B8%BD%E7%9A%84%E7%BA%A2%E8%8B%B9%E6%9E%9C%E5%92%8C%E7%BB%BF%E8%8B%B9%E6%9E%9C%E6%9E%9C%E5%AE%9E%E7%89%B9%E5%86%99%E5%9B%BE%E5%83%8F-174192824.jpg")
data = plt.imread(f, format='jpeg')
data.shape
plt.imshow(data)

def SAM(origin, target):
   IMG_res = np.full((origin.shape[0], origin.shape[1]), 0.0)
   for h in range(0, origin.shape[0]):
      for w in range(0, origin.shape[1]):
        b, g, r = origin[h, w, 0], origin[h, w, 1], origin[h, w, 2]
        bgr = np.array ([int(b), int(g), int(r)]).reshape(3,1)
        a = 0
        for i in range(0, len(bgr)):
          a += int(bgr[i])*int(target[i])
        c = float (np.sqrt(np.sum([int(x)**2 for x in bgr]))) * float (np.sqrt(np.sum([int(x)**2 for x in target])))
        IMG_res[h][w] = np.arccos(a/c)

   return IMG_res
target_b, target_g, target_r = data[400, 600, 0], data [400, 600, 1], data [400, 600, 2]
bgrtar = np.array([int(target_b), int(target_g), int(target_r)]). reshape(3,1)
IMG_res = SAM(data, bgrtar)
plt.imshow(IMG_res)
plt.plot(600, 400, "ro")
plt.title("SAM_APPLE")

fig=plt.figure()
fig.suptitle("difference")

ax = fig.add_subplot(221)
ax.title.set_text('Original Image')
ax.imshow(data,interpolation='nearest')

ax1 = fig.add_subplot(222)
ax1.title.set_text('SAM Image')
plt.plot(600, 400, "ro")
ax1.imshow(IMG_res,interpolation='nearest')
plt.show()

