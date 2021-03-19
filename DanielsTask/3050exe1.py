import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
red1,green1=0,0
raccoon = Image.open(r'unit10\raccoon.png')
array = np.array(raccoon)
plt.imshow(raccoon)
plt.show()
tic = time.time()
for r in range(len(array)):
    for c in range(len(array[0])):
        if(array[r][c][0] > array[r][c][1]):
            red1 += 1
        if(array[r][c][1] > array[r][c][0]):
            green1 += 1
toc = time.time();
print ("Non Vectorized version: red = " + str(red1) + ", green = " + str(green1) + ". It took " + str(1000*(toc-tic)) + "ms")
tic = time.time()
red_array = array[:,:,0]
green_array = array[:,:,1]
red1 = np.sum(red_array > green_array)
green1 = np.sum(red_array < green_array)
toc = time.time();
print ("Vectorized version: red = " + str(red1) + ", green = " + str(green1) + ". It took " + str(1000*(toc-tic)) + "ms")