import cv2,os
import pyglet,os,sys
import numpy as np
import struct
#import progressbar
from numpy import *
from pyglet.window import key,mouse
from pyglet.gl import *
import time
import numpy as np

#path_2 = r'D:\RockGan\3DGANServer\tool\time64'
#path_22 = str(1) + '.jpg'
#path2 = os.path.join(path_2, path_22)
bound_array = array([], dtype=bool)
for count in range(1,200):
    data =  np.load('X64.npy')
    #img = cv2.imread(path2)
    one_array = zeros(shape=(64,64,64),dtype=bool)
    num = count-1
    for x in range(0,64):
        for y in range(0,64):
            if data[num][x][y] > 0.5:
                for i in range(0,64):
                    one_array[y][x][i] = True

    bound_array = np.append(bound_array, one_array)
    bound_array = bound_array.reshape([count,64,64,64])

np.save('bound_64.db', bound_array)




#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
