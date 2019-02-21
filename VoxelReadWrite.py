import numpy as np
from numpy import *
import struct
import random
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import time

import os
def cuboid_data(pos, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return x, y, z

def plotCubeAt(pos=(0,0,0),ax=None):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos )
        ax.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=1)

def plotMatrix(ax, matrix):
    # plot a Matrix 
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i,j,k] == 1:
                    # to have the 
                    plotCubeAt(pos=(i-0.5,j-0.5,k-0.5), ax=ax)    

#import vox file and return a 3-dimition nparray
def importVox(file_name,size):
    result_array=zeros(shape=(size,size,size),dtype=bool)
    with open(file_name, 'rb') as f:
        bytes = f.read(4)
        file_id = struct.unpack(">4s",  bytes)
        if file_id[0] == b'VOX ':
            f.seek(8)
            bytes=f.read(4)
            main_string=struct.unpack(">4s",bytes);
            if main_string[0]==b'MAIN':
                f.seek(72,0)
                #read number of voxels, stuct.unpack parses binary data to variables
                bytes = f.read(4)
                numvoxels = struct.unpack('<I', bytes)
            
                #iterate through voxels
                for x in range(0, numvoxels[0]):
                    bytes = f.read(4)
                    voxel = struct.unpack('<bbbB', bytes)
                    result_array[voxel[0]][voxel[1]][voxel[2]]=True
    return result_array
def importVoxByPath(path,size):
    result_array=array([],dtype=bool)
    count=0
    print('start procession:\n')
    for i in os.walk(path):
        for filename in i[2]:
            file_array=importVox(i[0]+filename,size)
            result_array=np.append(result_array,file_array)
            file_array_fliped=np.array(file_array,copy=True)
            file_array_fliped=np.fliplr(file_array_fliped)
            result_array=np.append(result_array,file_array_fliped)
            file_array_fliped=np.array(file_array,copy=True)
            file_array_fliped=np.flipud(file_array_fliped)
            result_array=np.append(result_array,file_array_fliped)
            count=count+3
    result_array=result_array.reshape((count,64,64,64))
    np.save('train_data.db',result_array)
    return result_array

def writeVox(nparray):
    pass

def showVox(nparray):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    plotMatrix(ax, nparray)
    plt.show()

if __name__=='__main__':
    ma=importVoxByPath('E:\\ObjData\\voxs_64\\',64)
    showVox(ma[0])
    showVox(ma[1])
    showVox(ma[2])
    showVox(ma[3])
