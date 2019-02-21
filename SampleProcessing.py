import numpy as np
import sys
import os
import math
#样本中心，目前设置为（32,32,0）
center=np.array([32,32,0])

def down_scale_sample(source_array, scale):
    ''' 向下缩放一个样本，返回缩放后的结果 '''
    target_array= np.zeros(source_array.shape,dtype=bool)
    for x in range(0,64):
        for y in range(0,64):
            for z in range(0,64):
                if source_array[x,y,z] == True:
                    position = np.array([x,y,z])
                    # 先将坐标减去32，32，0，获得相对于中心的坐标，然后缩放，最后加上32,32,0，转换回真正的坐标
                    new_position = np.floor((position-center)*scale)+center
                    new_position= new_position.astype(int)
                    target_array[new_position[0],new_position[1],new_position[2]]= True 
    return target_array

def revolve_sample(source_array, j):
    target_array = np.zeros(source_array.shape, dtype=bool)
    for x in range(0, 64):
        for y in range(0, 64):
            for z in range(0, 64):
                if source_array[x, y, z] == True:
                    position = np.array([x, y, z])
                    vector=position-center
                    a=35*j*3.14/180
                    array1=np.array([math.cos(a),math.sin(a),0])
                    array2=np.array([-math.sin(a),math.cos(a),0])
                    array3=np.array([0,0,1])
                    position_x=vector[0]*array1[0]+vector[1]*array1[1]+vector[2]*array1[2]
                    position_y=vector[0]*array2[0]+vector[1]*array2[1]+vector[2]*array2[2]
                    position_z=vector[0] * array3[0] + vector[1] * array3[1] + vector[2] * array3[2]
                    position=np.array([position_x,position_y,position_z])
                    new_position=position+center
                    for i in range(0,2):
                        if new_position[i]>63:
                            new_position[i]=63
                        if new_position[i]<0:
                            new_position[i]=0

                    new_position = new_position.astype(int)
                    target_array[new_position[0], new_position[1], new_position[2]] = True
    return target_array










            #usage SampleProcessing train_data.db.npy 0.8 6 train_data2.db
if __name__ == '__main__':
    result_list=[]
    if len(sys.argv)==5:
        samples= np.load(sys.argv[1])
        base_scale=float(sys.argv[2])
        if len(samples.shape)==4:
            for i in range(0,samples.shape[0]):
            #for i in range(0, 10):
                print ("%d / %d \n"%(i,samples.shape[0]))
                result_list.append(samples[i])
                for j in range (1,int(sys.argv[3])):
                    scaled_result=down_scale_sample(samples[i],math.pow(base_scale,j))
                    result_list.append(scaled_result)
                    for k in range (1,6):
                        result_list.append(revolve_sample(scaled_result,k))
            np.save(sys.argv[4],np.array(result_list,dtype=bool,copy=False))