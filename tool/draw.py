import numpy as np

from tool import PyVoxelVisualizer as pyvoxel

#
#结果存放在Documents/820 15:40
n = np.load("./results/600_0.npy")
pyvoxel.RequestRenderVoxel(n)
#

# n = []
# for i in range(50):
#   for j in range(1):
#    str1 = './results/' + str(i * 200) + '_' + str(j) + '.npy'
#
#    n.append(np.load(str1))
#
# n = np.array(n)
#
# pyvoxel.RequestRenderVoxel(n)

# source_array=np.load('train_data.db.npy')
# result_array=[]
# counter=0
# for i in range(source_array.shape[0]):
#     # if i >=63 and i <=69:
#     #     continue
#     # if i >=72 and i <= 74:
#     #     continue
#     # if i >= 90 and i <= 92:
#     #     continue
#     result_array.append(source_array[i])
#     counter+=1
#
# result_array=np.array(result_array)
# result_array=result_array.reshape((counter, 64, 64, 64))
# #np.save('train_data.db',result_array)
# pyvoxel.RequestRenderVoxel(result_array)







# g_lr = 0.0025
# d_lr = 0.000001  #全部都是正方形                                  有数据


# g_lr = 0.0025
# d_lr = 0.00001 #前三千六百次可以,训练时间越长,模型越失真              有数据


# g_lr = 0.008
# d_lr = 0.0000001 全是正方形                                      有数据

# g_lr = 0.0025
# d_lr = 0.0001 #前四千次可以,训练时间越长,模型越失真

# g_lr = 0.008
# d_lr = 0.000001 #2000次生成的是正方形，不行