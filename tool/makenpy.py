import numpy as np
import os
from PIL import Image

x=[]
for i in range(1,200):
    path_2 = r'C:\Users\Administrator\Documents\Tencent Files\183827350\FileRecv\gaoqing'
    path_22 = str(i) + '.jpg'
    path2 = os.path.join(path_2, path_22)

    img = Image.open(path2)
    img_np=np.array(img)
    img_np_1d=img_np.reshape(32*32)/255
    x.append(img_np_1d)
    print(img_np_1d)

x=np.array(x)
np.save('X.npy',x)
# np.save('Y.npy',y)