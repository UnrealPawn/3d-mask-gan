import os,shutil,sys
import importlib
from   PIL import Image
import numpy as np
from numpy import *
rootDir = r"D:\RockGan\image_1/"
saveDir = r"D:\RockGan\image_1_cute/"
sobelDir = r"D:\RockGan\image_1_sobel/"
sobel=importlib.import_module("GetImageSobel")

n=0
im_array = array([], dtype=int)
label_array = array([],dtype=int)
def traverse(f):
    global im_array,label_array
    fs = os.listdir(f)
    fs.sort()
    for tmp_path in fs:
        if tmp_path.endswith('.png'):
            full_path = f+tmp_path
            save_path = saveDir + tmp_path
            sobel_path = sobelDir +tmp_path
            global n
            n=n+1
            x_cute = 350
            y_cute = 35
            w_cute = 500
            h_cute = 500
            im = Image.open(full_path)
            im = im.crop((x_cute, y_cute, x_cute + w_cute, y_cute + h_cute))
            im.save(save_path)
            im_mat = sobel.lines_rec(save_path,sobel_path)
            im_array = np.append(im_array,im_mat)
            im_array = im_array.reshape([n,500,500])


            if tmp_path[2] == '-':
                if tmp_path[4] == 'x':
                    y = -int(tmp_path[3])
                else:
                    y = - int (tmp_path[3:5])
            else:
                if tmp_path[3] == 'x':
                    y = int (tmp_path[2])
                else:
                    y = int (tmp_path[2:4])

            number = tmp_path.find('x_') + 2
            if tmp_path[number] == '-':
                if tmp_path[number+2] == '.':
                    z = -int(tmp_path[number+1])
                else:
                    z = - int(tmp_path[number+1:number+3])
            else:
                if tmp_path[number+1] == '.':
                    z = int(tmp_path[number])
                else:
                    z = int(tmp_path[number:number+2])
            print (str(y)+"------" + str(z) )
            label_array = np.append(label_array, y)
            label_array = np.append(label_array, z)
            label_array = label_array.reshape([n, 2])

            # path=os.path.join(r"F:\models",str(n))
            # path=path+'.obj'
            # shutil.copy(tmp_path,path)
            # print(tmp_path)
    np.save('image_1.db', im_array)
    np.save('label_1.db', label_array)

traverse(rootDir )