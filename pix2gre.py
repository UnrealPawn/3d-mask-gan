import cv2,os
import numpy as np
from PIL import Image


def ImageToMatrix(image):
    matrix = np.asarray(image)
    return matrix


def matrixpic(img, x, y):
    matrixpic = ImageToMatrix(img)
    print(matrixpic.shape)
    for i in range(512):
        for j in range(512):
            if matrixpic[i][j] > x:

                    matrixpic[i][j] = 255

            else:
                matrixpic[i][j] =0

    return matrixpic
path_1=r'D:\Unity\New Unity Project'
path_2 = r'D:\Unity\image256'
# for i in range(1,1000):
#
#     if i < 10 :
#         number = '00' + str(i)
#     else:
#         if i < 100:
#             number = '0' + str(i)
#         else:
#             number =str(i)
#
#     path_11=number+'.png'
#     pathx=os.path.join(path_1,path_11)
#
#
#     #path_22 = str(i) + '.jpg'
#     pathy = os.path.join(path_2, path_11)
#
#     print(pathx)
#     print(pathy)
#     img = cv2.imread(pathx)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
#     matrixpicd=matrixpic(gray,254,255)
#     cv2.imwrite(pathy, matrixpicd)

for i in range(1,1000):
    if i < 10:
        number = '00' + str(i)
    else:
        if i < 100:
            number = '0' + str(i)
        else:
            number =str(i)
    name = number + 'y30.png'
    pathx = os.path.join(path_1, name)
    pathy = os.path.join(path_2, name)
    img = Image.open(pathx)
    dst = img.resize((256, 256), Image.ANTIALIAS)
    print(i)
    # i = i[:-4] + '.png'
    dst.save(pathy, quality=100)


for i in range(1,1000):
    if i < 10:
        number = '00' + str(i)
    else:
        if i < 100:
            number = '0' + str(i)
        else:
            number =str(i)
    name = number + 'y50.png'
    pathx = os.path.join(path_1, name)
    pathy = os.path.join(path_2, name)
    img = Image.open(pathx)
    dst = img.resize((256, 256), Image.ANTIALIAS)
    print(i)
    # i = i[:-4] + '.png'
    dst.save(pathy, quality=100)


for i in range(1,1000):
    if i < 10:
        number = '00' + str(i)
    else:
        if i < 100:
            number = '0' + str(i)
        else:
            number =str(i)
    name = number + 'y-50.png'
    pathx = os.path.join(path_1, name)
    pathy = os.path.join(path_2, name)
    img = Image.open(pathx)
    dst = img.resize((256, 256), Image.ANTIALIAS)
    print(i)
    # i = i[:-4] + '.png'
    dst.save(pathy, quality=100)
#
#
#
# x=[]
# for i in range(1,200):
#     #path_2 = r'D:\RockGan\3DGANServer\tool\time2'
#     path_22 = str(i) + '.jpg'
#     pathx = os.path.join(path_2, path_22)
#
#     img = Image.open(pathx)
#     img_np=np.array(img)/255
#     #img_np_1d=img_np.reshape(250000)/255
#     x.append(img_np)
#     print(img_np)
#
# x=np.array(x)
# np.save('X128_front_left.npy',x)
