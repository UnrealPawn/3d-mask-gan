import os,shutil
import re
rootDir = r"F:\迅雷下载\ShapeNetCore.v2\ShapeNetCore.v2\02958343_car"
changepath = r"F:\model_car"

n=0
def changeline():
    for n in range(464,1000):
        path =os.path.join(changepath,str(n))+'.txt'
        file = open(path,'r')
        lines = file.readlines()
        if re.search('model_normalized',lines[3]):
            lines[3] = re.sub('model_normalized',str(n),lines[3])
        file.close()
        file1 = open(path, 'w')
        file1.writelines(lines)
        file1.close()

        new_file_name = path.replace(".txt", ".obj")
        os.rename(path, new_file_name)
        print(n)


def traverse(f):
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f,f1)
        if not os.path.isdir(tmp_path):
            if tmp_path.endswith('.obj'):
                global n
                n=n+1

                path=os.path.join(r"F:\model_car",str(n))
                path=path+'.txt'
                shutil.copy(tmp_path,path)
                print(tmp_path)

                mtlpath = tmp_path[0:-4] + '.mtl'
                path = os.path.join(r"F:\model_car", str(n))
                path = path + '.mtl'
                shutil.copy(mtlpath, path)

        else:

            #print('文件夹：%s'%tmp_path)
            traverse(tmp_path)
#traverse(rootDir )
changeline()