import pyglet,os,sys
import numpy as np
import struct
import cv2
#import progressbar
from numpy import *
from pyglet.window import key,mouse
from pyglet.gl import *
from math import sin,cos
#sys.setrecursionlimit(5000)

'''
文档快速生成注释的方法介绍,首先我们要用到__all__属性
在Py中使用为导出__all__中的所有类、函数、变量成员等
在模块使用__all__属性可避免相互引用时命名冲突
'''
__all__ = ['RequestRenderVoxel', 'importVoxByPath', 'importVox']

__author__ =  'Luo Dingli'
__version__=  '1.0'

#Camera
camera_distance=100
frame_counter=10
phi=0.0
theta=0.01
#render control varialbe
vertex_list=None
instancelist=[]
max_count=0
current_count=0
preview_voxel_list=None
slice_layer=-1
cube_data = (
    -1.0,-1.0,-1.0, 
    -1.0,-1.0, 1.0,
    -1.0, 1.0, 1.0,
    1.0, 1.0,-1.0, 
    -1.0,-1.0,-1.0,
    -1.0, 1.0,-1.0,
    1.0,-1.0, 1.0,
    -1.0,-1.0,-1.0,
    1.0,-1.0,-1.0,
    1.0, 1.0,-1.0,
    1.0,-1.0,-1.0,
    -1.0,-1.0,-1.0,
    -1.0,-1.0,-1.0,
    -1.0, 1.0, 1.0,
    -1.0, 1.0,-1.0,
    1.0,-1.0, 1.0,
    -1.0,-1.0, 1.0,
    -1.0,-1.0,-1.0,
    -1.0, 1.0, 1.0,
    -1.0,-1.0, 1.0,
    1.0,-1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0,-1.0,-1.0,
    1.0, 1.0,-1.0,
    1.0,-1.0,-1.0,
    1.0, 1.0, 1.0,
    1.0,-1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0,-1.0,
    -1.0, 1.0,-1.0,
    1.0, 1.0, 1.0,
    -1.0, 1.0,-1.0,
    -1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    -1.0, 1.0, 1.0,
    1.0,-1.0, 1.
)
cube_normal_data=(
        -1.0, 0.0, 0.0,-1.0, 0.0, 0.0,-1.0, 0.0, 0.0,
		 0.0, 0.0, -1.0,0.0, 0.0, -1.0,0.0, 0.0, -1.0,
		 0.0,-1.0, 0.0,0.0,-1.0, 0.0,0.0,-1.0, 0.0, 
		 0.0, 0.0, -1.0,0.0, 0.0, -1.0,0.0, 0.0, -1.0,
		-1.0, 0.0, 0.0,-1.0, 0.0, 0.0,-1.0, 0.0, 0.0,
		 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 
		 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 
		 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
		 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
		 0.0, 1.0, 0.0,0.0, 1.0, 0.0,0.0, 1.0, 0.0,
		 0.0, 1.0, 0.0,0.0, 1.0, 0.0,0.0, 1.0, 0.0,
		 0.0, 0.0, 1.0,0.0, 0.0, 1.0,0.0, 0.0, 1.0
         )

instanced_cube_data=[]
instanced_normal_data=[]
window = pyglet.window.Window()

label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

def vec(*args):
    return (GLfloat * len(args))(*args)

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    global phi,theta,camera_distance
    if buttons & mouse.LEFT:
        phi+=dx/5
        theta+=-dy/5
@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    global camera_distance
    camera_distance-=scroll_y

@window.event
def on_draw():
    window.clear()
    glClear(GL_COLOR_BUFFER_BIT)
    glMatrixMode( GL_PROJECTION )
    glLoadIdentity()
    gluPerspective( 60, 800/600, 1, 3000 )
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    glTranslatef(0, 0, -camera_distance);
    glRotatef(theta, 1, 0, 0);
    glRotatef(phi, 0, 1, 0);
    glPushMatrix()  
    glBegin(GL_QUADS)
    glColor3f(0.1,0.1,0.1)
    glVertex3f(-100,-32,-100)
    glVertex3f(-100,-32,100)
    glVertex3f(100,-32,100)
    glVertex3f(100,-32,-100)
    glEnd()
    glLightfv(GL_LIGHT0, GL_POSITION, vec(128, 64, 64, 0))
    glLightfv(GL_LIGHT1, GL_POSITION, vec(128, 128, 0, 0))
    glTranslatef(-32,-32,-32)
    glPushMatrix()
    glEnable(GL_LIGHTING)
    if len(instancelist)!=0:
        vertex_list.draw(pyglet.gl.GL_TRIANGLES)
    glDisable(GL_LIGHTING)
    if slice_layer!=-1:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(1.0,1.0,0.0,0.4)
        glBegin(GL_QUADS)
        glVertex3f(slice_layer+0.1,0,0)
        glVertex3f(slice_layer+0.1,0,64)
        glVertex3f(slice_layer+0.1,64,64)
        glVertex3f(slice_layer+0.1,64,0)
        glEnd()
    glPopMatrix()
    glPopMatrix()
    glFlush()


def InitOpenGL():
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHT0)
    #glLightfv(GL_LIGHT0, GL_SPECULAR, vec(1, 1, 1, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, vec(.1, .1, 1, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(.2, .2, .2, 1))
    glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT1, GL_DIFFUSE, vec(.4, .4, .4, 1))


def IsInnerCube(x,y,z,voxel_array):
    if (
       x-1<0 or x +1>=64
    or y-1<0 or y+1>=64
    or z-1<0 or z+1>=64
    ):
        return False
    if (voxel_array[x-1][y][z]==True and voxel_array[x+1][y][z]==True
        and voxel_array[x][y-1][z]==True and voxel_array[x][y+1][z]==True
        and voxel_array[x][y][z-1]==True and voxel_array[x][y][z+1]==True):
        return True
    return False

def CPUinstanceVoxel(voxel_array):
    global vertex_list,instancelist
    for x in range(0,64):
        for y in range(0,64):
            for z in range(0,64):
                if (voxel_array[x][y][z]==True):
                    if(IsInnerCube(x,y,z,voxel_array)==False):
                        instancelist.append((x,y,z))

def UpdateVoxel(voxel_array):
    global vertex_list,instancelist,preview_voxel_list
    global instanced_cube_data,instanced_normal_data
    instancelist.clear()
    instanced_cube_data.clear()
    instanced_normal_data.clear()
    if vertex_list!=None:
        vertex_list.delete()
    real_voxel_array = np.copy(voxel_array)
    if slice_layer!=-1: 
        real_voxel_array[slice_layer:64,...]=False

    if real_voxel_array==None:
        instancelist.append((0,0,0))
    else:
        if len(real_voxel_array.shape)==3:
            CPUinstanceVoxel(real_voxel_array)
    for instance in instancelist:
        for i in range (0,int(len(cube_data)/3)):
            instanced_cube_data.append(cube_data[i*3]+instance[0])
            instanced_cube_data.append(cube_data[i*3+1]+instance[1])
            instanced_cube_data.append(cube_data[i*3+2]+instance[2])
            instanced_normal_data.append(cube_normal_data[i*3])
            instanced_normal_data.append(cube_normal_data[i*3+1])
            instanced_normal_data.append(cube_normal_data[i*3+2])
    vertex_list = pyglet.graphics.vertex_list(int(len(instanced_cube_data)/3),('v3f', tuple(instanced_cube_data)),('n3f', tuple(instanced_normal_data)))


@window.event
def on_key_press(symbol, modifiers):
    global max_count,current_count,preview_voxel_list,slice_layer
    if preview_voxel_list==None:
        return
    if symbol == key.RIGHT:
        current_count=(current_count+1)%max_count
        print("voxel:%d in total %d \n"%(current_count,max_count))
        UpdateVoxel(preview_voxel_list[current_count])
    elif symbol == key.LEFT:
        current_count=(current_count-1)%max_count
        print("voxel:%d in total %d \n"%(current_count,max_count))
        UpdateVoxel(preview_voxel_list[current_count])
    elif symbol==key.UP:
        slice_layer=(slice_layer+1)%64
        UpdateVoxel(preview_voxel_list[current_count])
    elif symbol==key.DOWN:
        slice_layer=(slice_layer-1)%64
        UpdateVoxel(preview_voxel_list[current_count])
    elif symbol==key.R:
        slice_layer=-1
        UpdateVoxel(preview_voxel_list[current_count])


def RequestRenderVoxel(voxel_array=None):
    '''''
    该函数接收一个三维数组[64][64][64]或四维数组[模型数量][64][64][64]
    前者直接显示，后者可以用方向键的左右来在一套模型中做切换
    '''''
    global vertex_list,instancelist,preview_voxel_list
    global max_count,current_count
    InitOpenGL()
    if len(voxel_array.shape)==3:
        UpdateVoxel(voxel_array)
    elif len(voxel_array.shape)==4:
        preview_voxel_list=voxel_array
        max_count=voxel_array.shape[0]
        UpdateVoxel(voxel_array[0])
    pyglet.app.run()

def importVox(file_name,size):
    '''''
    使用该函数能够载入一个指定的导出Voxel文件中的体素,返回nparray数组[size][size][size]
    通常size设置为64
    '''''
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

def IterateFillVoxel(array,buffer_set):
    new_buffer_set=set()
    current_set=buffer_set
    bNeedNextLoop=True
    while bNeedNextLoop:
        bNeedNextLoop =False
        new_buffer_set.clear()
        for position in list(current_set):
            if array[position[0]][position[1]][position[2]]==False:
                array[position[0]][position[1]][position[2]]=True
                x=position[0]
                y=position[1]
                z=position[2]
                if x-1>=0 and array[x-1][y][z] ==False:
                    bNeedNextLoop =True
                    new_buffer_set.add((x-1,y,z))
                if x+1<64 and array[x+1][y][z] ==False:
                    bNeedNextLoop =True
                    new_buffer_set.add((x-1,y,z))
                if y-1>=0 and array[x][y-1][z] ==False:
                    bNeedNextLoop =True
                    new_buffer_set.add((x,y-1,z))
                if y+1<64 and array[x][y+1][z] ==False:
                    bNeedNextLoop=True
                    new_buffer_set.add((x,y+1,z))
                if z-1>=0 and array[x][y][z-1] ==False:
                    bNeedNextLoop=True
                    new_buffer_set.add((x,y,z-1))
                if z+1<64 and array[x][y][z+1] ==False:
                    bNeedNextLoop=True
                    new_buffer_set.add((x,y,z+1))
        current_set=new_buffer_set.copy()
    pass
def getAABB(array):
    min=[64,64,64]
    max=[0,0,0]
    for x in range(0,64):
        for y in range(0,64):
            for z in range(0,64):
                if array[x][y][z]==True:
                    if x < min[0]:
                        min[0]=x
                    if y < min[1]:
                        min[1]=y
                    if z < min[2]:
                        min[2]=z
                    if x > max[0]:
                        max[0]=x
                    if y > max[1]:
                        max[1]=y
                    if z > max[2]:
                        max[2]=z
    return max, min

def fillVoxel(array):
    buffer_set=set()
    #Calcu AABB
    max, min = getAABB(array)
    print("AABB min %d,%d,%d max %d,%d,%d \n"%(min[0],min[1],min[2],max[0],max[1],max[2]))
    buffer_set.add((int((min[0]+max[0])/2),int((min[1]+max[1])/2),int((min[2]+max[2])/2)))
    IterateFillVoxel(array,buffer_set)

def importVoxByPath(path,size):
    '''''
    使用这个函数，根据一个给定的路径，载入Voxel数据，形成一个四维数组：[模型数量][64][64][64]
    该函数会自动填充内部的体素
    '''''
    result_array=array([],dtype=bool)
    count=0
    print('start procession:\n')
    #bar=progressbar.ProgressBar(redirect_stdout=True)

    for i in os.walk(path):
	    for filename in i[2]:
                file_array=importVox(i[0]+filename,size)
                fillVoxel(file_array)
                result_array=np.append(result_array,file_array)
                file_array_fliped=np.array(file_array,copy=True)
                file_array_fliped=np.fliplr(file_array_fliped)
                result_array=np.append(result_array,file_array_fliped)
                count=count+2
    result_array=result_array.reshape((count,64,64,64))
    np.save('train_data.db',result_array)
    return result_array

def loadimage(path,size):
    result_array = zeros(shape=(size, size, size), dtype=bool)
    image = cv2.imread(path)
    matrixpic =  np.asarray(image)
    print(matrixpic.shape)
    for i in range(size):
        for j in range(size):
            #for n in range(int((int(matrixpic[i][j][0])+int (matrixpic[i][j][1])+int (matrixpic[i][j][2]))/6),size):
            a = (int(matrixpic[i][j][0]) + int(matrixpic[i][j][1]) + int(matrixpic[i][j][2])) / 6
            if a -int(a) > 0.5:
                deep = int(a)+1
            else:
                deep = int(a)
            #deep = int((int(matrixpic[i][j][0]) + int(matrixpic[i][j][1]) + int(matrixpic[i][j][2])) / 6)
            if deep != 127:
                result_array[127-deep][127-i][j]=True
    return result_array

if __name__ == '__main__':
    # if len(sys.argv)==1:
    #     ma=importVoxByPath('E:\\ObjData\\voxs_64\\',64)
    # elif len(sys.argv)==2:
    #     if sys.argv[1]=='test':
    #         ma=np.load('train_data.db.npy')
    #ma = loadimage(r'C:\Users\Administrator\Desktop\image256\image_6\images\00032500-184-targets.png', 128)  #targets
    #ma = np.load(r'D:\Paper\vox.db.npy')
    ma = np.load(r'D:\Paper\mask.db.npy')
    RequestRenderVoxel(ma)
