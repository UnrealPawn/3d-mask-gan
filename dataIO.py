import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import scipy.ndimage as nd
import skimage.measure as sk
from mpl_toolkits import mplot3d

import tool.VoxelReadWrite as Voxel

try:
    import trimesh
    from stl import mesh
except:
    pass
    print('All dependencies not loaded, some functionality may not work')

LOCAL_PATH = 'E:/work/GAN/GAN/volumetric_data/'
SERVER_PATH = 'E:/work/GAN/GAN/volumetric_data/'


def getVF(path):
    raw_data = tuple(open(path, 'r'))
    header = raw_data[1].split()
    n_vertices = int(header[0])
    n_faces = int(header[1])
    vertices = np.asarray([map(float, raw_data[i + 2].split()) for i in range(n_vertices)])
    faces = np.asarray([map(int, raw_data[i + 2 + n_vertices].split()) for i in range(n_faces)])
    return vertices, faces


def plotFromVF(vertices, faces):
    input_vec = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            input_vec.vectors[i][j] = vertices[f[j], :]
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(input_vec.vectors))
    scale = input_vec.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()


def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.show()


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes(voxels, level=threshold)
    return v, f


def plotMeshFromVoxels(voxels, threshold=0.5):
    v, f = getVFByMarchingCubes(voxels, threshold)
    plotFromVF(v, f)


# def plotVoxelVisdom(voxels, visdom, title):
def plotVoxelVisdom(voxels, title):
    # print("ok")
    np.save('./results/' + title, voxels)


#  v, f = getVFByMarchingCubes(voxels)
#   visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

def plotFromVertices(vertices):
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.scatter(vertices.T[0, :], vertices.T[1, :], vertices.T[2, :])
    plt.show()


def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.Voxel(mesh, 0.5).raw
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float),
                     (sideLen / x, sideLen / y, sideLen / z),
                     order=1,
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)


def getVoxelFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


def getAll(obj='airplane', train=True, is_local=False, cube_len=64, obj_ratio=1.0):
    # objPath = SERVER_PATH + obj + '/30/'
    # if is_local:
    #    objPath = LOCAL_PATH + obj + '/30/'
    # objPath += 'train/' if train else 'test/'
    # fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
    # fileList = fileList[0:int(obj_ratio*len(fileList))]
    # volumeBatch = np.asarray([getVoxelFromMat(objPath + f, cube_len) for f in fileList],dtype=np.bool)
    if os.path.exists('vox_fill2.db.npy'):
        return np.load('vox_fill2.db.npy')
    volumeBatch = Voxel.importVoxByPath('.\\ObjData\\voxs_64\\', 64)
    return volumeBatch


# result array: [Sample number, 64,64,64]
def getVoxel():
    pass


def getBounds():
    if os.path.exists('Bound_200.db.npy'):
        bound_list = np.load('Bound_200.db.npy')
        bound_list = bound_list.reshape((int(bound_list.shape[0] * bound_list.shape[1] / 6), 6))
        mask_list = get_mask_list_by_bound(bound_list)
        return bound_list, mask_list
    print("can't find BoundList.npy in dir" + os.path.curdir)
    exit()
    return

def getMask():
    if os.path.exists('bound_64.db.npy'):
        mask_list = np.load('bound_64.db.npy')
        return mask_list
    print("can't find BoundList.npy in dir" + os.path.curdir)
    exit()
    return

def get_mask_list_by_bound(bound_list):
    mask_list = np.zeros(shape=[bound_list.shape[0], 64, 64, 64, 1], dtype=float)
    bound_list=bound_list.astype(int)
    for i in range(0, bound_list.shape[0]):
        bound = bound_list[i]
        mask_list[i, bound[0]:bound[3], bound[1]:bound[4], bound[2]:bound[5], 0] = 1.0
    return mask_list


if __name__ == '__main__':
    volumes = getAll(obj='chair', train=True, is_local=True, obj_ratio=1.0)
    # path = sys.argv[1]
    # volume = getVolumeFromOFF(path)
    # plotFromVoxels(volume)
