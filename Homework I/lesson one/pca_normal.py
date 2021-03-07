# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    pc_array = np.asarray(data.points)
    pc_array = np.transpose(pc_array)
    # normalize by the center
    pc_mean = np.mean(pc_array, axis=1, keepdims=True)
    pc_norm = pc_array - pc_mean
    # compute SVD
    H = np.cov(pc_norm)
    # find two principle vectors
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def draw_pca(w, v, data):
    pc_array = np.asarray(data.points)
    points = np.transpose(pc_array) # 3,1000

    new_ax = np.transpose(v[:, :2])
    pca_o3d = np.dot(new_ax, points)

    return pca_o3d

def main():
    root = "../.."
    #root = "."
    shape_names_path = os.path.join(root, "data/modelnet40_shape_names.txt")
    pc_paths = []
    with open(shape_names_path, 'r') as f:
        shapes_names = f.readlines()
        print(len(shapes_names))
        for s in shapes_names:
            s = s.strip()
            s_path = os.path.join(root, "data",s,s+"_0002.txt")
            pc_paths.append(s_path)

    names = [name.strip() for name in shapes_names]
    shape_point = {}

    pc_path = pc_paths[0]

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(pc_path, sep=",", names=["x", "y", "z", "nx", "ny", "nz"])
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云 TODO: check slack

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(point_cloud_o3d)
    point_cloud_vector = v[:, 0] #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA


    projs = draw_pca(w, v, point_cloud_o3d)
    plt.scatter(projs[0,:], projs[1,:])

    plt.show()
    

    
    # # 循环计算每个点的法向量
    # pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    # normals = []
    # # 作业2
    # # 屏蔽开始

    # # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数

    # # 屏蔽结束
    # normals = np.array(normals, dtype=np.float64)
    # # TODO: 此处把法向量存放在了normals中
    # point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
