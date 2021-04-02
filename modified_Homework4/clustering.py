# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dbscan import DBSCAN
import open3d as o3d
import math

# set hyperparameters
tau = 0.6
N = 35
ratio = 0.5

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def show_np_pts3d(np_pts):
    pc_view = o3d.geometry.PointCloud()
    pc_view.points = o3d.utility.Vector3dVector(np_pts)
    o3d.visualization.draw_geometries([pc_view])
# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    best_cnt = 0
    best_model = None
    best_inliers = None
    best_outliers = None
    for j in range(N):
        n = [0.0, 0.0, 0.0]
        while n[0]==0.0 and n[1]==0.0 and n[2]==0.0:
            sampled_inds = np.random.randint(0, data.shape[0], size=3)
            sampled_points = data[sampled_inds] # [3, 3]
            k1 = sampled_points[0] - sampled_points[1]
            k2 = sampled_points[0] - sampled_points[2]
            n = np.cross(k1, k2)
            n = n/np.linalg.norm(n)
        
        cur_cnt = 0
        inliers_ind = []
        outliers_ind = []
        for i in range(data.shape[0]):
            cur_v = data[i] - sampled_points[0]
            dist = math.fabs(np.dot(cur_v, n))
            if dist < tau:
                cur_cnt += 1
                inliers_ind.append(i)
            else:
                outliers_ind.append(i)
        if cur_cnt > best_cnt:
            best_cnt = cur_cnt
            best_model = sampled_points
            best_inliers = inliers_ind
            best_outliers = outliers_ind
            if best_cnt/data.shape[0] > ratio:
                break
    
    segmented_cloud = data[best_outliers]
    ground_cloud = data[best_inliers]
    show_np_pts3d(segmented_cloud)
    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmented_cloud.shape[0])
    return segmented_cloud, ground_cloud

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    radius = 0.5
    min_pts = 4
    # option1: use open3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data)
    # with o3d.utility.VerbosityContextManager(
    #     o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(
    #         pcd.cluster_dbscan(eps=radius, min_points=min_pts, print_progress=True))

    # max_label = labels.max()
    # print(f"point cloud has {max_label + 1} clusters")
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd],
    #                                 zoom=0.455,
    #                                 front=[-0.4999, -0.1659, -0.8499],
    #                                 lookat=[2.1813, 2.0619, 2.0999],
    #                                 up=[0.1204, -0.9852, 0.1215])
    # clusters_index = labels
    # option2: use dbscan.py
    clus = DBSCAN(radius, min_pts)
    clus.fit(data)
    clusters_index = clus.predict()
    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    filename = "./data/000000.bin"

    origin_points = read_velodyne_bin(filename)
    segmented_points, ground_cloud = ground_segmentation(data=origin_points)
    cluster_index = clustering(segmented_points)
    ypred = np.array(cluster_index)

    planepcd = o3d.geometry.PointCloud()
    planepcd.points = o3d.utility.Vector3dVector(ground_cloud)
    c = [0,0,255]
    cs = np.tile(c,(ground_cloud.shape[0],1))
    planepcd.colors = o3d.utility.Vector3dVector(cs)
    ddraw = []
    colorset = [[222, 0, 0], [0, 224, 0], [0, 255, 255], [222, 244, 0], [255, 0, 255], [128, 0, 0]]
    for cluuus in set(ypred):

        kaka = np.where(ypred == cluuus)
        ppk = o3d.geometry.PointCloud()
        ppk.points = o3d.utility.Vector3dVector(segmented_points[kaka])

        c = colorset[cluuus % 6]
        if cluuus == -1:
            c = [0, 0, 0]

        cs = np.tile(c, (segmented_points[kaka].shape[0], 1))
        ppk.colors = o3d.utility.Vector3dVector(cs)
        ddraw.append(ppk)

    ddraw.append(planepcd)
    o3d.visualization.draw_geometries(ddraw)

    # plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
