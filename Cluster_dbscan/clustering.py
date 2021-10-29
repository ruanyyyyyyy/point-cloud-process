# 1. Load point cloud from dataset
# 2. Filter ground point clouds from data
# 3. Extract clusters from the left point cloud

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


def ground_segmentation(data):
    """
    Filter ground points to delete
    Output:
        segmented_cloud: pc w/o ground points
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])

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

    
    pcd.points = o3d.utility.Vector3dVector(segmented_cloud)
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])

    print('origin data points num:', data.shape[0])
    print('segmented data points num:', segmented_cloud.shape[0])
    return segmented_cloud


def clustering(data):
    """
    Extract clusters from pc
    Input:
        data
    Output:
        cluster_index: 1 dim array storing the cluster index of each point cloud
    """
    radius = 0.5
    min_pts = 10
    # option1: use open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=radius, min_points=min_pts, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])
    clusters_index = labels
    # option2: use dbscan.py
    # clus = DBSCAN(radius, min_pts)
    # clus.fit(data)
    # clusters_index = clus.predict()

    return clusters_index


def plot_clusters(data, cluster_index):
    """
    Visualize pc. Each cluster has one different color.
    """
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = 'data/' 
    cat = os.listdir(root_dir)
    cat = cat[1:]
    iteration_num = len(cat)

    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        segmented_points = ground_segmentation(data=origin_points)
        cluster_index = clustering(segmented_points)

        # plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
