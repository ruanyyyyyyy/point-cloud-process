# For point clouds in dataset, build trees and search, including kdtree and octree. Estimate the running time.

import random
import math
import numpy as np
import time
import os
import struct

import octree 
import kdtree 
from result_set import KNNResultSet, RadiusNNResultSet

from scipy.spatial import KDTree

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
    return np.asarray(pc_list, dtype=np.float32).T

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    # root_dir = '../kitti' 
    # cat = os.listdir(root_dir)
    iteration_num = 1 # len(cat)

    print("octree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    # for i in range(iteration_num):
    filename = '../000000.bin' #os.path.join(root_dir, cat[i])
    db_np = read_velodyne_bin(filename)

    begin_t = time.time()
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    construction_time_sum += time.time() - begin_t

    query = db_np[0,:]

    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    octree.octree_knn_search(root, db_np, result_set, query)
    knn_time_sum += time.time() - begin_t

    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    octree.octree_radius_search_fast(root, db_np, result_set, query)
    radius_time_sum += time.time() - begin_t

    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000/iteration_num,
                                                                     knn_time_sum*1000/iteration_num,
                                                                     radius_time_sum*1000/iteration_num,
                                                                     brute_time_sum*1000/iteration_num))

    print("scipy spatial kdtree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    # for i in range(iteration_num):
    filename = '../000000.bin' # os.path.join(root_dir, cat[i])
    db_np = read_velodyne_bin(filename)

    begin_t = time.time()
    root = KDTree(db_np, leaf_size)
    construction_time_sum += time.time() - begin_t

    query = db_np[0,:]

    begin_t = time.time()
    result_set = root.query(query, k)
    
    knn_time_sum += time.time() - begin_t

    begin_t = time.time()
    result_set = root.query_ball_point(db_np, radius)
    radius_time_sum += time.time() - begin_t

    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                    knn_time_sum * 1000 / iteration_num,
                                                                    radius_time_sum * 1000 / iteration_num,
                                                                    brute_time_sum * 1000 / iteration_num))

    print("kdtree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    # for i in range(iteration_num):
    filename = '../000000.bin' # os.path.join(root_dir, cat[i])
    db_np = read_velodyne_bin(filename)

    begin_t = time.time()
    root = kdtree.kdtree_construction(db_np, leaf_size)
    construction_time_sum += time.time() - begin_t

    query = db_np[0,:]

    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    kdtree.kdtree_knn_search(root, db_np, result_set, query)
    knn_time_sum += time.time() - begin_t

    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    kdtree.kdtree_radius_search(root, db_np, result_set, query)
    radius_time_sum += time.time() - begin_t

    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                    knn_time_sum * 1000 / iteration_num,
                                                                    radius_time_sum * 1000 / iteration_num,
                                                                    brute_time_sum * 1000 / iteration_num))




if __name__ == '__main__':
    main()