# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as LA
import pandas as pd
import open3d as o3d
from pyntcloud import PyntCloud
from collections import defaultdict
import matplotlib.pyplot as plt

def read_oxford_bin(bin_path):
    '''
    :param path:
    :return: [x,y,z,nx,ny,nz]: 6xN
    '''
    data_np = np.fromfile(bin_path, dtype=np.float32)
    return np.transpose(np.reshape(data_np, (int(data_np.shape[0]/6), 6)))

# used by ransac_init
def find_matchings(src_features, tgt_features, 
                   src_tree = None, tgt_tree = None):
    # src_features, tgt_features: (length of feature, #points)
    if src_tree is None:
        src_tree = o3d.geometry.KDTreeFlann(src_features)
    if tgt_tree is None:
        tgt_tree = o3d.geometry.KDTreeFlann(tgt_features)
    
    N = src_features.shape[1]
    dist_thres = None
    matchings = []
    
    for src_idx in range(N):
        src_feature = src_features[:,src_idx]
        # find src_feature in tgt_tree and get their distance
        dist = None
        
        # if their distance is small enough, consider them as a pair
        if None:
            matchings.append((src_idx, tgt_idx))
    
    return np.array(matchings)
        
def procrustes_transformation(A, B):
    # A, B: (length of feature, #points)
    N = A.shape[1]
    L = None
    Ap = np.matmul(A, L)
    Bp = np.matmul(B, L)
    u, s, vt = None
    R = None
    t = None
    cost = None
    
    return R, t, cost
    
def ransac_init(src_cloud, tgt_cloud):
    """
    feature detection
    """
    src_keypoints_idxs = None
    tgt_keypoints_idxs = None
    
    """
    feature description
    """
    # (33, N) numpy array
    src_features = None
    tgt_features = None
    
    src_key_features = src_features[:,src_keypoints_idxs]
    tgt_key_features = tgt_features[:,tgt_keypoints_idxs]
    
    """
    establish correspondence
    """
    matchings = find_matchings(src_key_features, tgt_key_features)
    
    max_iter = None
    dist_thres = None
    inlier_count = 0
    max_inlier_count = 0
    best_transform = None
    
    # (3,N)
    src_points = np.asarray(src_cloud.points).T
    tgt_points = np.asarray(tgt_cloud.points).T
    
    src_keypoints = src_points[:,src_keypoints_idxs]
    tgt_keypoints = tgt_points[:,tgt_keypoints_idxs]
    
    """
    RANSAC
    """
    for i in range(max_iter):
        # random select 3 pairs from "matchings"
        matching_sample = None
        # solve R, t by Procrustes Transformation
        R, t, cost = procrustes_transformation(src_keypoints[:, matching_sample[:,0]], 
                                  tgt_keypoints[:, matching_sample[:,1]])
        
        # length N numpy array, distance of target point with transformed source point
        dists = None
        # find inlier_count by dists and dist_thres
        inlier_count = None
        
        if inlier_count >= max_inlier_count:
            max_inlier_count = inlier_count
            best_transform = (R,t)
        
    return best_transform

# used by ICP
def find_associations(src_points, tgt_tree = None):
    # src_points： (3,N)
    dist_thres = None
    associations = []
    N = src_points.shape[1]
    
    for src_idx in range(N):
        src_point = src_points[:,src_idx]
        # find src_point in tgt_tree and get their distance
        dist = None
        if dist < dist_thres:
            associations.append((src_idx, idxs[0]))
    
    return np.array(associations)
    
def ICP(src_cloud, tgt_cloud):
    # initial pose
    R_init = np.identity(3)
    t_init = np.zeros((3,1))
    # previous pose
    R_last = np.identity(3)
    t_last = np.zeros((3,1))
    # the transformation matrix to be returned
    homo_mat_total = np.identity(4)
    
    # (3,N)
    src_points = np.asarray(src_cloud.points).T
    tgt_points = np.asarray(tgt_cloud.points).T
    
    """
    initial solution using RANSAC
    """
    init_use_ransac = True
    if init_use_ransac:
        R_init, t_init = ransac_init(src_cloud, tgt_cloud)
        R_last, t_last = R_init, t_init
        # update homo_mat_total with R_init and t_init
        homo_mat_total = None
        # transform src_points using R_init and t_init
        src_points = None
    
    """
    ICP
    """
    max_iteration = None
    R_diff_thres = None
    t_diff_thres = None
    
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_cloud)
    log = defaultdict(lambda : [])
    
    for iteration in range(max_iteration):
        """
        data association
        """
        associations = find_associations(src_points, tgt_tree)
        if associations.shape[0] < 3:
            print("ICP failed, cannot find enough associations!")
            break
        
        """
        solve R and t
        """
        src_associated_points = src_points[:,associations[:,0]]
        tgt_associated_points = tgt_points[:,associations[:,1]]
        R, t, cost = procrustes_transformation(src_associated_points, tgt_associated_points)
        
        
        """
        check convergance
        """
        # http://www.boris-belousov.net/2016/12/01/quat-dist/
        R_diff = None
        t_diff = None
        R_last = R
        t_last = t
        log["R_diff"].append(R_diff)
        log["t_diff"].append(t_diff)
        
        if R_diff <= R_diff_thres and t_diff <= t_diff_thres:
            break
        else:
            # update src_points with R and t
            src_points = None
            # update homo_mat_total with R and t
            homo_mat_total = None
    
    return homo_mat_total, log

def copysign(v, s):
    # copy the sign of s to v
    if v * s < 0:
        v *= -1
    return v

def rotmat2quaternion(m):
    #https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    trace = m[0][0]+m[1][1]+m[2][2]
    qw = np.sqrt(max(0, trace+1))/2
    qx = np.sqrt(max(0, 1+m[0][0]-m[1][1]-m[2][2]))/2
    qy = np.sqrt(max(0, 1-m[0][0]+m[1][1]-m[2][2]))/2
    qz = np.sqrt(max(0, 1-m[0][0]-m[1][1]+m[2][2]))/2
    qx = copysign(qx, m[2][1]-m[1][2])
    qy = copysign(qy, m[0][2]-m[2][0])
    qz = copysign(qz, m[1][0]-m[0][1])
    return qw, qx, qy, qz

def homo2tq(homo_mat):
    tx, ty, tz = homo_mat[:3,3]
    rot_mat = homo_mat[:3,:3]
    qw, qx, qy, qz = rotmat2quaternion(rot_mat)
    return tx, ty, tz, qw, qx, qy, qz

if __name__ == "__main__":
    reg_result = np.genfromtxt("registration_dataset/reg_result.txt", delimiter=',')
    reg_result = reg_result[1:]
    
    pairs = [(int(idx1), int(idx2)) for idx1, idx2 in reg_result[:,:2]]
    
    fname = "registration_dataset/point_clouds/{}.bin"
    resfname = "result.csv"
    result = np.zeros((len(pairs), 9))
    
    salient_radius=None
    non_max_radius=None
    gamma_21=None
    gamma_32=None
    
    feature_radius=None
    
    for i, (tgt_idx, src_idx) in enumerate(pairs):
        print("=============", i, ":", src_idx, tgt_idx, "=============")
        src_fname = fname.format(src_idx)
        tgt_fname = fname.format(tgt_idx)
        src_np = read_oxford_bin(src_fname).T
        tgt_np = read_oxford_bin(tgt_fname).T
        
        point_cloud_pynt = PyntCloud(
            pd.DataFrame(src_np[:,:3], 
                         columns = ['x','y','z']))
        src_cloud = point_cloud_pynt.to_instance("open3d", mesh=False)
        src_cloud.normals = o3d.cpu.pybind.utility.Vector3dVector(src_np[:,3:])
        
        point_cloud_pynt = PyntCloud(
            pd.DataFrame(tgt_np[:,:3], 
                         columns = ['x','y','z']))
        tgt_cloud = point_cloud_pynt.to_instance("open3d", mesh=False)
        tgt_cloud.normals = o3d.cpu.pybind.utility.Vector3dVector(tgt_np[:,3:])
        
        homo_mat, log = ICP(src_cloud, tgt_cloud)
        
        # (optional) visualize the log
        
        
        tx, ty, tz, qw, qx, qy, qz = homo2tq(homo_mat)
        result[i] = [tgt_idx, src_idx, tx, ty, tz, qw, qx, qy, qz]

    np.savetxt(resfname, result, delimiter=',', 
                header="idx1,idx2,t_x,t_y,t_z,q_w,q_x,q_y,q_z",
                fmt='%i,%i,%f,%f,%f,%f,%f,%f,%f')