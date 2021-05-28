# Compute ISS Keypoints on Armadillo
import os
import numpy as np
import struct
import open3d as o3d
import time
import copy

def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])



def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(path_src, path_trg, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source=o3d.geometry.PointCloud()
    source.points= o3d.utility.Vector3dVector(read_bin_velodyne(path_src))
    source.estimate_normals()
    target=o3d.geometry.PointCloud()
    target.points= o3d.utility.Vector3dVector(read_bin_velodyne(path_trg))
    target.estimate_normals()
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)) 
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def icp(source, target, transformation, voxel_size):
    radius_feature = voxel_size * 5  
    source = source.transform(transformation)
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    # for each point p_i find the nearest neighbor in Q
    target_tree = o3d.geometry.KDTreeFlann(target_fpfh) # 33, 604
    src_points = np.asarray(source.points) # 20000, 3
    pairs = []
    for i in range(len(src_points)):
        [k, idx, _] = target_tree.search_knn_vector_xd(source_fpfh.data[:,i].reshape(33,1), 2) # the first is the anchor point
        
        pairs.append([i, idx[1]])
    pairs = np.asarray(pairs)

    # calculate A, b 
    N = len(pairs)
    q_normals = np.asarray(target.normals) # N,3
    n_ix, n_iy, n_iz = q_normals[:, 0, None], q_normals[:, 1, None], q_normals[:,2, None]
    p_points = np.asarray(source.points)
    p_ix, p_iy, p_iz = p_points[:, 0, None], p_points[:, 1, None], p_points[:, 2, None]
    q_points = np.asarray(target.points)
    q_ix, q_iy, q_iz = q_points[:, 0, None], q_points[:, 1, None], q_points[:, 2, None]

    
    A0= n_iz[pairs[:,1]] * p_iy[pairs[:,0]] - n_iy[pairs[:, 1]]*p_iz[pairs[:,0]]
    A1 = n_ix[pairs[:,1]] * p_iz[pairs[:,0]] - n_iz[pairs[:, 1]]*p_ix[pairs[:,0]]
    A2 = n_iy[pairs[:,1]] * p_iz[pairs[:,0]] - n_ix[pairs[:, 1]]*p_iy[pairs[:,0]]
    A3 = n_ix[pairs[:, 1]]
    A4 = n_iy[pairs[:, 1]]
    A5 = n_iz[pairs[:, 1]]
    A = np.concatenate((A0, A1, A2, A3, A4, A5), axis = 1)

    b = np.zeros((N, 1))
    b = n_ix[pairs[:, 1]] * q_ix[pairs[:, 1]] + n_iy[pairs[:, 1]] * q_iy[pairs[:, 1]] + n_iz[pairs[:, 1]] * q_iz[pairs[:, 1]] - n_ix[pairs[:, 1]] * p_ix[pairs[:,0]] - n_iy[pairs[:, 1]] * p_iy[pairs[:,0]] - n_iz[pairs[:, 1]] * p_iz[pairs[:,0]]

    # calculate x_hat
    x_hat = np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)), np.transpose(A))
    x_hat = np.matmul(x_hat, b) # 6,1

    # calculate R,t from x_hat
    R = [[1, x_hat[2], x_hat[1]], 
         [-x_hat[2], 1, x_hat[0]],
         [-x_hat[1], x_hat[0], 1]]
    t = [x_hat[3], x_hat[4], x_hat[5]]

    # Check converge
    dic = {"R":R, "t":t}
    transform = np.array([[1, x_hat[2], x_hat[1], x_hat[3]], 
         [-x_hat[2], 1, x_hat[0], x_hat[4]],
         [-x_hat[1], x_hat[0], 1, x_hat[5]],
         [0, 0, 0, 1]])
    return transform
        



if __name__=="__main__":
    root = 'registration_dataset/point_clouds'
    binfiles = os.listdir(root)
    with open('registration_dataset/reg_result.txt') as f:
        lines = f.readlines()

    # for i in range(len(lines)):
    #     if i < 4:
    #         continue
    if True:
        i = 10
        line = lines[i].split(',')
        src, trg = line[0], line[1]
        path_src = os.path.join(root, src+".bin")
        path_trg = os.path.join(root, trg+".bin")

        voxel_size = 2.0  # means 2m for the dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(path_src, path_trg, voxel_size)

        result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
        print(result_ransac)
        draw_registration_result(source_down, target_down, result_ransac.transformation)

        # Global registration is performed on a heavily down-sampled point cloud. Then use Point-to-plane ICp to further refine the alignment
        # result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                #  voxel_size)
        # print(result_icp)
        # draw_registration_result(source, target, result_icp.transformation)
        result_icp = icp(source, target, result_ransac.transformation, voxel_size)
        print(result_icp)
        draw_registration_result(source, target, result_icp)