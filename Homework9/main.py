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

#TODO: implement it 
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


if __name__=="__main__":
    root = 'registration_dataset/point_clouds'
    binfiles = os.listdir(root)
    with open('registration_dataset/reg_result.txt') as f:
        lines = f.readlines()

    # for i in range(len(lines)):
    #     if i < 4:
    #         continue
    if True:
        i = 6
        line = lines[i].split(',')
        src, trg = line[0], line[1]
        path_src = os.path.join(root, src+".bin")
        path_trg = os.path.join(root, trg+".bin")

        voxel_size = 2.0  # means 70cm for the dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(path_src, path_trg, voxel_size)

        result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
        print(result_ransac)
        draw_registration_result(source_down, target_down, result_ransac.transformation)

        # Global registration is performed on a heavily down-sampled point cloud. Then use Point-to-plane ICp to further refine the alignment
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                 voxel_size)
        print(result_icp)
        draw_registration_result(source, target, result_icp.transformation)
        