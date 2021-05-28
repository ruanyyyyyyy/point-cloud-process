# Compute ISS Keypoints on Armadillo
import os
import numpy as np
import struct
import open3d as o3d
import time
import copy
import tqdm

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
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(path_src, path_trg, voxel_size):
    # print(":: Load two point clouds and disturb initial pose.")
    source=o3d.geometry.PointCloud()
    source.points= o3d.utility.Vector3dVector(read_bin_velodyne(path_src))
    source.estimate_normals()
    target=o3d.geometry.PointCloud()
    target.points= o3d.utility.Vector3dVector(read_bin_velodyne(path_trg))
    target.estimate_normals()
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                            #  [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
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
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def icp_point2point(source, target, transformation):
    max_iteration = 100
    R_last = transformation[:3, :3]
    t_last = transformation[:3, 3]
    R_diff_thres = 0.5
    t_diff_thres = 0.5
    dist_thres = 5
    dists = []
    target_tree = o3d.geometry.KDTreeFlann(target) # 33, 604

    for iteration in range(max_iteration):
        dists = []
        source_temp = copy.deepcopy(source)
        source_temp = source.transform(transformation)
        # for each point p_i find the nearest neighbor in Q
       
        src_points = np.asarray(source_temp.points) # 20000, 3
        trg_points = np.asarray(target.points)
        pairs = []
        for i in range(len(src_points)):
            [k, idx, dist] = target_tree.search_knn_vector_3d(src_points[None, i,:].T, 1) # the first is the anchor point
            # reject outliers
            if dist[0] < dist_thres:
                dists.append(dist[0])
                pairs.append([i, idx[0]])
        # print(f'There are {len(pairs)} associations')
        pairs = np.asarray(pairs)
        # print(f'dists mean is {np.array(dists).mean()}')
        if pairs.shape[0] < 3:
            print("ICP failed, cannot find enough associations!")
            break

        # procrustes_transformation(A, B)
        # A, B: (length of feature, #points)
        B = trg_points[pairs[:,1]].T 
        A = src_points[pairs[:,0]].T 
        N = A.shape[1]
        L = np.identity(N) - 1.0/N * np.ones((N,1)) * np.ones((1,N))
        Ap = np.matmul(A, L)
        Bp = np.matmul(B, L)
        mediate = np.matmul(Bp, Ap.T)
        u, s, vt = np.linalg.svd(mediate)
        R = np.matmul(u, vt)
        t = 1.0/N * np.matmul((B - np.matmul(R, A)), np.ones((N,1)))
        cost = np.linalg.norm(B - (np.matmul(R, A) + t * np.ones((1, N))))
        
        transformation = np.zeros((4, 4))
        transformation[:3, :3] = R
        transformation[:3, 3] = t.squeeze()
        transformation[3,3] = 1.0

        #Check converge
        R_diff = np.linalg.norm(R - R_last)
        t_diff = np.linalg.norm(t - t_last)
        R_last = R
        t_last = t 
        if R_diff <= R_diff_thres and t_diff <= t_diff_thres:
            break

    return transformation

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

def copysign(v, s):
    # copy the sign of s to v
    if v * s < 0:
        v *= -1
    return v


if __name__=="__main__":
    root = 'registration_dataset/point_clouds'
    binfiles = os.listdir(root)
    with open('registration_dataset/reg_result.txt') as f:
        lines = f.readlines()[1:]

    result = np.zeros((len(lines), 9))
    for i in tqdm.tqdm(range(len(lines))):
        line = lines[i].split(',')
        trg, src = line[0], line[1]
        path_src = os.path.join(root, src+".bin")
        path_trg = os.path.join(root, trg+".bin")

        voxel_size = 2.0  # means 2m for the dataset
        source, target, source_down, target_down, source_fpfh, target_fpfh = \
                prepare_dataset(path_src, path_trg, voxel_size)

        result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
        # print(result_ransac)
        # draw_registration_result(source_down, target_down, result_ransac.transformation)

        # Global registration is performed on a heavily down-sampled point cloud. Then use Point-to-plane ICp to further refine the alignment
        # result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                #  voxel_size)
        # print(result_icp)
        # draw_registration_result(source, target, result_icp.transformation)
        result_icp = icp_point2point(source, target, result_ransac.transformation)
        # print(result_icp)
        # draw_registration_result(source, target, result_icp)

        tx, ty, tz, qw, qx, qy, qz = homo2tq(result_icp)
        result[i] = [trg, src, tx, ty, tz, qw, qx, qy, qz]

        

    np.savetxt('reg_result.txt', result, delimiter=',', 
                header="idx1,idx2,t_x,t_y,t_z,q_w,q_x,q_y,q_z",
                fmt='%i,%i,%f,%f,%f,%f,%f,%f,%f')