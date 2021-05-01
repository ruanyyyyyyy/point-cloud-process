import numpy as np
import scipy.spatial as spatial
import copy
import open3d as o3d
import tqdm

def readModelNet40(path):
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
        data = [[float(x) for x in line.split(",")] for line in lines]
    
    return np.array(data)



if __name__=="__main__":
    ### hyperparameters
    # the distance of a point
    radius = 0.2
    # determine if each point is key point
    lambda21 = 0.5
    lambda32 = 0.4
    # NMS radius
    non_max_radius = 0.5
    # the upper limit of iss points
    iss_count = 20


    ### three objects from modelNet40
    path_chair = "modelnet40_normal_resampled/chair/chair_0001.txt"
    path_plane = "modelnet40_normal_resampled/airplane/airplane_0001.txt"
    path_sofa = "modelnet40_normal_resampled/sofa/sofa_0001.txt"

    points = readModelNet40(path_sofa)
    point_tree = spatial.cKDTree(points[:,:3])
    cand_idx = []
    lambda3_idx = {}
    iss_idx = []

    for i in tqdm.tqdm(range(len(points))):
        pi = points[i, :3]
        neighbors = point_tree.data[point_tree.query_ball_point(pi, radius)]
        ### numerator
        nume = np.zeros((3,3))
        ### denominator
        denom = 0
        for pj in neighbors:
            weight = 1 / len(point_tree.query_ball_point(pj, radius))
            nume += weight * np.outer(pj-pi,pj-pi)
            denom += weight
        value, vector = np.linalg.eig(nume/denom)
        sortV = np.sort(value)[::-1]
        # print(sortV[1]/sortV[0], sortV[2]/sortV[1])
        if sortV[1]/sortV[0]<lambda21 and sortV[2]/sortV[1]<lambda32:
            cand_idx.append(i)
            lambda3_idx[i] = sortV[2]
    
    lambda3_idx_sort = dict(sorted(lambda3_idx.items(), key=lambda x: x[1], reverse=True))
    
    for idx in lambda3_idx_sort:
        if idx in lambda3_idx:
            iss_idx.append(idx)
        else:
            continue
        neighbor_idx = point_tree.query_ball_point(points[idx, :3], non_max_radius)
        
        for neigi in neighbor_idx:
            if neigi in lambda3_idx:
                del lambda3_idx[neigi]
        
        if len(iss_idx) > iss_count:
            break
    
    print(iss_idx)
    # iss_idx = [6210, 6515, 264, 1045, 6211, 673, 6769, 1883, 3505, 7969, 8498, 162, 815, 6126, 4688, 7551, 7600, 265, 9847, 6193, 547, 5458, 5241, 2294, 424, 8968, 1881, 1072, 2156, 4714, 6189]
    # visualization using open3d
    pcd= o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color([0, 1.0, 0])
    inlier_cloud = pcd.select_by_index(iss_idx)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd, inlier_cloud])
    
    










