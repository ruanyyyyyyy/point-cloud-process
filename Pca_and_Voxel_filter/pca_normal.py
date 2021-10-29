# PCA analysis and normal calculation. Load dataset to varify.

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt


def PCA(data, correlation=False, sort=True):
    """
    Calculate PCA 
    Input:
        data: Nx3
        correlation: distinguish cov and corrcoef in np. False by default
        sort: Eigenvalues sroting. True by default
    Output:
        eigenvalues
        eigenvectors
    """
    pc_array = np.transpose(data)
    # normalize by the center
    pc_mean = np.mean(pc_array, axis=1, keepdims=True)
    pc_norm = pc_array - pc_mean
    # compute SVD
    H = np.cov(pc_norm)
    # find two principle vectors
    eigenvalues, eigenvectors = np.linalg.eigh(H)


    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def draw_pca(w, v, data):
    points = np.transpose(data) # 3,1000

    new_ax = np.transpose(v[:, :2])
    pca_o3d = np.dot(new_ax, points)

    return pca_o3d

def main():
    # root = "../.."
    root = "."
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

    # load original pc
    point_cloud_pynt = PyntCloud.from_file(pc_path, sep=",", names=["x", "y", "z", "nx", "ny", "nz"])
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    pc_array = np.asarray(point_cloud_o3d.points)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # obtain points and process them
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # get main direction using PCA
    w, v = PCA(pc_array)
    point_cloud_vector = v[:, 0] #the main direction vector
    print('the main orientation of this pointcloud is: ', point_cloud_vector)


    projs = draw_pca(w, v, pc_array)
    # plt.scatter(projs[0,:], projs[1,:])
    # plt.show()

    # calculate normal for each point in a for loop 
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    for i in range(len(point_cloud_o3d.points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 5)
        w, v = PCA(pc_array[idx, :])
        normals.append(v[:,2])
   
    normals = np.array(normals, dtype=np.float64)
    # store normals in open3d normals. when normalize, press n to show the normals!
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
