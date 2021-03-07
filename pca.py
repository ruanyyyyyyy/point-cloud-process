import os
import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud

def pca_op(pc_array):
    # normalize by the center
    pc_mean = np.mean(pc_array, axis=0)
    # compute SVD

    # find two principle vectors




if __name__=="__main__":
    shape_names_path = "data/modelnet40_shape_names.txt"
    pc_paths = []
    with open(shape_names_path, 'r') as f:
        shapes_names = f.readlines()
        print(len(shapes_names))
        for s in shapes_names:
            s = s.strip()
            s_path = os.path.join("data",s,s+"_0002.txt")
            pc_paths.append(s_path)

    names = [name.strip() for name in shapes_names]
    shape_point = {}
    for i in range(1):
        pc_path = pc_paths[i]
        point_cloud_pynt = PyntCloud.from_file(pc_path, sep=",", names=["x", "y", "z", "nx", "ny", "nz"])
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
        pc_array = np.asarray(point_cloud_o3d.points)


    eigen_value, eigen_vector = pca_op(pc_array)




