# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import random

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, type):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    pc_array = np.asarray(point_cloud) # 1000, 3
    max_x, max_y, max_z = np.max(pc_array, axis=0) # (3,)
    min_x, min_y, min_z = np.min(pc_array, axis=0)
    Dx = (max_x-min_x)//leaf_size
    Dy = (max_y-min_y)//leaf_size 
    Dz = (max_z-min_z)//leaf_size
    N = pc_array.shape[0]
    p_h = []
    for i in range(N):
        p = pc_array[i, :]
        p = p[:, np.newaxis]
        hx = np.floor((p[0]-min_x)/leaf_size)
        hy = np.floor((p[1]-min_y)/leaf_size)
        hz = np.floor((p[2]-min_z)/leaf_size)
        h = hx + hy*Dx + hz*Dx*Dy
        p_h.append((h, p)) # add a tuple
    
    sortedph = sorted(p_h, key=lambda item: item[0])

    # centroid
    if type=="centroid":
        last = sortedph[0]
        group = []
        for each in sortedph:
            if last[0] == each[0]:
                group.append(each[1])
                last = each
            else:
                onegroup = np.hstack(group)
                onepoint = np.mean(onegroup, axis=1) # 3,1
                filtered_points.append(onepoint)
                group = [each[1]]
                last = each
    if type=="random":
        last = sortedph[0]
        group = []
        for each in sortedph:
            if last[0] == each[0]:
                group.append(each[1])
                last = each
            else:
                onegroup = len(group)
                onepoint = group[random.choice(range(onegroup))]
                filtered_points.append(onepoint[:,0])
                group = [each[1]]
                last = each

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    file_name = "data/chair/chair_0001.txt"
    point_cloud_pynt = PyntCloud.from_file(file_name, sep=",", names=["x", "y", "z", "nx", "ny", "nz"])

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_o3d.points, 0.1, "centroid")
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(filtered_cloud)
    filtered_cloud2 = voxel_filter(point_cloud_o3d.points, 0.1, "random")
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(filtered_cloud2)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([pcd1])
    o3d.visualization.draw_geometries([pcd2])

if __name__ == '__main__':
    main()
