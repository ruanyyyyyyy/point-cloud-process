from ISS import readModelNet40
import open3d as o3d

path_chair =  "./data/chair/chair_0056.txt"
points = readModelNet40(path_chair)

# [82, 2696, 169, 1339, 7337, 9429, 1119, 8628, 6059, 4862]
iss_idx =  [1119, 8628, 2696] # 
# 2696: middle
# 7337:leg
# 1119: leg
# 8628: close to end
# 6059: end

pcd= o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.paint_uniform_color([0, 1.0, 0])
inlier_cloud = pcd.select_by_index(iss_idx)
inlier_cloud.paint_uniform_color([1.0, 0, 0])

o3d.visualization.draw_geometries([pcd, inlier_cloud])