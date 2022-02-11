import open3d as o3d

pcd = o3d.io.read_point_cloud('00.ply')
pcd_new = o3d.geometry.PointCloud.remove_non_finite_points(pcd, remove_nan= True, remove_infinite=False)
pcd.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([pcd], zoom=0.3412, front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])