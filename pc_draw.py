import open3d as o3d

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("/Volumes/Feyyaz/MSc/lfd_improve_demos/close/1/pf_48_seg.pcd")
o3d.visualization.draw_geometries([pcd])