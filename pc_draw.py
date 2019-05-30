import open3d as o3d

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud("/Volumes/Feyyaz/MSc/lfd_improve_demos/open2/3/pf_60_seg.pcd")
o3d.visualization.draw_geometries([pcd])