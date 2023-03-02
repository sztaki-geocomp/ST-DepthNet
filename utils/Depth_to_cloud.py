import cv2 as cv
import open3d as o3d
import sys
import numpy as np
import pyrealsense2 as rs
from matplotlib import cm

#Plotting a depth image as a point cloud
def main():
    tmp = cv.imread(str(sys.argv[1]))
    tmp = cv.cvtColor(tmp, cv.COLOR_BGR2GRAY)
    tmp = np.array(tmp, dtype = np.float32)
    tmp[tmp < 20] = 0
    o3d_depth = o3d.geometry.Image(tmp)
    #Size for the point cloud
    width = 400
    height = 400
    focal_length = 212
    intrinsic_matrix = np.array([[focal_length, 0, width / 2.],
                                [0, focal_length, height / 2.],
                                [0, 0, 1.]])

    o3d_intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix[0][0], intrinsic_matrix[1][1],intrinsic_matrix[0][2],intrinsic_matrix[1][2])

    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsic_matrix)
    o3d.visualization.draw_geometries([pcd])
if __name__ == "__main__":
    main()