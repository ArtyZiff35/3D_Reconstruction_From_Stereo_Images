# 3D_Reconstruction_From_Stereo_Images

This project has been written in Python and its aim is to reconstruct 3d maps of an enviroment starting from pairs of 2d stereo images.

## Introduction
The code is able to perform camera calibration for radial and tangential distortion (by capturing images of a checkerboard or by using a stored set of chessboard images), stereo rectification and image capture.
Once a pair of stero images has been captured and rectified, a disparity map is calculated using our variant of the SAD algorithm.
From there, a depth map is retrieved using some triangulations in order to correctly locate in the 3d space each pixel of the map.
Eventually, a .ply coloured point cloud is generated, which can be interpolated to obtain a uniform 3d model of the environment.

## Samples
