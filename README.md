# 3D_Reconstruction_From_Stereo_Images

This project has been written in Python and its aim is to reconstruct 3d maps of an enviroment starting from pairs of 2d stereo images.

## Introduction
The code is able to perform camera calibration for radial and tangential distortion (by capturing images of a checkerboard or by using a stored set of chessboard images), stereo rectification and image capture.
Once a pair of stero images has been captured, rectified and pre-processed using a set of filters, a disparity map is calculated using our variant of the SAD (Sum of Absolute Differences) algorithm.
From there, a depth map is retrieved using some triangulations in order to correctly locate in the 3d space each pixel of the map.
Eventually, a .ply coloured point cloud is generated, which can be interpolated to obtain a uniform 3d model of the environment.

Here is shown more in detail the entire operative pipeline of the program:
![Pipeline](https://github.com/ArtyZiff35/3D_Reconstruction_From_Stereo_Images/blob/master/gitImages/pipeline.jpeg)

## Documentation
Detailed documentation about the whole pipeline of the program can be found in this Paper (pdf format):
https://github.com/ArtyZiff35/3D_Reconstruction_From_Stereo_Images/blob/master/documentation/3D_Model_Reconstruction_from_Stereo_2D_Images.pdf

## Samples
The following is an example result of the calibration and rectification process.
Before:
![Before rectification](https://github.com/ArtyZiff35/3D_Reconstruction_From_Stereo_Images/blob/master/gitImages/beforeRect.png)
After:
![After rectification](https://github.com/ArtyZiff35/3D_Reconstruction_From_Stereo_Images/blob/master/gitImages/afterRec.png)

Next step is the disparity map generation using our custom algorithm derived from an optimized version of SAD:
![Disparity heat-map](https://github.com/ArtyZiff35/3D_Reconstruction_From_Stereo_Images/blob/master/gitImages/bikeSmoothHeatmap.png)

Eventually, after some other processing, we end up with a depth map, and its corresponding 3d representation:
![Original Image](https://github.com/ArtyZiff35/3D_Reconstruction_From_Stereo_Images/blob/master/gitImages/loungimage.png)
![Ply 1](https://github.com/ArtyZiff35/3D_Reconstruction_From_Stereo_Images/blob/master/gitImages/loung_size1.png)
![Ply 2](https://github.com/ArtyZiff35/3D_Reconstruction_From_Stereo_Images/blob/master/gitImages/lounge_size2.png)




