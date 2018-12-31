import CalibrationRectification
import SubDisparityMap as dm
import SimpleDisparityMap as odm
import DPDisparityMap as dpdm
import FastDP as fdm
import RealDepth as rd
import time

import matplotlib.pyplot as plt
import seaborn as sns

'''instantiates class'''
c = CalibrationRectification.CalibrationRectification()

'''calibrates from images stored on file
the function will look for "calibration/original" containing two folders "left" and "right"
images in the left / right folder must be stored with the same relative order.
(calling them 0.png, 1.png, ... might be a good idea)'''

print("Starting Calibration")
# c.calibrate()
print("Calibration Completed\n")

'''
we can pass the image pairs we want to rectify by activating the camera (from_capture)
or by specifying a folder (from_path). 

from_camera parameters: 
    count: how many picture pairs we would like to be taken
    l/r_camera port: the camera ports for the cameras we want to use
'''

print("Starting Remap Procedure")

# TODO: CHECK THAT THESE WORK PROPERLY
# c.remap_from_capture(count=1, l_camera_port=0, r_camera_port=2)
# c.remap_from_path(path='./results/original')

print("Starting Remap Completed\n")

'''
From now on functions will take a single image pair, you need a for loop if you want to cycle over several images.
Right now we are specifying the paths to a single pair of images, just as an example.

INPUT:
    left_path
    right_path
    name            name used to save the result (for example the counter if using a loop)
    downsample_n:   how many times to apply gaussian down-sampling with factor 2
    block_size      (optional, default = 11)
    cmp_range       (optional, default = 70)
    
OUTPUT:
    disparity: matrix with size: input_height * (input_width-cmp_range) * 1
    
'''
path_l = 'remap/remapped/left/_bike_l.png'
path_r = 'remap/remapped/right/_bike_r.png'
name = 'bike'
start_time = time.time()
# disparity = dm.generate_disparity_map(left_path=path_l, right_path=path_r, downsample_n=1)
print("Starting Disparity Map Calculation")
disparity = fdm.generate_disparity_map(left_path=path_l, right_path=path_r, name=name, downsample_n=0)
print("Disparity Map Calculation Ended\n")
print("--- %s seconds ---" % (time.time() - start_time))

'''
We now generate the depth map using "generate_depth_map" from the RealDepth file
NOTE: RealDepth has been parametrized to deal with SubDisparityMap output and will only work with 
disparities outputted from that class
'''

print("Starting Depth Map Calculation")
model3D_matrix = rd.generate_depth_map(disparity)
print("Depth Map Calculation completed\n")

print("Starting Conversion to PLY")
rd.convert_to_ply(disparity=disparity, name=name, image_path=path_r, cmp_range=70)
print("Conversion to PLY Completed\n")

print("Starting Model Visualization")
rd.visualize_model('ply/' + str(name) + '.ply')



