import open3d
import numpy as np
import cv2
import plyfile
import os.path
import RealDepth as rd


def generate_depth_map(disparity_matrix, image_path, cmp_range=70, percent=15):

    # reading image
    image = cv2.imread(os.path.join(image_path))

    original_height, original_width = image.shape

    rows = disparity_matrix.shape[0]
    cols = disparity_matrix.shape[1]

    # depth of pixels interpolating thre measured points with a cubic function where:
    # Y = a*X^3 + b*X^2 + c*X + d where Y is the depth measure in meters and X is the disparity value
    # Last two pair of x,y coordinates have been arbitrarily added to better manipulate the resulting
    # interpolating function. The first three pairs have been measured on field
    x = np.array([5, 32, 50, 100, 200])
    y = np.array([9, 2, 1.5, 1, 0])
    coeff = np.polyfit(x, y, 3)

    # These are the measures retrieved from the rectangles of the field of vision of the
    # right camera at two different distances. We are going to use them to generate
    # the points belonging to the perspective lines passing through each pixel of the image plane
    height_a = 0.9
    width_a = 1.2
    depth_a = 1.5
    height_b = 0.5
    width_b = 0.67
    depth_b = 0.75

    # computing the pixels offset rescaling on the original image size
    x_offset_a = width_a / original_width
    y_offset_a = height_a / original_height
    x_offset_b = width_b / original_width
    y_offset_b = height_b / original_height

    # filling the points of a and b rectangles with the 3D coordinates
    points_a = np.ndarray((rows, cols, 3), dtype=np.float64)
    points_b = np.ndarray((rows, cols, 3), dtype=np.float64)
    for i in range(0, rows):
        for j in range(0, cols):

            points_a[i, j, 0] = j * x_offset_a
            points_a[i, j, 1] = i * y_offset_a
            points_a[i, j, 2] = depth_a

            points_b[i, j, 0] = j * x_offset_b
            points_b[i, j, 1] = i * y_offset_b
            points_b[i, j, 2] = depth_b

    points_a[:, :, 0] = points_a[:, :, 0] - points_a[int(rows/2), int(cols/2), 0]
    points_a[:, :, 1] = points_a[:, :, 1] - points_a[int(rows/2), int(cols/2), 1]

    points_b[:, :, 0] = points_b[:, :, 0] - points_b[int(rows/2), int(cols/2), 0]
    points_b[:, :, 1] = points_b[:, :, 1] - points_b[int(rows/2), int(cols/2), 1]

    # computing the equations for lines passing through each pixel of the image plan and the focal point
    points = np.ndarray((disparity_matrix.shape[0], disparity_matrix.shape[1], 3))
    diff = np.ndarray(3)
    for i in range(0, rows):
        for j in range(0, cols):

            z = coeff[0]*np.power(disparity_matrix[i, j], 3) + coeff[1]*np.power(disparity_matrix[i, j], 2) + coeff[2]*disparity_matrix[i, j] + coeff[3] + 0.5

            diff[0] = points_a[i, j, 0] - points_b[i, j, 0]
            diff[1] = points_a[i, j, 1] - points_b[i, j, 1]
            diff[2] = points_a[i, j, 2] - points_b[i, j, 2]
            val = (z - points_b[i, j, 2]) / diff[2]

            points[i, j, 0] = points_b[i, j, 0] + val * diff[0]
            points[i, j, 1] = points_b[i, j, 1] + val * diff[1]
            points[i, j, 2] = z

    return points

'''
Deprecated function left here for demonstration and comparison purposes
'''
def generate_depth_map_reprojected(disparity_matrix, Q):
    rows = disparity_matrix.shape[0]
    cols = disparity_matrix.shape[1]

    # computing depth with reprojection
    pointsReprojected = cv2.reprojectImageTo3D(disparity_matrix, Q)

    # reprojected points are dimensionally scaled
    points = np.ndarray((disparity_matrix.shape[0], disparity_matrix.shape[1], 3))
    for i in range(0, rows):
        for j in range(0, cols):
            points[i, j, 0] = pointsReprojected[i, j, 0]
            points[i, j, 1] = pointsReprojected[i, j, 1]
            points[i, j, 2] = pointsReprojected[i, j, 2]

    return points


def convert_to_ply(disparity, model_3d, name, image_path, cmp_range, percent=15, downsample_n=0):

    # setting min and max accepted range
    min_range = int(cmp_range * percent // 100)
    max_range = int(cmp_range * (100 - percent) // 100)

    file_path = 'ply/' + str(name) + '.ply'

    # reading image
    image = cv2.imread(os.path.join(image_path))

    # resizing it to be consistent to the measure of the disparity map
    rows = image.shape[0]
    cols = image.shape[1]
    for i in range(0, downsample_n):
        rows = rows // 2
        cols = cols // 2
    cols = cols - int(cmp_range//np.power(2, downsample_n))
    rows = rows * np.power(2, downsample_n)
    cols = cols * np.power(2, downsample_n)
    image = image[0:rows, 0:cols]

    rows = image.shape[0]
    cols = image.shape[1]

    vertex = np.zeros((rows * cols),
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('blue', 'u1'), ('green', 'u1'), ('red', 'u1')])

    # Filling in the vertex matrix with space coordinates and RGB values for each point
    index = 0
    for row in range(0, rows-1):
        for column in range(0, cols-1):
            # Filtering outliers
            if min_range <= disparity[row, column] <= max_range:
                vertex[index] = (model_3d[row, column, 0],
                                 model_3d[row, column, 1],
                                 model_3d[row, column, 2],
                                 image[row, column, 0],
                                 image[row, column, 1],
                                 image[row, column, 2])
                index = index + 1

    el = plyfile.PlyElement.describe(vertex, 'vertex')
    plyfile.PlyData([el], text=True).write(file_path)


def visualize_model(file_path):

    pcd = open3d.read_point_cloud(file_path)
    open3d.draw_geometries([pcd])

