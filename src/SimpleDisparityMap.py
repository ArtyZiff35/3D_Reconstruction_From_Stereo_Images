import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import Preprocessing


# Generates a DisparityMsp and stores it into runResults as disparity.png
def generate_disparity_map(left_path, right_path, name, downsample_n=1, block_size=11, cmp_range=70):

    # image retrieval, pyramiding and gray-scale conversion
    gray_left = cv2.imread(os.path.join(left_path))
    gray_left = Preprocessing.downsample(gray_left, downsample_n)
    gray_left = np.mean(gray_left, 2)

    gray_right = cv2.imread(os.path.join(right_path))
    gray_right = Preprocessing.downsample(gray_right, downsample_n)
    gray_right = np.mean(gray_right, 2)

    gray_left, gray_right = Preprocessing.intensity_offset_and_histogram_equalization(gray_left, gray_right)
    gray_left, gray_right = Preprocessing.filter_application(gray_left, gray_right)

    cmp_range = int(cmp_range // np.power(2, downsample_n))

    # SIMPLE BLOCK MATCHING

    # creating a matrix storing the SAD (Sum of Absolute Difference)
    # for each pixel of the image, considering block_size X block_size blocks.
    # Result in a matrix with dimension reduced by block_size-1,
    # because no padding has been considered


    row_size, col_size = gray_right.shape
    disparity_matrix = np.ndarray(shape=(row_size - block_size + 1, col_size - block_size + 1), dtype=np.float32)
    disparity_matrix[:, :] = 0
    offset = block_size // 2

    # these outer loops iterate over the right image (one pixel at a time)
    for i in range(offset, row_size - offset):

        if i % 10 == 0:
            print('row' + str(i))

        for j in range(offset, col_size - offset):

            subl = gray_left[i - offset:i + offset + 1, j - offset:j + offset + 1]
            subr = gray_right[i - offset:i + offset + 1, j - offset:j + offset + 1]
            diff = abs(subl - subr)
            c1, c2, c3 = 0, sum(sum(diff)), 0
            d = 0
            max_col = min(col_size - j - offset - 1, cmp_range)

            for k in range(0, max_col):
                start_col = j + k - offset
                end_col = j + k + offset + 1
                subl = gray_left[i - offset:i + offset + 1, start_col:end_col]
                new_dist = sum(sum(abs(subr - subl)))

                if new_dist < c2:
                    c2 = new_dist
                    d = k

            disparity_matrix[i - offset, j - offset] = d

    disparity_matrix = Preprocessing.upsample(disparity_matrix, downsample_n)
    disparity_matrix = disparity_matrix * np.power(2, downsample_n)

    disparity_matrix = Preprocessing.hole_filler(disparity_matrix, 7, 7)

    cv2.imwrite('disparity/simple/' + str(name) + '.png', disparity_matrix)

    # Plot the grid
    ax = sns.heatmap(disparity_matrix)
    fig = ax.get_figure()
    fig.savefig('./disparity/heatmap/' + str(name) + '.png')

    return disparity_matrix
