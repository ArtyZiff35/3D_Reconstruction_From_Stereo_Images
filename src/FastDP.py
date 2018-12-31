import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Preprocessing as p

# trying to eliminate loop over rows

def zero_left_roll(matrix, n):
    out = np.zeros(matrix.shape)
    rows, cols = matrix.shape
    out[:, 0:cols-n] = matrix[:, n:cols]
    return out


def rolling_sum_horizontal(a, n):
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    ret = ret[:, n - 1:]
    return ret


def rolling_sum_vertical(a, n):
    ret = np.cumsum(a, axis=0, dtype=float)
    ret[n:, :] = ret[n:, :] - ret[:-n, :]
    return ret[n - 1:, :]


def generate_disparity_map(left_path, right_path, name, downsample_n=1, block_size=11, cmp_range=70):

    # image retrieval and gray-scale conversion
    gray_left = cv2.imread(left_path)
    gray_left = np.mean(gray_left, 2)
    gray_left = p.downsample(gray_left, downsample_n)

    gray_right = cv2.imread(right_path)
    gray_right = np.mean(gray_right, 2)
    gray_right = p.downsample(gray_right, downsample_n)

    row_size, col_size = gray_right.shape

    matrices = np.zeros(shape=(cmp_range, row_size, col_size), dtype=float)
    matrices_hsum = np.zeros(shape=(cmp_range, row_size, col_size - block_size+1), dtype=float)
    matrices_final = np.zeros(shape=(cmp_range, row_size - block_size+1, col_size - block_size+1), dtype=float)

    # every matrix in matrices stores the pixel distances with offset = i
    for i in range(0, cmp_range):
        matrices[i] = np.abs(gray_right - zero_left_roll(gray_left, i))

    for i in range(len(matrices)):
        matrices_hsum[i] = rolling_sum_horizontal(matrices[i], block_size)
        matrices_final[i] = rolling_sum_vertical(matrices_hsum[i], block_size)

    disparity_matrix = np.argmin(matrices_final, 0).astype(np.uint8)

    disparity_matrix = p.upsample(disparity_matrix, downsample_n)

    cv2.imwrite('disparity/heatmap/' + str(name) + '.png', disparity_matrix)

    return disparity_matrix





