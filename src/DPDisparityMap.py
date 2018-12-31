import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import Preprocessing


def generate_disparity_map(left_path, right_path, name, downsample_n=0, block_size=11, cmp_range=70):
    # image retrieval and gray-scale conversion
    gray_l = cv2.imread(os.path.join(left_path))
    gray_l = Preprocessing.downsample(gray_l, downsample_n)
    gray_l = np.mean(gray_l, 2)

    gray_r = cv2.imread(os.path.join(right_path))
    gray_r = Preprocessing.downsample(gray_r, downsample_n)
    gray_r = np.mean(gray_r, 2)

    gray_l, gray_r = Preprocessing.intensity_offset_and_histogram_equalization(gray_l, gray_r)
    gray_l, gray_r = Preprocessing.filter_application(gray_l, gray_r)

    finf = 1e6
    offset = block_size // 2
    cmp_range = int(cmp_range // np.power(2, downsample_n))

    height, width = gray_l.shape
    dynamic = np.ndarray((height, width - cmp_range), dtype=np.float32)
    dynamic[:, :] = 0

    disparity_cost = finf * np.ones((width - cmp_range, cmp_range), 'single')
    disparity_penalty = 0.5

    # Iterating over rows
    for i in range(0, height):

        if i % 10 == 0:
            print('row' + str(i))

        disparity_cost[:] = finf

        min_row = max(0, i - offset)
        max_row = min(height, i + offset + 1)

        for j in range(0, width - cmp_range):

            min_col = max(0, j - offset)
            max_col = min(width, j + offset + 1)

            max_d = min(cmp_range, width - max_col)

            sub_r = gray_r[min_row:max_row, min_col:max_col]

            for d in range(0, max_d):
                sub_l = gray_l[min_row:max_row, min_col + d:max_col + d]
                cost_mtx = abs(sub_r - sub_l)
                disparity_cost[j, d] = sum(sum(cost_mtx))

        optimal_indices = np.zeros(disparity_cost.shape)
        cp = disparity_cost[-1]  # takes the last-offset row of the disparity_cost matrix

        end_col_cp = cp.shape[0]
        end_col_temp = end_col_cp - 2
        temp = np.empty(shape=(7, end_col_temp), dtype=float)

        for j in range(width - cmp_range - 1, 0, -1):
            cfinf = (width - cmp_range - j + 1) * finf

            '''
            temp matrix is used to find the optimal move for each column individually
            it has: 2 columns less than the the number of disparities ( blocks evaluated)
                    each row has the SAD values, but shifted by 1 pixel proceeding with the rows

            We find the minimum value in each column of temp matrix:
                    v = vector containing minimum values (per column)
                    ix = vector containing row index of each v value
            '''

            temp[0, 0] = cfinf
            temp[0, 1] = cfinf
            temp[0, 2:end_col_temp] = cp[0:end_col_cp - 4] + 3 * disparity_penalty

            temp[1, 0] = cfinf
            temp[1, 1:end_col_temp] = cp[0:end_col_cp - 3] + 2 * disparity_penalty

            temp[2, 0:end_col_temp] = cp[0:end_col_cp - 2] + disparity_penalty

            temp[3, 0:end_col_temp] = cp[1:end_col_cp - 1]

            temp[4, 0:end_col_temp] = cp[2:end_col_cp] + disparity_penalty

            temp[5, 0:end_col_temp - 1] = cp[3:end_col_cp] + 2 * disparity_penalty
            temp[5, end_col_temp - 1] = cfinf

            temp[6, 0:end_col_temp - 2] = cp[4:end_col_cp] + 3 * disparity_penalty
            temp[6, end_col_temp - 2] = cfinf
            temp[6, end_col_temp - 1] = cfinf

            mins = np.amin(temp, axis=0)
            mins_indices = np.argmin(temp, axis=0)

            cp[0] = cfinf
            cp[1:end_col_cp - 1] = disparity_cost[j, 1:- 1] + mins
            cp[- 1] = cfinf
            # mins_indices-3 tells us the offset of the found minimum [-3 -2 -1 0 1 2 3]
            optimal_indices[j - 1, 1:- 1] = range(1, disparity_cost.shape[1] - 1) + (mins_indices - 3)

        # recover optimal route
        # Get the minimum cost for the leftmost pixel and store it in Ddynamic
        mins_indices = np.argmin(cp)
        dynamic[i, 0] = mins_indices

        # For each of the remaining pixels in this row...
        for k in range(0, width - cmp_range - 1):
            prev = int(dynamic[i, k])
            y = min(optimal_indices.shape[1], prev)
            x = max(0, y)
            dynamic[i, k + 1] = optimal_indices[k, x]

    dynamic = Preprocessing.upsample(dynamic, downsample_n)
    dynamic = dynamic * np.power(2, downsample_n)

    dynamic = Preprocessing.hole_filler(dynamic, 7, 7)

    cv2.imwrite('disparity/dp/' + str(name) + '.png', dynamic)

    # Plot the grid
    ax = sns.heatmap(dynamic)
    fig = ax.get_figure()
    fig.savefig('./disparity/heatmap/'+str(name)+'.png')

    return dynamic
