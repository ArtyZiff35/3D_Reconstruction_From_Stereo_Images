import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path
import plyfile as ply


def intensity_offset_and_histogram_equalization(image_l, image_r, max_range_diff=5):
    im_r = np.ndarray(shape=image_r.shape, dtype=np.uint8)
    im_r[:, :] = np.round(image_r[:, :])
    hist_r = cv2.calcHist([im_r], [0], None, [256], [0, 256])
    tot = image_r.shape[0] * image_r.shape[1]
    out = 4 * tot / 100
    range_min_r = 0
    range_max_r = 0

    # getting range f ntensities involving 94% of pixels
    count = 0
    for i in range(0, 256):
        count = count + hist_r[i]
        if count >= out:
            range_min_r = i
            break
    count = 0
    for i in range(255, -1, -1):
        count = count + hist_r[i]
        if count >= out:
            range_max_r = i
            break

    # weighted average
    count = 0
    val = 0
    for i in range(range_min_r, range_max_r + 1):
        count = count + hist_r[i]
        val = val + hist_r[i] * i
    avg_r = np.uint8(np.round(val / count))

    im_l = np.ndarray(shape=image_l.shape, dtype=np.uint8)
    im_l[:, :] = np.round(image_l[:, :])
    hist_l = cv2.calcHist([im_l], [0], None, [256], [0, 256])
    tot = image_l.shape[0] * image_l.shape[1]
    out = 4 * tot / 100
    range_min_l = 0
    range_max_l = 0

    # getting range f ntensities involving 94% of pixels
    count = 0
    for i in range(0, 256):
        count = count + hist_l[i]
        if count >= out:
            range_min_l = i
            break
    count = 0
    for i in range(255, -1, -1):
        count = count + hist_l[i]
        if count >= out:
            range_max_l = i
            break

    # weighted average
    count = 0
    val = 0
    for i in range(range_min_l, range_max_l + 1):
        count = count + hist_l[i]
        val = val + hist_l[i] * i
    avg_l = np.uint8(np.round(val / count))

    # INTENSITY OFFSET
    # if range of intensities differs only by 4 values
    if np.abs(range_max_l - range_min_l - range_max_r + range_min_r) <= max_range_diff:
        avg = (avg_r + avg_l) / 2
        offset_l = avg - avg_l
        offset_r = avg - avg_r

        for i in range(0, image_l.shape[0]):
            for j in range(0, image_l.shape[1]):
                image_r[i, j] = image_r[i, j] + offset_r
                if image_r[i, j] > 255:
                    image_r[i, j] = 255
                else:
                    if image_r[i, j] < 0:
                        image_r[i, j] = 0
                image_l[i, j] = image_l[i, j] - offset_l
                if image_l[i, j] > 255:
                    image_l[i, j] = 255
                else:
                    if image_l[i, j] < 0:
                        image_l[i, j] = 0

        # Now that the offset has been applied and the images are supposed to have the same intensity range
        # we can execute histogram equalization
        image_l = cv2.equalizeHist(np.uint8(image_l))
        image_r = cv2.equalizeHist(np.uint8(image_r))

    cv2.imwrite('./preprocessing/left/or.png', np.uint8(im_l))
    cv2.imwrite('./preprocessing/right/or.png', np.uint8(im_r))
    cv2.imwrite('./preprocessing/left/pr.png', np.uint8(image_l))
    cv2.imwrite('./preprocessing/right/pr.png', np.uint8(image_r))

    return np.float64(image_l), np.float64(image_r)


def filter_application(left_image, right_image):
    # Sharpening with filter
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    left_image = cv2.GaussianBlur(left_image, (5, 5), 0)
    right_image = cv2.GaussianBlur(right_image, (5, 5), 0)
    left_image = cv2.filter2D(left_image, -1, kernel)
    right_image = cv2.filter2D(right_image, -1, kernel)

    return left_image, right_image


def hole_filler(disparity_map, kernel_size_x, kernel_size_y):
    result = np.ndarray(disparity_map.shape, dtype=np.float32)
    result[:, :] = 0

    for i in range(0, disparity_map.shape[0]):
        for j in range(0, disparity_map.shape[1]):
            if disparity_map[i, j] == 0:
                # Try to find non zero value in kernel
                maxValue = 0
                for x in range(-kernel_size_x, kernel_size_x):
                    for y in range(-kernel_size_y, kernel_size_y):
                        actual_x = i + x
                        actual_y = j + y
                        if actual_x > 0 and actual_x < disparity_map.shape[0] and \
                                actual_y > 0 and actual_y < disparity_map.shape[1]:
                            if disparity_map[actual_x, actual_y] > maxValue:
                                maxValue = disparity_map[actual_x, actual_y]
                # If a non zero value was found, then replace zero value with max value
                result[i, j] = maxValue
            else:
                result[i, j] = disparity_map[i, j]

    return result


def downsample(image, n):
    for i in range(0, n):
        image = cv2.pyrDown(image, dstsize=(image.shape[1] // 2, image.shape[0] // 2))
    return image


def upsample(image, n):
    for i in range(0, n):
        image = cv2.pyrUp(image, dstsize=(2 * image.shape[1], 2 * image.shape[0]))
    return image
