import numpy as np
import cv2
import glob
import time


class CalibrationRectification:

    def __init__(self):

        self.ret = None
        self.mtx_l = None
        self.dist_l = None
        self.mtx_r = None
        self.dist_r = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.roi_l = None
        self.roi_r = None
        self.mapx_l = None
        self.mapy_l = None
        self.mapx_r = None
        self.mapy_r = None
        self.R_l = None
        self.R_r = None
        self.P_l = None
        self.P_r = None
        self.Q = None
        self.size = None

    def calibrate(self):

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints_l = []  # 2d points in image plane.
        imgpoints_r = []  # 2d points in image plane.

        images_l = glob.glob('calibration/original/left/*.png')
        images_r = glob.glob('calibration/original/right/*.png')

        for i in range(len(images_l)):

            path_l = images_l.pop()
            path_r = images_r.pop()

            im_l = cv2.imread(path_l)
            im_r = cv2.imread(path_r)

            gray_l = cv2.cvtColor(im_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(im_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 6), None, flags=cv2.CALIB_CB_FILTER_QUADS)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 6), None, flags=cv2.CALIB_CB_FILTER_QUADS)

            # If found, add object points, image points (after refining them)
            if ret_l & ret_r:
                objpoints.append(objp)

                corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

                imgpoints_l.append(corners2_l)
                imgpoints_r.append(corners2_r)

                # Draw corners and save image
                img_drawn_l = cv2.drawChessboardCorners(im_l, (9, 6), corners2_l, ret_l)
                img_drawn_r = cv2.drawChessboardCorners(im_r, (9, 6), corners2_r, ret_r)
                cv2.imwrite('calibration/drawnCorners/left/' + str(i) + ".png", img_drawn_l)
                cv2.imwrite('calibration/drawnCorners/right/' + str(i) + ".png", img_drawn_r)

        cv2.destroyAllWindows()

        # this function returns the camera matrix, distortion coefficients, rotation and translation vectors etc.
        # this step is necessary to have initial camera matrices, to be furtherly optimized by StereoCalibrate later
        ret_l, self.mtx_l, self.dist_l, self.rvecs_l, self.tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l,
                                                                                         gray_l.shape[::-1], None, None)
        ret_r, self.mtx_r, self.dist_r, self.rvecs_r, self.tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r,
                                                                                         gray_r.shape[::-1], None, None)

        # todo decide whether to include 'getoptimalnewcameramatrix' [Gabry don't think is necessary]

        self.size = gray_l.shape[::-1]

        ''' 
        setting flags for StereoCalibration
        * CV_CALIB_FIX_INTRINSIC            camera matrices are used as they are
        * CV_CALIB_USE_INTRINSIC_GUESS      camera matrices are used as a starting point to optimize further the 
                                            intrinsic and distortion parameters for each camera and will be set to the 
                                            refined values on return from cvStereoCalibrate()
        '''
        flags = 0
        # flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        (ret, self.mtx_l, self.dist_l, self.mtx_r,
         self.dist_r, self.R, self.T, self.E, self.F)= cv2.stereoCalibrate( objpoints,
                                                                            imgpoints_l,
                                                                            imgpoints_r,
                                                                            self.mtx_l,
                                                                            self.dist_l,
                                                                            self.mtx_r,
                                                                            self.dist_r,
                                                                            self.size,
                                                                            criteria=stereocalib_criteria,
                                                                            flags=flags)  #todo check that R, T, E, F can be safely skipped for input

        # STEREO RECTIFICATION
        (self.R_l, self.R_r, self.P_l, self.P_r, self.Q, self.roi1, self.roi2) = cv2.stereoRectify(self.mtx_l,
                                                                              self.dist_l,
                                                                              self.mtx_r,
                                                                              self.dist_r,
                                                                              self.size,
                                                                              self.R,
                                                                              self.T,
                                                                              self.R_l, self.R_r, self.P_l, self.P_r,
                                                                              flags,
                                                                              alpha=0)

        self.mapx_l, self.mapy_l = cv2.initUndistortRectifyMap(self.mtx_l, self.dist_l, self.R_l, self.P_l, self.size,
                                                               cv2.CV_32FC1)
        self.mapx_r, self.mapy_r = cv2.initUndistortRectifyMap(self.mtx_r, self.dist_r, self.R_r, self.P_r, self.size,
                                                               cv2.CV_32FC1)

        # storing the principal points of both cameras
        (self.principal_xl, self.principal_yl) = (int(self.mtx_l[0, 2]), int(self.mtx_l[1, 2]))
        (self.principal_xr, self.principal_yr) = (int(self.mtx_r[0, 2]), int(self.mtx_r[1, 2]))

    def draw_comparison(self, im_l, im_r, name):
        # Draw horizontal lines on images
        for j in range(5, self.size[1], 40):
            cv2.line(im_l, (0, j), (self.size[0], j), (0, 0, 255), 1)
            cv2.line(im_r, (0, j), (self.size[0], j), (0, 0, 255), 1)

        # create unique image corresponding to the alignment of left and right
        aligned = np.hstack((im_l, im_r))
        cv2.imwrite('./results/undistortedRectified/aligned/' + str(name) + '.png', aligned)

    def remap(self, left_image, right_image, name):
        # undistorting the images witht the calculated undistortion map
        result_l = np.ndarray(shape=([self.size[1], self.size[0], 3]), dtype=np.uint8)
        result_l[:, :, 0] = cv2.remap(left_image[:, :, 0], self.mapx_l, self.mapy_l, cv2.INTER_LINEAR)
        result_l[:, :, 1] = cv2.remap(left_image[:, :, 1], self.mapx_l, self.mapy_l, cv2.INTER_LINEAR)
        result_l[:, :, 2] = cv2.remap(left_image[:, :, 2], self.mapx_l, self.mapy_l, cv2.INTER_LINEAR)

        # undistorting the images witht the calculated undistortion map
        result_r = np.ndarray(shape=([self.size[1], self.size[0], 3]), dtype=np.uint8)
        result_r[:, :, 0] = cv2.remap(right_image[:, :, 0], self.mapx_r, self.mapy_r, cv2.INTER_LINEAR)
        result_r[:, :, 1] = cv2.remap(right_image[:, :, 1], self.mapx_r, self.mapy_r, cv2.INTER_LINEAR)
        result_r[:, :, 2] = cv2.remap(right_image[:, :, 2], self.mapx_r, self.mapy_r, cv2.INTER_LINEAR)

        cv2.imwrite('./remap/remapped/left/' + str(name) + '.png', result_l)
        cv2.imwrite('./remap/remapped/right/' + str(name) + '.png', result_r)

        self.draw_comparison(result_l, result_r, name)

    def remap_from_capture(self, count, l_camera_port, r_camera_port):

        left_camera_port = l_camera_port
        right_camera_port = r_camera_port

        left_camera = cv2.VideoCapture(left_camera_port)
        time.sleep(0.5)
        right_camera = cv2.VideoCapture(right_camera_port)
        time.sleep(0.5)

        # MAIN CAPTURING LOOP, storing images in the folder
        counter = 0
        while counter < count:
            cv2.waitKey(1000)
            print('Capturing...')

            # Capturing image from camera
            left_return_value, left_image = left_camera.read()
            right_return_value, right_image = right_camera.read()

            cv2.imwrite('./remap/captures/left/' + str(counter) + '.png', left_image)
            cv2.imwrite('./remap/captures/right/' + str(counter) + '.png', right_image)

            self.remap(left_image, right_image, counter)

            counter += 1

    def remap_from_path(self, path):

        images_l = glob.glob(path + '/left/*.png')
        images_r = glob.glob(path + '/right/*.png')

        for i in range(len(images_l)):

            print('processing images ' + str(i))

            fname_l = images_l.pop()
            fname_r = images_r.pop()

            im_l = cv2.imread(fname_l)
            im_r = cv2.imread(fname_r)

            # This portion of code set to 255 all pixels surrounding the principal point detected by the calibration
            # step. This point is a feature of each camera, and it is related to how the resulting depth must be computed.
            # todo      check if is correct to use principal_x as column index and principal_y as row one
            # todo      it look correct considering the vale retrieved from the camera matrices, because approximately
            # todo      corresponding to the center of the original image
            '''
            im_l[self.principal_yl - 5:self.principal_yl + 5, self.principal_xl - 5:self.principal_xl + 5] = 255
            im_r[self.principal_yr - 5:self.principal_yr + 5, self.principal_xr - 5:self.principal_xr + 5] = 255
            '''

            self.remap(im_l, im_r, i)



