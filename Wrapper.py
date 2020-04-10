"""
CMSC733 Spring 2020: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework:1

@file    Wrapper.py
@author  Saket Seshadri Gudimetla Hanumath
"""

import numpy as np
import cv2
import argparse
import math
import glob
from utils.calibration_utils import CalibUtils
from utils.image_utils import ImgUtils
from scipy import optimize


class CalibrateCamera:

    def __init__(self, path, length, rows, cols):
        self.path = path
        self.length = length
        self.rows = rows
        self.cols = cols


    def initial_params_est(self):

        img_utils = ImgUtils()
        calib_utils = CalibUtils()

        ### define world coordinates of target image and scale them
        world_coord_four = np.array([[1, 1], [6, 1], [1, 6], [6, 6]]).astype(np.float32)
        world_coord_all = np.zeros(((self.rows-1)*(self.cols-1), 2), np.float32)
        world_coord_all = np.mgrid[1:self.rows , 1:self.cols].T.reshape(-1, 2)

        world_coord_all = self.length*world_coord_all
        world_coord_four = self.length*world_coord_four

        ### get the calibration images
        calib_images = img_utils.get_images(self.path)
        # img_utils.show_image(calib_images[0], "haha")

        ### find corners for all the input images
        grid_size = (self.rows-1, self.cols-1)
        corners_all_imgs = calib_utils.get_corner_pts(calib_images, grid_size)
        # corners_all_imgs = np.array(corners_all_imgs)

        # for each img obtain homography and store them
        h_init = []
        for img, corners in zip(calib_images, corners_all_imgs):
            homography = calib_utils.get_homography(img, corners, world_coord_four)
            h_init.append(homography)

        # obtain the intrinsic camera parameters (alpha, beta, gamma, uc, vc)
        intrinsic_mat, extrinsic_mat_all = calib_utils.get_all_parameters(h_init)

        return intrinsic_mat, extrinsic_mat_all, world_coord_all, corners_all_imgs



def optimize_params(x0, extrinsic_mat_all, world_coord, img_coord_all):

    calib_utils = CalibUtils()
    img_utils = ImgUtils()

    # recover intrinsic matrix and lens distortion parameters
    intrinsic_mat, k_mat = calib_utils.un_flatten_ak(x0)

    # calculate error over all the points in all the images
    error_all_imgs = 0

    # obtain the coordinates of image center
    uc = intrinsic_mat[0][2]
    vc = intrinsic_mat[1][2]

    calib_images = img_utils.get_images("../References/AutoCalib/Calibration_Imgs/")
    error_all_imgs = np.array([])

    # error_all_imgs = calib_utils.get_projection_error(world_coord, intrinsic_mat, extrinsic_mat_all, img_coord_all)
    i = 0
    for extrinsic_mat, img_coord in zip(extrinsic_mat_all, img_coord_all):

        U, X = calib_utils.get_projected_img_coord(world_coord, intrinsic_mat, extrinsic_mat)

        u = U[:, 0].reshape((X.shape[0], 1))
        v = U[:, 1].reshape((X.shape[0], 1))
        x = X[:, 0].reshape((X.shape[0], 1))
        y = X[:, 1].reshape((X.shape[0], 1))

        x_y = (x**2) + (y**2)
        lens_dist = k_mat[0][0]*(x_y) + k_mat[1][0]*(x_y**2)

        # projected image coordinates with distortion
        u_h = u + np.multiply((u-uc), lens_dist)
        v_h = v + np.multiply((v-vc), lens_dist)

        # observed image coordinates
        u_img = img_coord[:, 0].reshape((X.shape[0], 1))
        v_img = img_coord[:, 1].reshape((X.shape[0], 1))

        # for debuging projected points
        # img = img_utils.plot_points(calib_images[i], img_coord.astype(np.float32), color=(0, 0, 0))
        # img = img_utils.plot_points(img, U.astype(np.float32), color=(0, 0, 255))
        # img_utils.show_image(img, "img", resize=True)

        # l2 norm or sum of squared diff
        error_all_pts = (u_img-u_h)**2 + (v_img-v_h)**2
        # error_all_pts = np.sum(error_all_pts, axis=0)
        error_all_imgs = np.append(error_all_imgs, error_all_pts)
        i += 1

    return error_all_imgs


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument("--path", default="../References/AutoCalib/Calibration_Imgs/", help="provide path to images required for calibration")
    Parser.add_argument("--length", default=21.5, help="provide the length of each square of the calibration target")
    Parser.add_argument("--rows", default=10, help="provide the number of rows present in the calibration target")
    Parser.add_argument("--cols", default=7, help="provide the number of cols present in the calibration target")
    Args = Parser.parse_args()

    # aquire the arguments
    path = Args.path
    length = Args.length
    rows = Args.rows
    cols = Args.cols
    calib_utils = CalibUtils()
    calibrate = CalibrateCamera(path, length, rows, cols)

    # calc the initial parameters
    intrinsic_mat_init, extrinsic_mat_all, world_coord, img_coord_all = calibrate.initial_params_est()

    # lens distortion K = [k0, k1]
    k_init = np.array([[0], [0]]).astype(np.float32)

    # param to optimize
    x0 = calib_utils.flatten_ak(intrinsic_mat_init, k_init)

    # optimize_params(x0, extrinsic_mat_all, world_coord, img_coord_all)

    result = optimize.least_squares(fun=optimize_params, x0=x0, method="lm", args=[extrinsic_mat_all, world_coord, img_coord_all])
    x0_opt = result
    print(x0_opt)


if __name__ == '__main__':
    main()
