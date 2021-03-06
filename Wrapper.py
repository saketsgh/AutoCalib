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

        ### find corners for all the input images
        grid_size = (self.rows-1, self.cols-1)
        corners_all_imgs = calib_utils.get_corner_pts(calib_images, grid_size, save_image=True)
        print("found Chessboard Corners...")

        # for each img obtain homography and store them
        h_init = []
        for img, corners in zip(calib_images, corners_all_imgs):
            homography = calib_utils.get_homography(img, corners, world_coord_four)
            h_init.append(homography)
        print("\nfound homography for all images...")

        # obtain the intrinsic camera parameters (alpha, beta, gamma, uc, vc)
        intrinsic_mat, extrinsic_mat_all = calib_utils.get_all_parameters(h_init)

        return intrinsic_mat, extrinsic_mat_all, world_coord_all, corners_all_imgs


    def rectification(self, intrinsic_mat_final, k_final, world_coord):

        img_utils = ImgUtils()
        calib_utils = CalibUtils()

        world_coord_four = np.array([[1, 1], [6, 1], [1, 6], [6, 6]]).astype(np.float32)
        world_coord_four = self.length*world_coord_four

        calib_images = img_utils.get_images(self.path)
        print("\nperforming rectification of all images...")
        for i, img in enumerate(calib_images):
            distortion_params = np.array([k_final[0][0], k_final[1][0], 0, 0])
            rectified_img = cv2.undistort(img, intrinsic_mat_final, distortion_params)
            img_utils.save_image(rectified_img, "results/rectified_imgs/", "img"+str(i))

        grid_size = (self.rows-1, self.cols-1)
        rectified_imgs = img_utils.get_images("results/rectified_imgs/")
        img_coord_all = calib_utils.get_corner_pts(rectified_imgs, grid_size)

        h_final = []
        print("\nre-estimating homographies and cooresponding extrinsic parameters")
        for img, corners in zip(rectified_imgs, img_coord_all):
            homography = calib_utils.get_homography(img, corners, world_coord_four)
            h_final.append(homography)

        extrinsic_mat_final = calib_utils.get_new_extrinsic(intrinsic_mat_final, h_final)
        error_all_imgs = calib_utils.get_projection_error(world_coord, intrinsic_mat_final, extrinsic_mat_final, k_final, img_coord_all, save_image=True)

        # mean error
        mean_error = np.sum(error_all_imgs)/error_all_imgs.shape[0]

        return mean_error

def optimize_params(x0, extrinsic_mat_all, world_coord, img_coord_all):

    calib_utils = CalibUtils()
    img_utils = ImgUtils()

    # recover intrinsic matrix and lens distortion parameters
    intrinsic_mat, k_mat = calib_utils.unflatten_ak(x0)

    # calculate error over all the points in all the images
    error_all_imgs = calib_utils.get_projection_error(world_coord, intrinsic_mat, extrinsic_mat_all, k_mat, img_coord_all)

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
    print("\nobtained intrinsic and extrinsic parameters...")
    print("initial intrinsic camera matrix --> ")
    print(intrinsic_mat_init)

    # lens distortion K = [k0, k1]
    k_init = np.array([[0], [0]]).astype(np.float32)

    # param to optimize
    x0 = calib_utils.flatten_ak(intrinsic_mat_init, k_init)

    # perform optimization using least squares
    print("\nperforming optimisation using parameters obtained...")
    result = optimize.least_squares(fun=optimize_params, x0=x0, method="lm", args=[extrinsic_mat_all, world_coord, img_coord_all])
    x0_opt = result.x
    # print(result)
    intrinsic_mat_final, k_final = calib_utils.unflatten_ak(x0_opt)
    # intrinsic_mat_final = np.array( [[2.03371233e+03, -2.21824766e-01, 7.57693564e+02],
                                    # [ 0.00000000e+00, 2.02204892e+03, 1.37725480e+03],
                                    # [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # k_final = np.array([[ 0.08771432],
                        # [-0.6170109 ]])
    print("\nfinal intrinsic camera matrix and lens distortion matrix -->")
    print(intrinsic_mat_final)
    print("\n")
    print(k_final)


    # undistort the image and save them
    mean_reprojection_error = calibrate.rectification(intrinsic_mat_final, k_final, world_coord)
    print("\nmean re-projection error --> ")
    print(mean_reprojection_error)


if __name__ == '__main__':
    main()
