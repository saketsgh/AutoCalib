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
import scipy.optimize as opt


class CalibrateCamera:

    def __init__(self, path, length, rows, cols):
        self.path = path
        self.length = length
        self.rows = rows
        self.cols = cols
        self.camera_matrix = np.array((3, 3), dtype='float32')
        self.extrinsic_params = np.array((None, 6), dtype='float32')


    def get_params(self):
        return self.intrinsic_params, self.extrinsic_params


    def initial_params_est(self):

        img_utils = ImgUtils()
        calib_utils = CalibUtils()

        ### define world coordinates of target image and scale them
        world_coord_four = np.array([[1, 6], [1, 1], [6, 6], [6, 1]]).astype(np.float32)
        world_coord_all = np.zeros(((self.rows-1)*(self.cols-1), 2), np.float32)
        world_coord_all = np.mgrid[1:self.cols , 1:self.rows].T.reshape(-1, 2)

        world_coord_all = self.length*world_coord_all
        world_coord_four = self.length*world_coord_four

        ### get the calibration images
        calib_images = img_utils.get_images(self.path)
        # img_utils.show_image(calib_images[0], "haha")

        ### find corners for all the input images
        grid_size = (self.rows-1, self.cols-1)
        corners_all_imgs = calib_utils.get_corner_pts(calib_images, grid_size)
        corners_all_imgs = np.array(corners_all_imgs)

        # for each img obtain homography and store them
        h_init = []
        for img, corners in zip(calib_images, corners_all_imgs):
            homography = calib_utils.get_homography(img, corners, world_coord_four)
            h_init.append(homography)

        # obtain the intrinsic camera parameters (alpha, beta, gamma, uc, vc)
        self.camera_matrix = calib_utils.get_camera_matrix(h_init)
        # print(self.camera_matrix)

        # calculate projected points using the equation u = HX where X are world_coord
        # and u is the corresponding image coordinates
        projected_img_coord = calib_utils.get_projected_img_coord(world_coord_all)

        # plot
        for p, img in zip(projected_img_coord, calib_images):
            # convert back to cartesian
            p = p.T
            p[:, 0] = p[:, 0]/p[:, 2]
            p[:, 1] = p[:, 1]/p[:, 2]

            img2 = img_utils.plot_points(img, p.astype(np.float32), color=(0, 0, 255))
            img_utils.show_image(img2, "img", resize=True)

    # def rectify_params(self):


def optimize_params(world_coord_all, ):


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument("--path", default="../References/AutoCalib/Calibration_Imgs/", help="provide path to images required for calibration")
    Parser.add_argument("--length", default=21.5, help="provide the length of each square of the calibration target")
    Parser.add_argument("--rows", default=7, help="provide the number of rows present in the calibration target")
    Parser.add_argument("--cols", default=10, help="provide the number of cols present in the calibration target")
    Args = Parser.parse_args()

    # aquire the arguments
    path = Args.path
    length = Args.length
    rows = Args.rows
    cols = Args.cols

    calibrate = CalibrateCamera(path, length, rows, cols)

    # calc the initial parameters
    calibrate.initial_params_est()



if __name__ == '__main__':
    main()
