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
        self.intrinsic_params = np.array((1, 7), dtype='float32')
        self.extrinsic_params = np.array((None, 7), dtype='float32')


    def get_params(self):
        return self.intrinsic_params, self.extrinsic_params


    def initial_params_est(self):

        img_utils = ImgUtils()
        calib_utils = CalibUtils()

        ### define world coordinates of target image and scale them
        world_coord = np.array([[1, 6], [1, 1], [6, 6], [6, 1]]).astype(np.float32)
        world_coord *= self.length

        ### get the calibration images
        calib_images = img_utils.get_images(self.path)
        # img_utils.show_image(calib_images[0], "haha")

        ### find corners for all the input images
        grid_size = (self.rows-1, self.cols-1)
        corners_all_imgs = calib_utils.get_corner_pts(calib_images, grid_size)

        # for each img obtain homography and store them
        h_init = []
        for img, corners in zip(calib_images, corners_all_imgs):
            homography = calib_utils.get_homography(img, corners, world_coord)
            h_init.append(homography)

        # obtain the intrinsic camera parameters (alpha, beta, gamma, uc, vc)
        self.intrinsic_params = calib_utils.get_camera_intrinsics(h_init)

        # obtain the extrinsic parameters i.e. (px, py, pz, tx, ty, tz)

    # def rectify_params(self):



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
