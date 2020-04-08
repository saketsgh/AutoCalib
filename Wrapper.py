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
from utils.calib_params import *
from utils.img_utils import ImgUtils
import scipy.optimize as opt


class CalibrateCamera:

    def __init__(self, path, length, rows, cols):
        self.path = path
        self.length = length
        self.rows = rows
        self.cols = cols
        self.camera_params = np.array((None, 7), dtype='float32')
        self.extrinsic_params = np.array((None, 7), dtype='float32')

    def get_params(self):
        return self.camera_params, self.extrinsic_params

    def initial_params_est(self):

        image_utils = ImgUtils()

        # define world coordinates of target image
        target_pts = np.array([[1, 1], [4, 1], [4, 4], [1, 4]]).astype(np.float32)
        # scale them and get homogenuous versions
        target_pts *= self.length
        target_pts = image_utils.get_homogenuous(target_pts)

        # get the calibration images
        calib_images = image_utils.get_images(self.path)
        # image_utils.show_image(calib_images[0], "haha")

        # find corners
        gray = cv2.cvtColor(calib_images[0], cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if(ret==True):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            img = cv2.drawChessboardCorners(calib_images[0], (7,6), corners2,ret)
            image_utils.show_image(img, "corners", save=True)

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
