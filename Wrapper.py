"""
@file    Wrapper.py
@author  Saket Seshadri Gudimetla Hanumath
@date
"""

import numpy as np
import cv2
import argparse
import math
import glob
from utils.calib_params import *
from utils.img_utils import *
import scipy.optimize as opt


class Calibrate:

    def __init__(self, length):
        self.length = length
        self.camera_params = np.array((None, 7), dtype='float64')
        self.extrinsic_params = np.array((None, 7), dtype='float64')

    def get_params(self):
        return self.camera_params, self.extrinsic_params

    def initial_params_est(self):
        # define world coordinates of target image
        target_pts = np.array([[1, 1], [4, 1], [4, 4], [1, 4]]).astype(np.float64)
        # scale them and get homogenuous versions
        target_pts *= self.length
        target_pts = get_homogenuous(target_pts)

        print(target_pts)


    # def rectify_params(self):

def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument("--path", default="../References/AutoCalib/Calibration_Imgs/", help="provide path to images required for calibration")
    Parser.add_argument("--length", default=21.5, help="provide the length of each square of the calibration target")

    Args = Parser.parse_args()
    path = Args.path
    length = Args.length

    calibrate = Calibrate(length)
    calibrate.initial_params_est()

if __name__ == '__main__':
    main()
