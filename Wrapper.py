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
# from helper_functions.auto_calib import *
import scipy.optimize as opt


class Callibrate:

    def __init__(self):
		self.camera_params = np.array((None, 7), dtype='float64')
		self.extrinsic_params = np.array((None, 7))


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument("--path", default="../References/AutoCalib/Calibration_Imgs/", help="provide path to images required for calibration")
    Parser.add_argument("--length", default=21.5, help="provide the length of each square of the calibration target")

    Args = Parser.parse_args()
    path = Args.path
    length = Args.length


if __name__ == '__main__':
    main()
