import cv2
import numpy as np
import glob
import os

class ImgUtils:

    def show_image(self, img, title="image", resize=False, save=False):
        if(resize):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, (img.shape[0]/2, img.shape[0]/2))
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyWindow(title)
        if(save):
            cv2.imwrite(title+".png", img)


    def get_homogenuous(self, points):
        # for row vectors
        if(points.shape[1]>1):
            hom_points = np.append(points, np.ones((points.shape[0], 1)), axis=1)

        # for col vectors
        else:
            hom_points = np.append(points, np.ones((points.shape[1], 1)), axis=0)

        return hom_points


    def get_non_homogenuous(self, points):
        points[:, 0] = points[:, 0]/points[:, 2]
        points[:, 1] = points[:, 1]/points[:, 2]

        return points[:, 0:2]


    def get_images(self, path):
        filesnames = glob.glob(path+'*.jpg')
        filesnames.sort(key=lambda f: int(filter(str.isdigit, f)))
        calib_images = []

        for filename in filesnames:
    	    img = cv2.imread(filename)
    	    height, width, layers = img.shape
    	    size = (width,height)
    	    calib_images.append(img)

        return calib_images


    def plot_points(self, img, points, color):

        for c in points:
            img = cv2.circle(img, (c[0], c[1]), 10, color, 3)

        return img
