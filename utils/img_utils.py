import cv2
import numpy as np
import glob

class ImgUtils:

    def show_image(self, img, title="FaceSwap", save=False):
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
