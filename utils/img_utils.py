import cv2
import numpy as np

def show_image(img, title="FaceSwap", save=False):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyWindow(title)
    if(save):
        cv2.imwrite(title+".png", img)


def get_homogenuous(points):
    # for row vectors
    if(points.shape[1]>1):
        hom_points = np.append(points, np.ones((points.shape[0], 1)), axis=1)

    # for col vectors
    else:
        hom_points = np.append(points, np.ones((points.shape[1], 1)), axis=0)
    return hom_points
