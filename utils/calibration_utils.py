import cv2
import os
import numpy as np
from image_utils import ImgUtils


debug = False

class CalibUtils:

    def __init__(self):
        self.intrinsic_mat = np.zeros((3, 3)).astype(np.float32)
        # self.extrinsic_mat_all = np.zeros((None, 3, 4)).astype(np.float32)


    def get_corner_pts(self, calib_images, grid_size):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_all_imgs = []

        if not os.path.exists("results/"):
            os.makedirs("results/")

        for i, img in enumerate(calib_images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

            if(ret==True):
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                img = cv2.drawChessboardCorners(img, grid_size, corners2, ret)
                corners2 = corners2.reshape((corners.shape[0], -1))
                # self.show_image(img, "corners", save=True)
                # cv2.imwrite("results/corners_img_"+str(i)+".png", img)
                corners_all_imgs.append(corners2)

        return corners_all_imgs


    def get_homography(self, img, img_corners, world_coord):

        img_coord = []
        img_utils = ImgUtils()

        # choosing four points for finding homography
        img_coord.append(img_corners[0])
        img_coord.append(img_corners[5])
        img_coord.append(img_corners[30])
        img_coord.append(img_corners[35])

        img_coord = np.array(img_coord, np.float32)

        homography = cv2.getPerspectiveTransform(world_coord, img_coord)

        # to show which points where chosen
        if(debug):
            for c in img_coord:

                # font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # org
                org = (c[0],c[1])

                # fontScale
                fontScale = 1

                # Blue color in BGR
                color = (0, 0, 0)

                # Line thickness of 2 px
                thickness = 2
                img = cv2.putText(img, str(org), org, font,
                                   fontScale, color, thickness, cv2.LINE_AA)

            img_utils.show_image(img, "coord", resize=True)

        return homography


    def v_pq_h(self, homography, p, q):

        h = homography
        v_pq = np.array( [[  (h[0][p]*h[0][q]),
                            ((h[0][p]*h[1][q]) + (h[1][p]*h[0][q])),
                            (h[1][p]*h[1][q]),
                            ((h[2][p]*h[0][q]) + (h[0][p]*h[2][q])),
                            ((h[2][p]*h[1][q]) + (h[1][p]*h[2][q])),
                            (h[2][p]*h[2][q])]] )

        return v_pq


    def compute_intrinsic_params(self, h_init):

        v_mat = []
        for h in h_init:
            v01_h = self.v_pq_h(h, 0, 1)
            v00_h = self.v_pq_h(h, 0, 0)
            v11_h = self.v_pq_h(h, 1, 1)
            vh = np.vstack((v01_h, (v00_h-v11_h)))
            v_mat.append(vh)

        v_mat = np.array(v_mat)
        v_mat = v_mat.reshape((-1, v_mat.shape[2]))

        # apply SVD
        u, s, vt = np.linalg.svd(v_mat)
        # extract the last row and that'll give the B matrix
        b = vt[-1]

        # obtain the intric parameters
        w = (b[0]*b[2]*b[5]) - (b[1]*b[1]*b[5]) - (b[0]*b[4]*b[4]) + (2*b[1]*b[3]*b[4]) - (b[2]*b[3]*b[3])
        d = (b[0]*b[2]) - (b[1]*b[1])

        alpha = np.sqrt(w/(d*b[0]))
        beta = np.sqrt((w/d**2)*b[0])
        gamma = (np.sqrt(w/(d*d*b[0])))*b[1]
        uc = ((b[1]*b[4]) - (b[2]*b[3]))/d
        vc = ((b[1]*b[3]) - (b[0]*b[4]))/d

        # calculating intrinsic camera matrix or A matrix
        self.intrinsic_mat = np.array([[alpha, -1*gamma, uc], [0, beta, vc], [0, 0, 1]])



    def compute_extrinsic_params(self, h_init):

        # calculating extrinsic parameters
        a_inv = np.linalg.pinv(self.intrinsic_mat)
        extrinsic_mat_all = []

        for h in h_init:

            lambda_ = 1/np.linalg.norm(np.dot(a_inv, h[:, 0]))
            r0 = lambda_*np.dot(a_inv, h[:, 0])
            r1 = lambda_*np.dot(a_inv, h[:, 1])
            r2 = np.cross(r0, r1)

            r0 = r0.reshape((3, 1))
            r1 = r1.reshape((3, 1))
            r2 = r2.reshape((3, 1))

            q = np.hstack((r0, r1))
            q = np.hstack((q, r2))

            # obtain the best r matrix from q matrix
            u, s, vt = np.linalg.svd(q)
            r = np.dot(u, vt)

            # obtain translation
            t = lambda_*np.dot(a_inv, h[:, 2])
            t = t.reshape((3, 1))

            extrinsic_mat = np.hstack((r, t))
            extrinsic_mat_all.append(extrinsic_mat)

        # self.extrinsic_mat_all = np.array(extrinsic_mat_all)



    def get_camera_matrix(self, h_init):
        self.compute_intrinsic_params(h_init)
        self.compute_extrinsic_params(h_init)
        return self.intrinsic_mat
