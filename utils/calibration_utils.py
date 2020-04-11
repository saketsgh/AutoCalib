import cv2
import os
import numpy as np
from image_utils import ImgUtils

img_utils = ImgUtils()

class CalibUtils:

    def get_corner_pts(self, calib_images, grid_size):

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_all_imgs = []

        if not os.path.exists("results/"):
            os.makedirs("results/")

        for i, img in enumerate(calib_images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            if(ret==True):
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                corners2 = corners2.reshape((corners.shape[0], -1))
                # uncomment to get results
                '''img = plot_points(self, img, corners2, color=(255, 0, 0))
                cv2.imwrite("results/world_points/corners_img_"+str(i)+".png", img)'''
                corners_all_imgs.append(corners2)

        return corners_all_imgs


    def get_homography(self, img, img_corners, world_coord):

        img_coord = []
        # img_utils = ImgUtils()

        # choosing four points for finding homography
        img_coord.append(img_corners[0])
        img_coord.append(img_corners[5])
        img_coord.append(img_corners[45])
        img_coord.append(img_corners[50])

        img_coord = np.array(img_coord, np.float32)

        homography = cv2.getPerspectiveTransform(world_coord, img_coord)

        show_4_corners = False

        # to show which points where chosen
        if(show_4_corners):
            for i, c in enumerate(img_coord):

                # font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # org
                org = (c[0],c[1])

                text = str(org)+'('+ str(i) + ')'
                # fontScale
                fontScale = 1

                # Blue color in BGR
                color = (0, 0, 0)

                # Line thickness of 2 px
                thickness = 2
                img = cv2.putText(img, text, org, font,
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
        intrinsic_mat = np.array([[alpha, -1*gamma, uc], [0, beta, vc], [0, 0, 1]])

        return intrinsic_mat


    def compute_extrinsic_params(self, intrinsic_mat, h_init):

        # calculating extrinsic parameters
        a_inv = np.linalg.pinv(intrinsic_mat)
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

        return extrinsic_mat_all


    def get_all_parameters(self, h_init):

        intrinsic_mat = self.compute_intrinsic_params(h_init)
        extrinsic_mat_all = self.compute_extrinsic_params(intrinsic_mat, h_init)

        return intrinsic_mat, extrinsic_mat_all


    def get_projected_img_coord(self, world_coord, intrinsic_mat, extrinsic_mat):

        # convert the world coord to 3D homogenuous with Z=0
        zeros = np.zeros((world_coord.shape[0], 1))
        world_coord = np.hstack((world_coord, zeros))
        world_coord = img_utils.get_homogenuous(world_coord)

        # compute projected image coord
        normalised_img_coord = np.dot(extrinsic_mat, world_coord.T)
        projected_img_coord = np.dot(intrinsic_mat, normalised_img_coord)

        # converting back to non-homogenous form
        normalised_img_coord = img_utils.get_non_homogenuous(normalised_img_coord.T)
        projected_img_coord = img_utils.get_non_homogenuous(projected_img_coord.T)

        return projected_img_coord, normalised_img_coord


    def get_projection_error(self, world_coord, intrinsic_mat, extrinsic_mat_all, k_mat, img_coord_all, save_image=False):

        i = 0
        rectified_imgs = img_utils.get_images("../References/AutoCalib/rectified_imgs/")
        error_all_imgs = np.array([])
        # obtain the coordinates of image center
        uc = intrinsic_mat[0][2]
        vc = intrinsic_mat[1][2]

        for extrinsic_mat, img_coord in zip(extrinsic_mat_all, img_coord_all):

            U, X = self.get_projected_img_coord(world_coord, intrinsic_mat, extrinsic_mat)

            u = U[:, 0].reshape((X.shape[0], 1))
            v = U[:, 1].reshape((X.shape[0], 1))
            x = X[:, 0].reshape((X.shape[0], 1))
            y = X[:, 1].reshape((X.shape[0], 1))

            x_y = (x**2) + (y**2)
            lens_dist = k_mat[0][0]*(x_y) + k_mat[1][0]*(x_y**2)

            # projected image coordinates with distortion
            u_h = u + np.multiply((u-uc), lens_dist)
            v_h = v + np.multiply((v-vc), lens_dist)

            # observed image coordinates
            u_img = img_coord[:, 0].reshape((X.shape[0], 1))
            v_img = img_coord[:, 1].reshape((X.shape[0], 1))

            # for saving projected points
            if(save_image):
                img = img_utils.plot_points(rectified_imgs[i], img_coord.astype(np.float32), color=(0, 0, 0))
                img = img_utils.plot_points(img, U.astype(np.float32), color=(0, 0, 255))
                img_utils.save_image(img, path="../References/AutoCalib/reprojected_pts/", title="img"+str(i))

            # l2 norm or sum of squared diff
            error_all_pts = (u_img-u_h)**2 + (v_img-v_h)**2
            error_all_imgs = np.append(error_all_imgs, np.sqrt(error_all_pts))
            i += 1

        return error_all_imgs


    def flatten_ak(self, a, k):
        x0 = np.array([a[0][0], a[0][1], a[0][2],
                       a[1][1], a[1][2], k[0][0], k[1][0]])
        return x0


    def unflatten_ak(self, x):
        k = np.array([[x[5]], [x[6]]])
        a = np.array(
            [[x[0], x[1], x[2]], [0, x[3], x[4]], [0, 0, 1]])
        a = a.reshape((3, 3))
        return a, k
