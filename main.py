import cv2
import matplotlib.pyplot as plt
import numpy as np


class DoingPanorama:
    def __init__(self, image):
        self.image = image
        self.matches = None
        self.img1 = None
        self.img2 = None
        self.img3 = None
        self.src_pts = None
        self.dst_pts = None
        self.bestmatches = None
        self.homography = None
        self.imgs = None

    def features_extraction(self):
        # we are reading images
        imgs = []
        img1 = cv2.imread('img1.jpg', 1)
        img2 = cv2.imread('img2.jpg', 1)
        imgs.append(img1)
        imgs.append(img2)
        # remove colors on the two images
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # Using sift for recovering features of images
        sift = cv2.xfeatures2d.SIFT_create()

        # calculates characteristic points between the two images
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        # find the features matching
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # Match feature points between images.
        matches = brute_force_matcher.match(descriptors_1, descriptors_2)
        # making a sort according to the distance
        matches = sorted(matches, key = lambda x: x.distance)
        num_matches = 50
        self.best_matches = matches[:num_matches]
        # Extract the coordinates of the corresponding points in the two images:
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in self.best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in self.best_matches]).reshape(-1, 1, 2)
        # Draw the results of features comparisons and slice of a big number of correspondence
        img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, self.best_matches, img2, flags=2)
        plt.imshow(img3), plt.show()

        self.matches = matches
        self.img1 = img1
        self.img2 = img2
        self.img3 = img3
        self.src_pts = src_pts
        self.dst_pts = dst_pts

    # Estimate the homography matrix between each pair of images
    def estimate_homography(self):
        # Estimate the homography matrix between each pair of images
        self.homography, _ = cv2.findHomography(self.src_pts, self.dst_pts, cv2.RANSAC, 5.0)
        # homography = cv2.estimateAffine2D(self.img1[0][self.bestmatches], self.img2[0][self.bestmatches])
        print(self.homography)

    # Distort images using homography matrices to align them
    def wrapped_image(self):
        wrapped_image = cv2.warpPerspective(self.img1, self.homography, (self.img2.shape[1], self.img2.shape[0]))
        # The cv2.addWeighted() function will combine the two aligned images to create a panorama.
        img_blended = cv2.addWeighted(wrapped_image, 0.2, self.img2, 0.2, 0.3)

        # Show img1, img2 and aligned_image
        cv2.imshow("Image 1", self.img1)
        cv2.imshow("Image 2", self.img2)
        cv2.imshow("wrapped_image 1", img_blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_panorama(self):

        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

        status, panorama = stitcher.stitch(self.imgs)
        print(status, "status", cv2.Stitcher_OK)
        if status != cv2.Stitcher_OK:
            # recordering panorama in a file
            cv2.imwrite("\\panorama1.jpg", panorama)
            cv2.imshow("panorama1.jpg", panorama)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("panorama recorded sucessfully")
        else:
            print("Error during the panorama's creation")


new_panorama = DoingPanorama("img11.jpg")
new_panorama.features_extraction()
new_panorama.estimate_homography()
new_panorama.wrapped_image()
new_panorama.create_panorama()


