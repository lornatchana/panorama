import cv2
import numpy as np
import sys


class DoingPanorama:
    def __init__(self, images):
        self.images = images
        self.matches = None
        self.img1 = None
        self.img2 = None
        self.img3 = None
        self.src_pts = None
        self.dst_pts = None
        self.bestmatches = None
        self.homography = None
        self.stitcher = None

    def features_extraction(self):
        # we are reading images
        # The flag 1 specifies that the image should be read in color mode.
        # The flag 0 specifies that the image should be read in grayscale mode.
        images = ["img1.jpg", "img2.jpg"]
        img1 = cv2.imread('img1.jpg', 1)
        img2 = cv2.imread('img2.jpg', 1)

        # remove colors on the two images
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # Using sift for recovering features of images
        sift = cv2.xfeatures2d.SIFT_create()

        # calculates characteristic points between the two images
        # none want means no mask image should be used.
        #
        # A mask image is a grayscale image that is used to specify the regions of an image
        # to extract features from
        # descriptor is a vector that describes the local appearance of an image around a keypoint.
        keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        # find the features matching
        # A Brute Force Matcher is a type of feature matcher
        # that compares all possible pairs of features between two images.
        brute_force_matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        # Match feature points between images.
        matches = brute_force_matcher.match(descriptors_1, descriptors_2)
        # making a sort according to the distance
        matches = sorted(matches, key = lambda x: x.distance)
        num_matches = 50
        self.best_matches = matches[:num_matches]
        # Extract the coordinates of the corresponding points in the two images:
        # It creates a list of points src_pts.
        # Each point in the list is the pt attribute of a keypoint in keypoints_1.
        # The reshape() method of a NumPy array reshapes the array to have the specified dimensions.
        # The -1 dimension in the (-1, 1, 2) shape specifies that the first dimension should be

        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in self.best_matches]).reshape(-1, 1, 2)
        #
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in self.best_matches]).reshape(-1, 1, 2)
        # Draw the results of features comparisons and slice of a big number of correspondence
        img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, self.best_matches, img2, flags=2)
        # plt.imshow(img3), plt.show()

        self.matches = matches
        self.img1 = img1
        self.img2 = img2
        self.img3 = img3
        self.src_pts = src_pts
        self.dst_pts = dst_pts

    # Estimate the homography matrix between each pair of images
    # Homography can be used to map the corresponding points in the two images.
    # This mapping can be used to stitch the two images together

    def estimate_homography(self):
        # Estimate the homography matrix between each pair of images
        self.homography, _ = cv2.findHomography(self.src_pts, self.dst_pts, cv2.RANSAC, 5.0)
        # homography = cv2.estimateAffine2D(self.img1[0][self.bestmatches], self.img2[0][self.bestmatches])
        # print(self.homography)

    # Distort images using homography matrices to align them
    def wrapped_image(self):
        wrapped_image = cv2.warpPerspective(self.img1, self.homography, (self.img2.shape[1], self.img2.shape[0]))
        # The cv2.addWeighted() function will combine the two aligned images to create a panorama.
        # blends the two images wrapped_image and self.img2 together to create a new image img_blended.
        # The cv2.addWeighted() function takes four arguments:The first image to blend, The weight of the first image,
        # The second image to blend,The weight of the second image, The gamma correction value.
        # The cv2.addWeighted() function will calculate a weighted average of the two input images,
        # using the specified weights.
        # The gamma correction value is used to adjust the brightness and contrast of the blended image.:
        img_blended = cv2.addWeighted(wrapped_image, 0.2, self.img2, 0.2, 0.3)


    def create_panorama(self):
        imgs = []
        try:
            # creation of an array of images containing  images

            images = ["img1.jpg", "img2.jpg"]
            # load images
            for img_name in images:
                # reading of every image in images
                # After reading in the image data will be stored in a cv::Mat object.
                img = cv2.imread(cv2.samples.findFile(img_name))

                # a check is executed,
                # if the image was loaded correctly.
                if img is None:
                    print("can't read image " + img_name)
                    sys.exit(-1)
                #  store every img in imgs
                imgs.append(img)
            # the goal it's to create and store the parameters of the panorama in stitcher
            stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

            status, pano = stitcher.stitch(imgs)

            if status != cv2.Stitcher_OK:
                print("Can't stitch images, error code = %d" % status)
                sys.exit(-1)
            # recordering the panorama in result.jpg
            print("stitching completed successfully. %s saved!" % 'result.jpg')

            print('Done')
            cv2.imwrite('result.jpg', pano)
            cv2.imshow('result.jpg', pano)
            cv2.waitKey(0)
            # To close the window before the user presses a key,
            cv2.destroyAllWindows()

        except Exception as ex:
            print(ex)


new_panorama = DoingPanorama("img11.jpg")
new_panorama.features_extraction()
new_panorama.estimate_homography()
new_panorama.wrapped_image()
new_panorama.create_panorama()


