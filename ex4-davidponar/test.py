import sol4
import cv2
import sol4_utils
import numpy as np

def match_features_test():    
    points_and_descriptors = []
    #images = [ "external/oxford1.jpg" , "external/oxford2.jpg" ]
    images = []
    for file in [ "external/oxford1.jpg" , "external/oxford2.jpg" ]:
        image = sol4_utils.read_image(file, 1)
        images.append(image)
    #   self.h, self.w = image.shape
        print(image.shape)
        pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
        points_and_descriptors.append(sol4.find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(len(points_and_descriptors) - 1):
        points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
        desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]
      # Find matching feature points.
      
        ind1, ind2 = sol4.match_features(desc1, desc2, .7)
        print(ind1.shape)
        print(points1.shape)
        points1, points2 = points1[ind1, :], points2[ind2, :]
        H, inl = sol4.ransac_homography(points1, points2,  100, 6, translation_only=True  )
        sol4.display_matches(images[i], images[i+1], points1, points2,inl )
        # draw_lines( images[i], points1, sol4.apply_homography(points1, H).astype(np.int))
        # draw_corners( images[i+1] , points2)
      # Compute homography using RANSAC.
        # H12, inliers = ransac_homography(points1, points2, 100, 10, translation_only)


def draw_lines(im, p1, p2):
    im = cv2.imread(im)
    for _p1 , _p2 in zip(p1, p2):
        _x11,_x12 = _p1
        _x21,_x22 = _p2
        cv2.line(im, (_x11,_x12), (_x21,_x22),(0, 255, 0), thickness=1)
    
    cv2.imshow("line", im)
    cv2.waitKey(0)

def draw_corners(im, local_maxes):
    im = cv2.imread(im)
    for loc_max in local_maxes:
        x, y = loc_max
        cv2.circle(im, (x, y), 2, (0, 255, 0), thickness=1)

    cv2.imshow("corners", im)
    cv2.waitKey(0)


# path = "external/oxford1.jpg"
# im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# im_clr = cv2.imread(path)
# # cv2.imshow("fd", im)
# #
# # cv2.waitKey(0)

# # p = sol4.harris_corner_detector(im)
# p = sol4.spread_out_corners(im, 7, 7, 3)
# print(p)

# draw_corners(im_clr, p)

match_features_test()




