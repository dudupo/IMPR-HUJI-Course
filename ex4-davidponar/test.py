import sol4
import cv2

def draw_corners(im, local_maxes):
    for loc_max in local_maxes:
        x, y = loc_max
        cv2.circle(im, (x, y), 2, (0, 255, 0), thickness=1)

    cv2.imshow("corners", im)
    cv2.waitKey(0)


path = "external/oxford1.jpg"
im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
im_clr = cv2.imread(path)
# cv2.imshow("fd", im)
#
# cv2.waitKey(0)

# p = sol4.harris_corner_detector(im)
p = sol4.spread_out_corners(im, 7, 7, 3)
print(p)

draw_corners(im_clr, p)


