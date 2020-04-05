import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
Brute-Force蛮力匹配


'''


# 1对1匹配
def bf_match_demo(image1, image2):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1, None)  # 关键点kp1, 关键点对应的特征向量des1
    kp2, des2 = sift.detectAndCompute(image2, None)

    # create BFMatcher object
    # crossCheck=True 表示两个特征点要互相匹，例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是
    bf = cv.BFMatcher(crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # 取前10个关键点
    image3 = cv.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=2)

    cv.imshow("Brute-Force match", image3)


# k对最佳匹配
def bf_match_demo2(image1, image2):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1, None)  # 关键点kp1, 关键点对应的特征向量des1
    kp2, des2 = sift.detectAndCompute(image2, None)

    # create BFMatcher object
    # BFMatcher with default params
    bf = cv.BFMatcher()

    # k=2,一个点对应两个最近的点
    matches2 = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    # 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
    # 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
    good = []
    for m, n in matches2:
        if m.distance < 0.75 * n.distance:  # 这里指定了一个过滤的方法：m/n < 0.75
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    image4 = cv.drawMatchesKnn(image1, kp1, image2, kp2, good, None, flags=2)

    cv.imshow("k对最佳匹配", image4)


def main():
    # 读取图片
    img1 = cv.imread("../code_images/box.png", 0)
    img2 = cv.imread("../code_images/box_in_scene.png", 0)

    # bf_match_demo(img1, img2)
    bf_match_demo2(img1, img2)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
