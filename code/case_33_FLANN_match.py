import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
FLANN 是快速最近邻搜索包（Fast_Library_for_Approximate_Nearest_Neighbors）的简称。

它是一个对大数据集和高维特征进行最近邻搜索的算法的集合，而且这些算法都已经被优化过了。在面对大数据集时它的效果要好于BFMatcher。


'''


def flann_match_demo(image1, image2):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1, None)  # 关键点kp1, 关键点对应的特征向量des1
    kp2, des2 = sift.detectAndCompute(image2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 使用的算法和其他相关参数
    search_params = dict(checks=50) # 指定递归遍历的次数。值越高结果越准确，但是消耗的时间也越

    flann = cv.FlannBasedMatcher(index_params, search_params)

    # k=2,一个点对应两个最近的点
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    match_mask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            match_mask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=match_mask,
                       flags=0
                       )

    image = cv.drawMatchesKnn(image1, kp1, image2, kp2, matches, None, **draw_params)

    cv.imshow("FLANN匹配", image)


def main():
    # 读取图片
    img1 = cv.imread("../code_images/box.png", 0)
    img2 = cv.imread("../code_images/box_in_scene.png", 0)

    flann_match_demo(img1, img2)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
