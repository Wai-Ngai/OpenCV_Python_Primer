import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
联合使用特征提取和calib3d 模块中的findHomography 在复杂图像中查找已知对象。

    基本步骤：

​	1）使用SHIT检测特征点

​	2）使用FLANN匹配器进行匹配

​	3）选取好的匹配

​	4）根据原图像和目标图像中对应的特征点，使用上述其中一种算法求变换矩阵

​			先将两幅图像的特征点传给函数cv2.findHomography() ,函数会找到对象的透视图变换

​			再使用cv2.perspectiveTransform() 找到对象。至少**4个正确点**才能找到变换！

​	5）最后将原图像的边界经变换矩阵变换后画到目标图像上


    M, mask = cv2.findHomography(srcPoints, dstPoints, method, ransacReprojThreshold, maxIters, confidence)**
        - srcPoints：原图像中对应的特征点坐标
        - dstPoints：目标图像中对应的特征点坐标
        - method：计算单应矩阵的方法
              - 0：使用所有点的常规方法，即最小二乘法
              - RANSAC：基于RANSAC的方法
              - LMEDS：最小中值稳健方法
              - RHO：基于PROSAC的方法
        - ransacReprojThreshold：取值范围1-10，是个阈值，是原图像的点经过变换后点与目标图像上对应点的误差。超过误差就是outliers，返回值的M是变换矩阵。
        
        - mask：可选输出掩码由稳健方法（RANSAC或LMEDS）设置。掩模确定inlier点和outlier点。好的匹配所提供的正确估计称为inliers，剩下称为outliers。
        - M：变换矩阵

'''


def flann_match_demo2(image1, image2):
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image1, None)  # 关键点kp1, 关键点对应的特征向量des1
    kp2, des2 = sift.detectAndCompute(image2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 使用的算法和其他相关参数
    search_params = dict(checks=50)  # 指定递归遍历的次数。值越高结果越准确，但是消耗的时间也越

    flann = cv.FlannBasedMatcher(index_params, search_params)

    # 使用FLANN匹配器进行匹配
    # k=2,一个点对应两个最近的点
    matches = flann.knnMatch(des1, des2, k=2)

    # 按照Lowe的比率存储所有好的匹配。
    good = []
    # store all the good matches as per Lowe's ratio test.
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # 只有好的匹配点多于10个才查找目标，否则显示匹配不足
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        # 获取关键点坐标
        # kp1：原图像的特征点
        # m.queryIdx：匹配点在原图像特征点中的索引
        # .pt：特征点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 获取变换矩阵，采用RANSAC算法
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # 图像变换，将原图像变换为检测图像中匹配到的形状
        # 获得原图像的高和宽
        h, w = image1.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标。
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # 对角点进行变换
        dst = cv.perspectiveTransform(pts, M)

        # 画出边框
        image2 = cv.polylines(image2, [np.int32(dst)], True, 255, 10, cv.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    # 画出匹配点
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green colo
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2
                       )
    image = cv.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)

    cv.imshow("FLANN匹配", image)


def main():
    # 读取图片
    img1 = cv.imread("../code_images/box.png", 0)
    img2 = cv.imread("../code_images/box_in_scene.png", 0)

    flann_match_demo2(img1, img2)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
