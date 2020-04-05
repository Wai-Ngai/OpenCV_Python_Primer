import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
SIFT算法

    因为OpenCV 3.4.1.15 以上的版本针对有些方法已经做了一些专利保护，没有办法用（3.4.2.16也可以用）

    查看当前的版本：  cv2.__version__    
    
    将opencv版本退到3.4.2，卸载之前的包
    
        pip uninstall opencv-python
        
        pip uninstall opencv-contrib-python
    
    重新安装
        
        pip install opencv-python==3.4.1.15 
        
        pip install opencv-contrib-python==3.4.1.15
    
    opencv将SIFT等算法整合到xfeatures2d集合里面了
    
        siftDetector=cv2.SIFT()
        
    变更后为
    
        siftDetector= cv2.xfeatures2d.SIFT_create()

'''


def sift_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 实例化sift算法
    sift = cv2.xfeatures2d.SIFT_create()
    # 得到关键点。返回的关键点是一个带有很多不同属性的特殊结构体，这些属性中包含它的坐标（x，y），有意义的邻域大小，确定其方向的角度等。
    kp = sift.detect(gray, None)

    # 绘制关键点的函数，drawKeyPoints()，它可以在关键点的部位绘制一个小圆圈
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 就会绘制代表关键点大小的圆圈甚至可以绘制除关键点的方向。
    img = cv2.drawKeypoints(gray, kp, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("drawKeypoints", img)

    # 使用函数sift.compute() 来计算这些关键点的描述符
    kp, des = sift.compute(gray, kp)

    #  kp是一个list，需要转换一下
    print(np.array(kp).shape)
    # 每个关键点相应的特征
    print(des.shape)

    des[0]


def main():
    # 读取图片
    img = cv2.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv2.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv2.imshow("lena", img)

    sift_demo(img)

    # 等待键盘输入
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
