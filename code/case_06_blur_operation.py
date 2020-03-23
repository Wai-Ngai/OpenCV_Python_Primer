import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
模糊操作：

    均值模糊：简单的平均卷积操作，也就是求卷积区域所有像素的均值
        cv.blur(src, ksize, dst=None, anchor=None, borderType=None)
            - src：要处理的原图像
            - ksize: 周围关联的像素的范围：代码中（5，5）就是9*5的大小，就是计算这些范围内的均值来确定中心位置的大小
    
    方框滤波：基本和均值一样，可以选择归一化
        cv.boxFilter(img,-1,(3,3), normalize=True) 
            
            -normalize=True  基本和均值一样，可以选择归一化
                      =False 此时就是卷积区域的像素值总和
    
    中值模糊：将卷积区域的像素进行排序，用中间值替代原始值
        boxFilter(src, ddepth, ksize, dst=None, anchor=None, normalize=None, borderType=None)
            
     
   
    高斯模糊：高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的
        cv.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
    
    
    双边滤波函数
        cv.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None)
            - d：邻域直径
            - sigmaColor：颜色标准差
            - sigmaSpace：空间标准差
    
    自定义模糊
        cv.filter2D(src,ddepth,kernel)
            - ddepth：深度，输入值为-1时，目标图像和原图像深度保持一致
            - kernel: 卷积核（或者是相关核）,一个单通道浮点型矩阵
    
    基本原理：
    1.基于离散卷积
    2.定义好每个卷积核
    3.不同卷积核得到不同卷积效果
    4.模糊是卷积的一种表象



'''


# 均值模糊
def blur_operation(image):
    # 卷积核大小5*5
    dst = cv.blur(image, (5, 5))
    cv.imshow("blur", dst)

def box_filter(image):
    # normalize=True 这时和均值滤波没有区别
    box = cv.boxFilter(img, -1, (3, 3), normalize=True)
    cv.imshow("box filter",box)


# 中值模糊，用于椒盐噪声
def median_blur_operation(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur", dst)

#自定义模糊
def customer_blur_operation(image):

    # kernel=np.ones([5,5],np.float32)/25

    # kernel=np.array([[1,1,1],[1,1,1],[1,1,1]],np.float32)/9

    kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)  # 锐化算子
    dst = cv.filter2D(image, -1,kernel)
    cv.imshow("customer_blur", dst)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/uu.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("uu", img)

    blur_operation(img)

    median_blur_operation(img)

    customer_blur_operation(img)
    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
