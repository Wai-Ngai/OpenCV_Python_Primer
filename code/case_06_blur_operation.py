import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
模糊操作(图像平滑)：

    低通滤波（ LPF）:帮助我们去除噪音，模糊图像。其实就是去除图像中的高频成分（比如：噪音，边界）。
    高通滤波（ HPF）:帮助我们找到图像的边缘

    均值模糊：简单的平均卷积操作，也就是求卷积区域所有像素的均值
        cv.blur(src, ksize, dst=None, anchor=None, borderType=None)
            - src：要处理的原图像
            - ksize: 周围关联的像素的范围：代码中（5，5）就是5*5的大小，就是计算这些范围内的均值来确定中心位置的大小
    
    方框滤波：基本和均值一样，可以选择归一化
        cv.boxFilter(img,-1,(3,3), normalize=True) 
            
            -normalize=True  基本和均值一样，可以选择归一化
                      =False 此时就是卷积区域的像素值总和
    
    中值模糊：将卷积区域的像素进行排序，用中间值替代中心像素的值。经常用来去除椒盐噪声
        boxFilter(src, ddepth, ksize, dst=None, anchor=None, normalize=None, borderType=None)
              
   
    高斯模糊：高斯模糊的卷积核里的数值是满足高斯分布，方框中心的值最大，其余方框根据距离中心元素的距离递减，构成一个高斯小山包。相当于更重视中间的
        cv.GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
            - ksize表示卷积核大小，并且ksize为大于0的奇数
            - sigmaX/Y表示x，y方向上的标准差，这两者只需一个即可。如果我们只指定了 X 方向的的标准差， Y 方向也会取相同值。如果两个标准差都是 0，那么函数会根据核函数的大小自己计算。
        
        高斯滤波可以有效的从图像中去除高斯噪音。可以使用函数 cv2.getGaussianKernel() 自己构建一个高斯核。
    
    双边滤波：能在保持边界清晰的情况下有效的去除噪音。但是这种操作与其他滤波器相比会比较慢。
        同时使用空间高斯权重和灰度值相似性高斯权重。
            空间高斯函数：确保只有邻近区域的像素对中心点有影响
            灰度值相似性：确保只有与中心像素灰度值相近的才会被用来做模糊运算
            
        所以这种方法会确保边界不会被模糊掉，因为边界处的灰度值变化比较大。
        
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


# 方框滤波
def box_filter(image):
    # normalize=True 这时和均值滤波没有区别
    box = cv.boxFilter(img, -1, (3, 3), normalize=True)
    cv.imshow("box filter", box)


# 中值模糊，用于椒盐噪声
def median_blur_operation(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur", dst)


# 双边滤波
def bilateral_filter():
    img = cv.imread("../code_images/bilateral.jpg")
    # 9 邻域直径，两个 75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
    blur = cv.bilateralFilter(img, 9, 75, 75)
    cv.imshow("bilateral_filter", blur)


# 自定义模糊
def customer_blur_operation(image):
    # kernel=np.ones([5,5],np.float32)/25

    # kernel=np.array([[1,1,1],[1,1,1],[1,1,1]],np.float32)/9

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化算子
    dst = cv.filter2D(image, -1, kernel)
    cv.imshow("customer_blur", dst)


def main():
    # 读取图片
    img = cv.imread("../code_images/uu.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("uu", img)

    blur_operation(img)

    median_blur_operation(img)

    bilateral_filter()

    customer_blur_operation(img)
    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
