import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
学习对图像进行各种几个变换，例如移动，旋转，仿射变换等。
将要学到的函数有： cv2.getPerspectiveTransform

变换 OpenCV提供了两个变换函数，cv2.warpAﬃne和cv2.warpPerspective，使用这两个函数你可以实现所有类型的变换。
    cv2.warpAﬃne 接收的参数是 2×3 的变换矩阵
    cv2.warpPerspective 接收的参数是 3×3 的变换矩阵。

扩展缩放
    扩展缩放只是改变图像的尺寸大小。 
    OpenCV 提供的函数 cv2.resize()可以实现这个功能。图像的尺寸可以自己手动设置，你也可以指定缩放因子。我们可以选择使用不同的插值方法。
        在缩放时我们推荐使用 cv2.INTER_AREA，
        在扩展时我们推荐使用 v2.INTER_CUBIC（慢) 和 v2.INTER_LINEAR。
        默认情况下所有改变图像尺寸大小的操作使用的插值方法都是 cv2.INTER_LINEAR
平移
    平移就是将对象换一个位置。
    如果你要沿（ x， y）方向移动，移动的距离是（ tx， ty），你可以以下面的方式构建移动矩阵：
    M = np.array([[1, 0, tx], [0, 1, ty]])
    你可以使用 Numpy 数组构建这个矩阵（数据类型是 np.float32），然后把它传给函数 cv2.warpAffine()。
    
旋转
    对一个图像旋转角度 θ, 需要使用到下面形式的旋转矩阵
    为了构建这个旋转矩阵， OpenCV 提供了一个函数： cv2.getRotationMatrix2D

仿射变换
    在仿射变换中，原图中所有的平行线在结果图像中同样平行。
    为了创建这个矩阵我们需要从原图像中找到三个点以及他们在输出图像中的位置。
    然后cv2.getAffineTransform 会创建一个 2x3 的矩阵，
    最后这个矩阵会被传给函数 cv2.warpAffine

透视变换
    对于视角变换，我们需要一个 3x3 变换矩阵。
    在变换前后直线还是直线。
    要构建这个变换矩阵，你需要在输入图像上找 4 个点，以及他们在输出图像上对应的位置。
    这四个点中的任意三个都不能共线。
    这个变换矩阵可以有函数 cv2.getPerspectiveTransform() 构建。
    然后把这个矩阵传给函数cv2.warpPerspective。
    
'''


# 扩展缩放
def resize_demo(image):
    print("Origin size:", image.shape)
    # 第一种方法：通过fx，fy缩放因子
    # 下面的 None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子
    res = cv.resize(image, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    print("After resize 1 size:", res.shape)

    # 第二种方法：直接设置输出图像的尺寸，所以不用设置缩放因子
    height, width = image.shape[:2]
    res = cv.resize(image, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)
    print("After resize 2 size:", res.shape)

    while True:
        cv.imshow("image", image)
        cv.imshow("res", res)

        if cv.waitKey(1):
            break


# 平移
def move_demo(image):
    rows, cols = image.shape[:2]
    # 将图像移动（ 100, 50）个像素。
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(image, M, (cols, rows))
    cv.imshow('image', dst)


# 旋转
def rotation_demo(img):
    rows, cols = img.shape[:2]
    # 将图像相对于中心旋转90度，而不进行任何缩放。旋转中心，角度，缩放比率
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv.warpAffine(img, M, (cols, rows))
    cv.imshow('original', img)
    cv.imshow('result', dst)


# 仿射变换
def affine_demo(img):
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 100], [200, 50], [100, 250]])

    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(img, M, (cols, rows))

    plt.subplot(121, plt.show(img), plt.title("imput"))
    plt.subplot(122, plt.show(dst), plt.title("output"))
    plt.show()


# 透视变换
def perspective_demo():
    img=cv.imread("../code_images/sudoku.jpg")
    rows,cols,ch=img.shape

    pts1=np.float32([[56,65],[368,52],[28,378],[389,390]])
    pts2=np.float32([[0,0],[300,0],[0,300],[300,300]])

    M=cv.getPerspectiveTransform(pts1,pts2)
    dst=cv.warpPerspective(img,M,(300,300))

    plt.subplot(121, plt.show(img), plt.title("imput"))
    plt.subplot(122, plt.show(dst), plt.title("output"))
    plt.show()

def main():
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    # resize_demo(img)

    # move_demo(img)

    # rotation_demo(img)

    # affine_demo(img)

    perspective_demo()

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
