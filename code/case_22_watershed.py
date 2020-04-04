import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
分水岭算法

基于距离变换，找到种子点

基于距离的分水岭分割流程：（需要去除噪声）
输入图像->灰度化->二值化->距离变换->寻找种子->生成marker->分水岭变换->输出图像
'''


def watershed_demo(image):
    # remove noise
    blured = cv.pyrMeanShiftFiltering(image, 10, 100)

    # gray/binary
    gray = cv.cvtColor(blured, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    # morphology operation
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)  # 连续两次开操作
    sure_bg = cv.dilate(mb, kernel, iterations=3)  # 连续三次膨胀操作
    cv.imshow("morphology operation", sure_bg)

    # Distance Transform
    # 第二个参数计算距离发方式 0,1,2 分别表示 CV_DIST_L1, CV_DIST_L2 , CV_DIST_C
    # 5 距离变换掩摸的大小
    dist = cv.distanceTransform(mb, 1, 5)
    # 将距离变换的结果归一化到[0,1]之间，为了很好的显示距离变换的结果
    dis_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("distance", dis_output * 50)

    # 距离变换中最亮的地方就是markers
    ret, surface = cv.threshold(dist, 0.6 * dist.max(), 255, cv.THRESH_BINARY)
    cv.imshow("sufrace", surface)

    surface_fg = np.uint8(surface)
    # Finding unknown region
    unknown = cv.subtract(sure_bg, surface_fg)

    # markers
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)

    # watershed transform
    '''
    现在知道了那些是背景那些是硬币了。那我们就可以创建标签（一个与原图像大小相同，数据类型为in32 的数组），并标记其中的区域了。
    对我们已经确定分类的区域（无论是前景还是背景）使用不同的正整数标记，
    对我们不确定的区域使用0 标记。
    我们可以使用函数cv2.connectedComponents()来做这件事。它会把将背景标记为0，其他的对象使用从1 开始的正整数标记。
    但是，我们知道如果背景标记为0，那分水岭算法就会把它当成未知区域了,所以这里要+1。所以我们想使用不同的整数标记它们。而对不确定的区域（函数cv2.connectedComponents 输出的结果中使用unknown 定义未知区域）标记为0。
    '''
    markers1 = markers + 1
    markers1[unknown == 255] = 0

    markers2 = cv.watershed(image, markers)
    image[markers2 == -1] = [0, 0, 255]
    cv.imshow("result", image)

def main():
    # 读取图片
    img = cv.imread("../code_images/coins.png")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("input_image", img)

    watershed_demo(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
