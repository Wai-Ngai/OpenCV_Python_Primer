import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
图像直方图的应用

1.直方图均衡化：用于增强图像对比度，即黑的更黑，白的更白
    全局直方图均衡化
        dst = cv.equalizeHist(gray) 
    局部直方图均衡化
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
2.直方图比较

'''


# 1.直方图均衡化

def equalHist_demo(image):
    # openCV中的直方图均衡化都是基于灰度图像，所以要将彩色图像转换成灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 全局直方图均衡化
    dst = cv.equalizeHist(gray)
    cv.imshow("equalizeHist", dst)


def clahe_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 局部直方图均衡化
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_dst = clahe.apply(gray)
    cv.imshow("clahe", clahe_dst)


# 创建一个RGB三通道的直方图
def create_rgb_demo(image):
    h, w, c = image.shape
    # 直方图初始化，这里都是意思是每个通道16个bin，一共组合出16*16*16种
    # 注意：这里直方图类型必须是float32，不然后面比较的时候会出错
    rgbHist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16

    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            # 下面三个维度，b*16*16,g*16,r*1相当于三维变一维
            # 想一想如何用一个一维的数组表示一个三维的数组
            # 然后数组中的行索引就是各个像素加权值，索引中对应的值就是统计出来它出现的次数
            # 不推荐np.int(),可以用b//bsize
            index = np.int(b / bsize) * 16 * 16 + np.int(g / bsize) * 16 + np.int(r / 16)

            rgbHist[index, 0] += 1
    return rgbHist


# 2.直方图比较
# 利用直方图比较相似性，用巴氏和相关性比较好
# 巴氏距离越小越相似
# 相关性越大越相似
# 卡方越大越相似

def hist_compare(image1, image2):
    hist1 = create_rgb_demo(image1)
    hist2 = create_rgb_demo(image2)

    # 注意：直方图比较需要两张图大小一直，否则需要做归一化
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)

    print("巴式距离：%s, 相关性：%s, 卡方：%s" % (match1, match2, match3))
    # 巴式距离：0.5424051452634745, 相关性：0.5834816306624869, 卡方：8334059.811340665


def main():
    # 读取图片
    img = cv.imread("../code_images/rice.png")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("rice", img)

    # equalHist_demo(img)
    # clahe_demo(img)

    image1 = cv.imread("../code_images/rice.png")
    image2 = cv.imread("../code_images/noise_rice.png")
    hist_compare(image1, image2)
    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
