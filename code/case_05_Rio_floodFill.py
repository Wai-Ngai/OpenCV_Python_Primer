import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
ROI：Region Of Interest


泛洪填充
FLOODFILL_FIXED_RANGE 改变图像泛洪填充
FLOODFILL_MASK_ONLY   不改变图像，只填充Mask层本身
'''

# ROI操作
def rio_test(image):
    # 高度[200:400]  宽度[200:360]
    face=image[200:400,200:360]

    # 将脸部转换成灰度图，这个时候只有一个通道
    gray=cv.cvtColor(face,cv.COLOR_BGR2GRAY)
    cv.imshow("face",gray)

    # 将脸部转换成BGR，这个时候脸部图像才有3个通道，才可以进行下面的图像合并
    # 注意：这里脸部已经是灰度图了，还原回BGR也无法显示颜色，只是为了增加通道数
    back_face=cv.cvtColor(gray,cv.COLOR_GRAY2BGR)
    cv.imshow("back_face",back_face)

    image[200:400,200:360]=back_face
    cv.imshow("lena_face",image)




# 彩色图像填充
def fill_color(image):
    copyImag=image.copy()
    h,w=image.shape[:2]
    mask= np.zeros([h+2,w+2],np.uint8)


    # 参数：原图，mask图，起始点，起始点值减去该值作为最低值，起始点值加上该值作为最高值，彩色图模式
    cv.floodFill(copyImag,mask,(30,30),(0,255,255),(100,100,100),(50,50,50),cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("flood_fill",copyImag)


# 二值图像填充
def fill_binary():
    # 创建一张400*400的彩色图像
    image=np.zeros([400,400,3],np.uint8)
    image[100:300,100:300,:]=255
    cv.imshow("fill_binary",image)

    # 创建一个单通道的mask，注意mask一定要+2，且mask像素全部为1
    mask=np.zeros([402,402,1],np.uint8)
    # 填充的区域像素值为0
    mask[101:301,101:301]=0

    cv.floodFill(image,mask,(200,200),(0,0,255),cv.FLOODFILL_MASK_ONLY)
    cv.imshow("filled_binary",image)




if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    # cv.imshow("lena", img)

    # RIO操作
    # rio_test(img)

    # 泛洪填充
    # fill_color(img)
    fill_binary()

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
