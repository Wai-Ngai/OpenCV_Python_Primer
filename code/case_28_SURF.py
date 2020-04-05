import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
SURF算法


'''


def surf_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 10000
    surf=cv.xfeatures2d.SURF_create(10000)

    kp,des=surf.detectAndCompute(gray,None)

    print(len(kp))

    image2=cv.drawKeypoints(gray,kp,image,(0,0,255),4)

    cv.imshow("surf",image2)



def main():
    # 读取图片
    img = cv.imread("../code_images/Butterfly2.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("Butterfly", img)

    surf_demo(img)



    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
