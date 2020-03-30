import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
OpenCV 中的绘图函数

cv2.line()         画线
cv2.circle()       画圆
cv2.rectangle()    画矩形
cv2.ellipse()      画椭圆
cv2.putText()      添加文字


上面所有的这些绘图函数需要设置下面这些参数：
• img：你想要绘制图形的那幅图像。
• color：形状的颜色。以 RGB 为例，需要传入一个元组，例如：（ 255,0,0）代表蓝色。对于灰度图只需要传入灰度值。
• thickness：线条的粗细。如果给一个闭合图形设置为 -1，那么这个图形就会被填充。默认值是 1.
• linetype：线条的类型， 8 连接，抗锯齿等。默认情况是 8 连接。 cv2.LINE_AA为抗锯齿，这样看起来会非常平滑。
'''


def main():
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("lena", img)

    img2=np.zeros((512,512,3),np.uint8)
    # 画一条从左上方到右下角的蓝色线段
    cv.line(img2,(0,0),(511,511),(255,0,0),5)

    cv.rectangle(img2,(384,0),(510,128),(0,255,0),3)

    cv.circle(img2,(447,63),63,(0,255,255))

    cv.ellipse(img2,(256,256),(100,50),0,0,180,255,-1)

    font=cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img2,"hello opencv",(50,500),font,4,(0,0,255),2)


    cv.imshow("line",img2)




    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
