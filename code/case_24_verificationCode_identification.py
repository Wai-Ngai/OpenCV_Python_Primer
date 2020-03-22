import cv2 as cv
import numpy as np
from PIL import Image
import pytesseract as tess

'''
预处理-去除干扰线和点
不同的结构元素中选择
Image和numpy array相互转换
识别和输出
'''


def verificationCode_recongnition(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imshow("binary", binary)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 4))
    bin1 = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    cv.imshow("bin1", bin1)

    # 将黑色背景转换成白色背景
    cv.bitwise_not(bin1,bin1)

    text_image = Image.fromarray(bin1)

    text = tess.image_to_string(text_image)
    print("result:", text)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/yzm.jpg")

    verificationCode_recongnition(img)

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
