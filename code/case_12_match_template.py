import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
模板匹配：在整个图像区域发现与给定子图像匹配的小块区域

通俗讲就是以图找图，通过图中的一部分来找它在图中的位置

    cv.matchTemplate(image, templ, method, result=None, mask=None)
    
    cv.TM_SQDIFF_NORMED
    cv.TM_CCORR_NORMED
    cv.TM_CCOEFF_NORMED
'''


def template_demo():
    tpl = cv.imread("../code_images/sample_4.jpg")
    target = cv.imread("../code_images/target2.jpg")
    cv.imshow("template", tpl)
    cv.imshow("target", target)

    # 三种模板匹配的方法
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    h, w = tpl.shape[:2]

    for md in methods:
        print(md)

        # 得到匹配的结果
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_val = cv.minMaxLoc(result)

        if md == cv.TM_SQDIFF_NORMED:  # cv.TM_SQDIFF_NORMED最小时最相似，其他最大时最相似
            t1 = min_loc
        else:
            t1 = max_val

        # 计算矩形框右下角坐标
        br = (t1[0] + w, t1[1] + h)
        # tl为左上角坐标，br为右下角坐标，在原图上画出一个红色的矩形框
        cv.rectangle(target, t1, br, (0, 0, 255), 2)
        cv.imshow("match-" + np.str(md), target)


        cv.imshow("result-" + np.str(md), result)


if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/lena.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    # cv.imshow("lena", img)

    template_demo()

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
