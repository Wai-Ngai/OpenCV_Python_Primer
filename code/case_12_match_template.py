import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
模板匹配：在整个图像区域发现与给定子图像匹配的小块区域

通俗讲就是以图找图，通过图中的一部分来找它在图中的位置

和2D卷积一样，它也是用模板图像在输入图像（大图）上滑动，并在每一个位置对模板图像和与其对应的输入图像的子区域进行比较。 

    result =cv.matchTemplate(image, templ, method, result=None, mask=None)
        
        - method: 尽量选用带归一化的方法
            TM_SQDIFF：计算平方不同，计算出来的值越小，越相关
            TM_CCORR：计算相关性，计算出来的值越大，越相关
            TM_CCOEFF：计算相关系数，计算出来的值越大，越相关
            TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
            TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
            TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关
            
        返回的结果是一个灰度图像，每一个像素值表示了此区域与模板的匹配程度。
        如果输入图像的大小是（ WxH），模板的大小是（ wxh），输出的结果的大小就是（ W-w+1， H-h+1）
            
    cv2.minMaxLoc(result) 找到其中的最小值和最大值的位置
    
'''


def template_demo():
    tpl = cv.imread("../code_images/sample_4.jpg")
    target = cv.imread("../code_images/target2.jpg")
    cv.imshow("template", tpl)
    cv.imshow("target", target)
    print(target.shape)
    print(tpl.shape)

    # 三种模板匹配的方法
    methods = [cv.TM_SQDIFF_NORMED, cv.TM_CCORR_NORMED, cv.TM_CCOEFF_NORMED]
    h, w = tpl.shape[:2]

    for md in methods:
        print(md)

        # 得到匹配的结果
        result = cv.matchTemplate(target, tpl, md)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if md == cv.TM_SQDIFF_NORMED:  # cv.TM_SQDIFF_NORMED最小时最相似，其他最大时最相似
            t1 = min_loc
        else:
            t1 = max_loc

        # 计算矩形框右下角坐标
        br = (t1[0] + w, t1[1] + h)
        # tl为左上角坐标，br为右下角坐标，在原图上画出一个红色的矩形框
        cv.rectangle(target, t1, br, (0, 0, 255), 2)
        cv.imshow("match-" + np.str(md), target)

        cv.imshow("result-" + np.str(md), result)
        print(result.shape)




def main():
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


if __name__ == '__main__':
    main()
