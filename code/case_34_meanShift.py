import cv2 as cv
import numpy as np

'''
Meanshift 算法



首先我们要对目标对象进行设置，计算目标对象的直方图，这样在执行meanshift 算法时我们就可以将目标对象反向投影到每一帧中去了。
另外我们还需要提供窗口的起始位置。在这里我们值计算H（Hue）通道的直方图，同样为了避免低亮度造成的影响，我们使用函数cv2.inRange() 将低亮度的值忽略

'''

cap = cv.VideoCapture("../code_images/slow.mp4")

# 取出视频的第一帧
ret, frame = cap.read()

# 为了找到你要追踪的物体，通过获取第一帧图像，在图像上面找到你要追踪物体的位置
cv.imshow("init",frame)


# 设置追踪窗口的初始位置
# r：矩形框右上角y坐标
# h：矩形框的高度
# c：矩形框右上角x坐标
# w：矩形框的宽度
r, h, c, w = 200, 50, 250, 100
track_window = (c, r, w, h)

# 设置用于追踪的ROI
roi = frame[r:r + h, c:c + w]
hsv_roi = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while True:

    ret, frame = cap.read()

    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 使用均值迁移得到一个新的位置
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # 将追踪窗口画在图像上
        x, y, w, h = track_window
        img = cv.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv.imshow("image", img)

        k = cv.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv.imwrite(chr(k) + ".jpg", img)

    else:
        break

cv.destroyAllWindows()
cap.release()
