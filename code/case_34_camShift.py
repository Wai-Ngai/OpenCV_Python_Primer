import cv2 as cv
import numpy as np

'''
Camshift 算法

Camshift它是MeanShift算法的改进，称为连续自适应的MeanShift算法.
它的基本思想是视频图像的所有帧作MeanShift运算，并将上一帧的结果（即Search Window的中心和大小）作为下一帧MeanShift算法的Search Window的初始值，如此迭代下去。

'''

cap = cv.VideoCapture("../code_images/slow.mp4")

# 取出视频的第一帧
ret, frame = cap.read()

# 设置追踪窗口的初始位置
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

        # 使用CamShift得到一个新的位置
        ret, track_window = cv.CamShift(dst, track_window, term_crit)

        # 将追踪窗口画在图像上
        pts=cv.boxPoints(ret)
        pts=np.int0(pts)
        img=cv.polylines(frame,[pts],True,255,2)

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
