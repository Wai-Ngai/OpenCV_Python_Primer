import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
光流估计

'''


def optical_flow_estimation():
    cap = cv.VideoCapture("../code_images/test.avi")
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    p0 = cv.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, mask=None)
    mask = np.zeros_like(old_frame)

    color=np.random.randint(0,255,(100,3))
    while True:
        ret, frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2)

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv.add(frame, mask)

        cv.imshow("frame", img)
        k = cv.waitKey(150) & 0xff
        if k == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


def main():
    optical_flow_estimation()

    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
