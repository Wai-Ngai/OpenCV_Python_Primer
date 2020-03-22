import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
人脸检测

HAAR 与 LBP特征数据
'''


def face_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier("../code_images/haarcascade_frontalface_alt_tree.xml")

    faces = face_detector.detectMultiScale(gray, 1.02, 5) # 1.02, 5 需要进行调整

    for x, y, w, h in faces:
        # 绘制在原图上面
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow("face_detection", image)


def video_face_detection():
    # 视频检测，检测摄像头里面的人脸
    capture=cv.VideoCapture(0)
    # cv.namedWindow("video_face_detection", cv.WINDOW_AUTOSIZE)
    while True:
        ret,frame=capture.read()
        frame=cv.flip(frame,1)

        face_detection(frame)

        c=cv.waitKey(0)
        if c==27:
            break

if __name__ == '__main__':
    # 读取图片
    img = cv.imread("../code_images/face3.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    cv.imshow("Dilraba", img)

    # face_detection(img)
    video_face_detection()



    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()
