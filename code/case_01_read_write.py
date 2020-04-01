import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
图像、视频的加载和保存
图像的基础知识

    cv.imread(filename, flags=None)
        - filename:图片地址
        - flags:如何读取这幅图片
            cv2.IMREAD_COLOR：读入一副彩色图像。图像的透明度会被忽略，这是默认参数。 1
            cv2.IMREAD_GRAYSCALE：以灰度模式读入图像 0
            cv2.IMREAD_UNCHANGED：读入一幅图像，并且包括图像的 alpha 通道  -1
    
    cv.imshow(winname, mat)
        - winname：窗口名称
        - mat：要显示的图片
    
    cv.imwrite(filename, img, params=None)
        -filename
        -img
        -params=None
    
    cv.VideoCapture(*args, **kwargs)
        - 打开摄像头，0代表的是设备id，如果有多个摄像头，可以设置其他数值
        - 也可以是视频文件地址，调用视频文件，如果要播放要设置帧的循环
        
    cv2.imwrite()
    
    
    cv.waitKey(delay=None)
        - delay : 窗口显示时间，单位：毫秒
             k=0: （也可以是小于0的数值）一直显示，键盘上按下一个数字键即会消失
             k>0:  显示多少毫秒。特定的几毫秒之内，如果按下任意键，这个函数会返回按键的 ASCII 码值，程序将会继续运行。如果没有键盘输入，返回值为 -1
    
    cv.namedWindow(winname, flags=None)
        - winname：窗口名称
        - flags
            为0或cv.WINDOW_NORMAL：可以改变窗口大小
            不写或cv.WINDOW_AUTOSIZE则不可改变大小
          
    cv.destroyAllWindows()
        删除建立的全部窗口，释放资源
"""


# 获取图片信息
def get_image_info(image):
    print("图片类型：", type(image))  # <class 'numpy.ndarray'>
    print("图片形状（长,宽,通道数）：", image.shape)  # <class 'tuple'>
    print("图片像素数目（长×宽×通道数）：", image.size)
    print("图片像素数据类型：", image.dtype)  # uint8

    # 将图片转换成数组,是一个3维数组
    pixel_data = np.array(image)
    print("像素数组：", pixel_data)


def image_read():
    img = cv.imread("../code_images/target3.jpg")

    '''
    由于OpenCV是以BGR模式加载图像，而matplotlib则是以常见的RGB模式显示图像，
    因此通过OpenCV加载的彩色图像用matplotlib显示时会出现问题，针对此问题，这里主要提出三种解决方案
    '''
    # 方法一：利用cv2.split()和cv2.merge()函数将加载的图像先按照BGR模式分开哥哥通道，然后再按照RGB模式合并图像
    # B,G,R=cv.split(img)
    # img2=cv.merge([R,G,B])

    # 方法二：使用数组的逆序，将最后一位转置一下
    # img2=img[...,::-1]

    # 方法三：使用opencv自带的模式转换函数cv2.cvtColor()
    img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.imshow(img2)
    plt.xticks([]), plt.yticks([])
    plt.show()


# 保存图片
def save_image(image):
    # 将图片转换为灰度图
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 保存图片
    cv.imwrite("../code_images/new.png", gray)


'''
    Ubuntu下打开笔记本内置摄像机失败解决(本人用的VMware)
        
        1、首先  虚拟机->可移动设备 连接摄像头；        
        2、运行cheese，发现漆黑一片；      
        3、虚拟机->设置->USB控制器，将USB兼容性改成USB3.0即可。
'''


# 读取视频
def video_read():
    # 读取摄像头视频，用数字来控制不同的设备，例如0,1。
    # capture = cv.VideoCapture(0)

    # 读取本地文件视频
    capture = cv.VideoCapture("../code_images/cxk_playBB.mp4")
    print("类型", type(capture))  # <class 'cv2.VideoCapture'>

    while True:
        # 获取相机图像，返回ret(结果为True/False)，和每一帧图片frame
        ret, frame = capture.read()
        print("ret:", ret)

        # 将图片水平翻转，竖直翻转为0
        frame = cv.flip(frame, 1)

        # 将每一帧图片放入video窗口
        cv.imshow("video", frame)

        # 等有键输入(这里指c=Esc键)或者50ms后自动将窗口消除
        # 如果设置的太低视频就会播放的非常快，如果设置的太高就会播放的很慢（你可以使用这种方法控制视频的播放速度）。通常情况下 25 毫秒就可以了。
        c = cv.waitKey(25)
        # esc的ASCII编码为27
        if c == 27:
            break


'''
视频保存，通过创建一个VideoWriter对象。
我们应该确定一个输出文件的名字。接下来指定 FourCC 编码。
播放频率和帧的大小也都需要确定。
最后一个是 isColor 标签。如果是 True，每一帧就是彩色图，否则就是灰度图
FourCC 就是一个 4 字节码，用来确定视频的编码格式。
可用的编码列表 可以从fourcc.org查到。这是平台依赖的。
在Windows上常用的是DIVX。FourCC码以cv.VideoWriter_fourcc('D', 'I','V', 'X')形式传给程序
'''


# 保存视频
def video_write():
    cap = cv.VideoCapture(0)

    fourcc = cv.VideoWriter_fourcc('D', 'I', 'V', 'X')
    # 参数说明：输出视频名称，编码格式，播放频率，帧的大小
    out = cv.VideoWriter("../code_images/ouput.avi", fourcc, 20.0, (640, 480))

    while cap.isOpened():# 你可以使用 cap.isOpened()，来检查是否成功初始化了
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv.imshow("frame", frame)
            if cv.waitKey(1):
                break
        else:
            break


def main():
    # 读取图片,可以通过设置第二个参数来将一张彩色图像读取为一张灰度图
    img = cv.imread("../code_images/lena.jpg", cv.IMREAD_GRAYSCALE)

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    # cv.imshow("lena", img)

    get_image_info(img)
    # save_image(img)

    image_read()

    # video_read()

    # video_write()

    # 等待时间，毫秒级，1000ms后自动将窗口消除,0表示任意键终止
    cv.waitKey(1000)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
