import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
傅里叶变换

    对于一个正弦信号，如果它的幅度变化非常快，我们可以说他是高频信号，如果变化非常慢，我们称之为低频信号。
    你可以把这种想法应用到图像中，图像那里的幅度变化非常大呢？边界点或者噪声。
    所以我们说边界和噪声是图像中的高频分量（注意这里的高频是指变化非常快，而非出现的次数多）。
    如果没有如此大的幅度变化我们称之为低频分量。
    
    - 低通滤波器LPF：只保留低频，会使得图像模糊
    - 高通滤波器HPF：只保留高频，会使得图像细节增强
    
    Numpy 中的傅里叶变换
        np.fft.fft2()
            对信号进行频率转换，输出结果是一个复杂的数组
            
            np.fft.fftshift()
                现在我们得到了结果，频率为0 的部分（直流分量）在输出图像的左上角。如果想让它（直流分量）在输出图像的中心，我们还需要将结果沿两个方向平移N/2
        
            np.fft.ifftshift() 
                进行逆平移操作，所以现在直流分量又回到左上角了，左后使用函数
            
        np.ifft2() 
            进行FFT 逆变换。
            
    OpenCV 中的傅里叶变换
        cv2.dft() 
            和前面输出的结果一样，但是是双通道的。第一个通道是结果的实数部分，第二个通道是结果的虚数部分。
            输入图像要首先转换成np.float32 格式
        
        cv2.idft()
  
  OpenCV 中的函数cv2.dft() 和cv2.idft() 要比Numpy 快。但是Numpy 函数更加用户友好。  
  
  当数组的大小是2的指数时DFT效率最高。当数组的大小是2，3，5 的倍数时效率也会很高。
  所以如果你想提高代码的运行效率时，你可以修改输入图像的大小（补0）。
  对于OpenCV 你必须自己手动补0(以创建一个大的0 数组，然后把我们的数据拷贝过去，或者使用函数cv2.copyMakeBoder())。
  但是Numpy，你只需要指定FFT 运算的大小，它会自动补0。
  
  cv2.getOptimalDFTSize()确定数组最佳大小
'''


# 使用numpy进行傅里叶变换
def fft_numpy(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    rows, cols = gray.shape
    crows, ccols = int(rows / 2), int(cols / 2)

    # 高通滤波HPF,只留下了一些高频的边界
    # 使用一个60x60 的矩形窗口对图像进行掩模操作从而去除低频分量。
    f_shift[crows - 20:crows + 20, ccols - 20:ccols + 20] = 0
    f_ishift = np.fft.ifftshift(f_shift)
    image_back = np.fft.ifft(f_ishift)
    image_back = np.abs(image_back)

    plt.subplot(231), plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB), cmap="gray"), plt.title("imput_image"), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(gray, cmap="gray"), plt.title("gray"), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(np.abs(f), cmap="gray"), plt.title("fft"), plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(np.abs(f_shift), cmap="gray"), plt.title("f_shift"), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(magnitude_spectrum, cmap="gray"), plt.title("magnitude_spectrum"), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(image_back, cmap="gray"), plt.title("image_back"), plt.xticks([]), plt.yticks([])

    plt.show()


# 使用OpenCV进行傅里叶变换
def fft_opencv(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 执行傅里叶变换，得到频谱图
    dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    # 将低频的值转换到中间的位置
    dft_shift = np.fft.fftshift(dft)
    # 得到灰度图能表示的形式，转换完的结果是非常小的值，所以还要做了一步骤，将结果映射到[0-255]
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # LPF（低通滤波）将高频部分去除,，只保留了低频，图像变得模糊
    rows, cols = gray.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

    mask = np.zeros((rows, cols, 2), np.uint8)  # 创建掩膜，进行滤波
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    fshift = dft_shift * mask  # 只留下了中间的一个区域

    f_ishift = np.fft.ifftshift(fshift)

    image_back = cv.idft(f_ishift)  # 还原回去的结果还是一个实部和虚部的
    image_back = cv.magnitude(image_back[:, :, 0], image_back[:, :, 1])

    plt.subplot(131), plt.imshow(gray, cmap="gray"), plt.title("imput_image"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(magnitude_spectrum, cmap="gray"), plt.title("magnitude_spectrum"), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(image_back, cmap="gray"), plt.title("image_back"), plt.xticks([]), plt.yticks([])

    plt.show()


def main():
    # 读取图片
    img = cv.imread("../code_images/face.jpg")

    # 创建窗口，窗口尺寸自动调整
    # cv.namedWindow("lena", cv.WINDOW_AUTOSIZE)

    # 显示图片
    # cv.imshow("lena", img)

    fft_numpy(img)

    fft_opencv(img)
    # 等待键盘输入
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
