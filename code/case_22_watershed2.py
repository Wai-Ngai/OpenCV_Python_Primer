import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

'''
基于距离变换，找到种子点

基于距离的分水岭分割流程：（需要去除噪声）
输入图像->灰度化->二值化->距离变换->寻找种子->生成marker->分水岭变换->输出图像
'''

img = cv.imread("../code_images/coins.png")
image=img.copy()
# cv.GaussianBlur(img,(5,5),0)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
cv.imshow("binary", binary)

# 使用形态学开运算去除白噪声
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
cv.imshow("open", opening)

# 使用膨胀去除对象上的小洞
# sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)
cv.imshow("sure background area", sure_bg)

# finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2,5)

ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
dist_transform=cv.normalize(dist_transform,0,1.0,cv.NORM_MINMAX)*50

cv.imshow("dist_transform", dist_transform)
cv.imshow("sure foreground area", sure_fg)

# finding unknown region
unknown = cv.subtract(sure_bg, sure_fg)
cv.imshow("unknown region", unknown)


# marker labelling
ret,markers1=cv.connectedComponents(sure_fg)

# add 0 to all labels ,so that sure background is not 0,but 1
markers = markers1+1

# mark the unknown region with 0
markers[unknown==255]=0

markers3=cv.watershed(image,markers)
image[markers3==-1]=[0,0,255]


cv.imshow("result",image)

plt.subplot(421), plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB), cmap="gray"), plt.title("imput_image"), plt.xticks([]), plt.yticks([])
plt.subplot(422), plt.imshow(binary, cmap="gray"), plt.title("binary"), plt.xticks([]), plt.yticks([])
plt.subplot(423), plt.imshow(opening, cmap="gray"), plt.title("opening"), plt.xticks([]), plt.yticks([])
plt.subplot(424), plt.imshow(sure_bg, cmap="gray"), plt.title("sure background area"), plt.xticks([]), plt.yticks([])
plt.subplot(425), plt.imshow(dist_transform, cmap="gray"), plt.title("dist_transform"), plt.xticks([]), plt.yticks([])
plt.subplot(426), plt.imshow(sure_fg, cmap="gray"), plt.title("sure foreground area"), plt.xticks([]), plt.yticks([])
plt.subplot(427), plt.imshow(unknown, cmap="gray"), plt.title("unknown"), plt.xticks([]), plt.yticks([])
plt.subplot(428), plt.imshow(cv.cvtColor(image,cv.COLOR_BGR2RGB), cmap="gray"), plt.title("result"), plt.xticks([]), plt.yticks([])

plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
