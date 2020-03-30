import cv2 as cv
from matplotlib import pyplot as plt

img=cv.imread("../code_images/target3.jpg")
cv.imshow("c",img)

# b,g,r=cv.split(img)
# img2=cv.merge([r,g,b])

# img2=img[...,::-1]

img2=cv.cvtColor(img,cv.COLOR_BGR2RGB)

plt.imshow(img2)
plt.xticks([])
plt.yticks([])
plt.show()
cv.waitKey(0)