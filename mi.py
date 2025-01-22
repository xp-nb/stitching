from stitching import Stitcher
import cv2

# 初始化 Stitcher
stitcher = Stitcher()

# 加载图像
image1 = cv2.imread("img/input/img1.jpg")
image2 = cv2.imread("img/input/img2.jpg")


# 拼接图像
panorama = stitcher.stitch([image1, image2])

# 保存结果
# cv2.imshow("result",panorama)
# cv2.waitKey(0)