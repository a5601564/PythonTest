import cv2
import numpy as np

img = cv2.imread("d:\lena.jpg")
cv2.imshow("lena",img)
cv2.waitKey(10000)