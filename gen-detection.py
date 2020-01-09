import numpy as np
import cv2
import imutils

font = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread('shapes.jpg')
img_proc = imutils.resize(img, width=300)
img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_proc = cv2.GaussianBlur(img_proc, (5, 5), 0)
ret,thresh = cv2.threshold(img_proc,127,255,1)

_, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, 0, (0), 5)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)

    x = approx.ravel()[0]
    y = approx.ravel()[1]

    if len(approx)==3:
        cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        cv2.putText(img, "Triangle", (x, y), font, 1, (0))
    elif len(approx)==4:
        cv2.drawContours(img,[cnt],0,(0,255,0),-1)
        cv2.putText(img, "square", (x, y), font, 1, (0))
    elif len(approx) > 15:
        cv2.drawContours(img,[cnt],0,(255,100,0),-1)
        cv2.putText(img, "circle", (x, y), font, 1, (0))

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

