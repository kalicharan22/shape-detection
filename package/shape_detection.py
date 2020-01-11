import numpy as np
import cv2
import imutils


font = cv2.FONT_HERSHEY_COMPLEX_SMALL

def detect_shape(img, ocrlist):
    # img = cv2.imread(imageName)
    img_proc = imutils.resize(img, width=300)
    img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_proc = cv2.GaussianBlur(img_proc, (5, 5), 0)
    _,thresh = cv2.threshold(img_proc,127,255,1)

    _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, 0, (0), 5)

    for i in range(0, len(contours)):
        # find largest contours
        area=cv2.contourArea(contours[i],False)
#        print(area)

        if(area > 100):
            cnt = contours[i]

            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if len(approx)==3:
                # cv2.drawContours(img,[cnt],0,(0,0,255),-1)
                cv2.putText(img, "2nd generation: is parent of ", (x, y), font, 1, (0))
            elif len(approx)==4:
                # cv2.drawContours(img,[cnt],0,(0,255,0),-1)
                cv2.putText(img, "3rd generation", (x, y), font, 1, (0))
            elif len(approx) > 14:
                # cv2.drawContours(img,[cnt],0,(255,100,0),-1)
                cv2.putText(img, "1st generation: is parent of ", (x, y), font, 1, (0))
    
    return img
