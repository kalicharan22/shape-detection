# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

def showImage(img):
	# show the output image
	cv2.imshow("Image", img)
	cv2.waitKey(0)

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
# args = vars(ap.parse_args())

# imageName = 'shapes_and_colors.png'
imageName = 'shapes.jpg'

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(imageName)
# resize
img_proc = imutils.resize(image, width=700)
ratio = image.shape[0] / float(img_proc.shape[0])
# gray
img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
# blur
img_proc = cv2.GaussianBlur(img_proc, (5, 5), 0)
# threshold
thresh = cv2.threshold(img_proc,200,255,cv2.THRESH_BINARY)[1]

showImage(img_proc)
showImage(thresh)

# find contours in the thresholded image and initialize the
# shape detector 
cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

# loop over the contours
for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)
 
	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (0, 0, 0), 2)
	
	showImage(image)

