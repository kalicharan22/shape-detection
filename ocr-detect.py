# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os


# load the example image and convert it to grayscale
image = cv2.imread('shapes.jpg')
imageProc = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# check to see if we should apply thresholding to preprocess the
# image
imageProc = cv2.threshold(imageProc, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
gray = cv2.medianBlur(imageProc, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, imageProc)



# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)

# show the output images
cv2.imshow("Image", image)
cv2.imshow("Output", imageProc)
cv2.waitKey(0)
