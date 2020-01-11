import cv2 
import pytesseract
from pytesseract import Output
import numpy as np
from ocrobject import ocr_object


# resize image
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


def read_ocr(imageName):
    image = cv2.imread(imageName)

    # resize = resize_image(image, 120)
    gray = get_grayscale(image)
    thresh = thresholding(gray)
    openg = opening(gray)
    cann = canny(gray)

    # # Adding custom options
    # custom_config = r'--oem 3 --psm 6'
    # print(pytesseract.image_to_string(cann, config=custom_config))

    # h, w, c = image.shape
    # boxes = pytesseract.image_to_boxes(cann)
    # for b in boxes.splitlines():
    #     b = b.split(' ')
    #     img = cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)


    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(cann, output_type=Output.DICT, config=custom_config)
    # print(d.keys())
#    print(d['text'])

    ocr_object_list = []

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ocr_object_list.append(ocr_object(d['text'][i], x, y, w, h))

    return ocr_object_list, image
