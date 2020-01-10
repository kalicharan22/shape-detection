import cv2
import ocr_detect as ocr
import shape_detection as shapedet
import argparse
import sys


def resizeImage(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

def showImage(image):
    cv2.imshow('img', image)
    cv2.waitKey(0)


parser = argparse.ArgumentParser("image name parameter")
parser.add_argument("--image", help="Please define the image filename by adding --image parameter", type=str)
args = parser.parse_args()
filename = args.image
if(filename == None):
    print("Error: Wrong parmaters, Please define the image filename by adding --image parameter.")
    sys.exit()


# if(len(sys.argv) > 0):
#     filename = sys.argv[0]
# else:
#     print('Error: empty image name, please define the image filename by adding -image parameter')
#     sys.exit

# filename = 'images/shapes.jpg'

# OCR reading
ocrlist, image = ocr.read_ocr(filename)
for ocrobj in ocrlist:
    print("text: {}, {}, {}", ocrobj.text, ocrobj.x, ocrobj.y)

# Shape detection
image = shapedet.detect_shape(image, ocrlist)
image = resizeImage(image, 80)
showImage(image)
