"""Author: ouk@andrew
   objectSegmentation.py
   Place in directory of your choosing. Will recursively find all .jpg files
   in your directory and segment the image from them, placing all new     segmented files in the TARGET_PATH"""

import cv2
import numpy as np
import os
import sys
INPUT_PATH = sys.argv[1]
TARGET_PATH = sys.argv[2]

if not os.path.exists(INPUT_PATH):
    os.makedirs(INPUT_PATH)

if not os.path.exists(TARGET_PATH):
    os.makedirs(TARGET_PATH)

CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 100

MARGIN = 300
RESULT_SIZE = 1002

BINARIZATION_THRESHOLD = 127

#From 112 TA notes code lol
def listJPEGFiles(path):
    if (os.path.isdir(path) == False):
        # base case:  not a folder, but a file, so return singleton list with its path
        if (path.endswith(".jpg")):
            return [path]
        else:
            return []
    else:
        # recursive case: it's a folder, return list of all paths
        files = [ ]
        for filename in os.listdir(path):
            files += listJPEGFiles(path + "/" + filename)
        return files

def main():
    files = listJPEGFiles(INPUT_PATH)
    print "Input path" + INPUT_PATH
    for file in files:
        img = cv2.imread(file)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blah, s, blah_ = cv2.split(imgHSV)

        edges = cv2.Canny(s, CANNY_THRESH_1, CANNY_THRESH_2)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)
        #cv2.imwrite('edges.jpg', edges)
        print "here"

        contour_info = []
        contours, blah = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))

        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0][0]
        (x,y,w,h) = cv2.boundingRect(max_contour)

        blank, newImage = cv2.threshold(s,BINARIZATION_THRESHOLD, 255,cv2.THRESH_BINARY)

        widthMargin = (RESULT_SIZE - w)//2
        heightMargin = (RESULT_SIZE - h)//2

        width = img.shape[1]
        height = img.shape[0]

        if (x - widthMargin < 0):
            widthMarginLeft = x
            widthMarginRight = RESULT_SIZE - widthMarginLeft - w
        elif (x + w + widthMargin > width):
            widthMarginRight = width - (x + w)
            widthMarginLeft = RESULT_SIZE - widthMarginRight - w
        else:
            if (((RESULT_SIZE - w) // 2) % 2 == 0):
                widthMarginLeft = widthMarginRight = widthMargin
            else:
                widthMarginLeft = widthMargin
                widthMarginRight = widthMargin + 1

        if (y - heightMargin < 0):
            heightMarginTop = y
            heightMarginBottom = RESULT_SIZE - heightMarginTop - h
        elif (y + h + heightMargin > height):
            heightMarginBottom = height - (y + h)
            heightMarginTop = RESULT_SIZE - heightMarginBottom - h
        else:
            if (((RESULT_SIZE - h) // 2) % 2 == 0):
                heightMarginTop = heightMarginBottom = heightMargin
            else:
                heightMarginTop = heightMargin
                heightMarginBottom = heightMargin + 1

        newImage = newImage[(y-heightMarginTop):y+h+heightMarginBottom,
                       (x-widthMarginLeft):x+w+widthMarginRight]
        print "blah"
        newImage = newImage[0:RESULT_SIZE-2, 0:RESULT_SIZE-2]
        print "meow"
        _, _, fileName = file.rpartition("/")

        newFileLocation = TARGET_PATH + "/" + "seg" + str(fileName)
        print("newImage size", newImage.shape[0], newImage.shape[1])

        cv2.imwrite(newFileLocation, newImage)

if __name__ == "__main__":
   main()
