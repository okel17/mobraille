import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt
#from sklearn import svm
import math

#INPUT_FILE = "./segmented_data_test/segspike27.jpg"
DEBUG = False


def mode(L):
    d = dict()
    for elem in L:
        if(elem not in d):
            d[elem] = 0
        d[elem] += 1

    maximum = None
    c = None

    for key in d:
        val = d[key]
        if(maximum == None or val > maximum):
            maximum = val
            c = key
    return (c, d[key])

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def findDistances(centroid, perimeter):
    centroidX, centroidY = centroid
    sample = len(perimeter)//360 + 1
    distances = []
    for i in range(0, 360):
        pixelIndex = (sample*i)%len(perimeter)
        pixelX, pixelY = perimeter[pixelIndex]
        pixDist = distance(centroidX, centroidY, pixelX, pixelY)
        distances.append((pixelIndex, pixDist))
    return distances

def findChain(firstPixel, image):
    prevPixel = None
    currentPixel = None
    currentDirection = None
    pixels = []

    prevPixel = firstPixel
    # directions = [(-1, -1), (-1, 0), (-1, 1), 
    #               (0, -1),           (0, 1),
    #               (1, -1),  (1, 0),  (1, 1)]
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), 
        (1, 1), (1, 0), (1, -1), (0, -1)]
    count = 0
    for directionIndex in range(len(directions)):
        direction = directions[directionIndex] 
        newPixelRow = prevPixel[0] + direction[0]
        newPixelCol = prevPixel[1] + direction[1]
        if(image[newPixelRow][newPixelCol] > 0):
            currentDirection = directionIndex
            break
    currentPixel = (newPixelRow, newPixelCol)
    pixels.append(currentPixel)
    while(currentPixel != firstPixel):
        count += 1
        startingDirection = (currentDirection + 5) % 8
        for directionOffset in range(len(directions)):
            directionIndex = (startingDirection + directionOffset) % 8
            direction = directions[directionIndex]
            newPixelRow = currentPixel[0] + direction[0]
            newPixelCol = currentPixel[1] + direction[1]
            if(newPixelRow >= 1000 or newPixelCol >= 1000 or 
                newPixelRow < 0 or newPixelCol < 0):
                continue
            if(image[newPixelRow][newPixelCol] > 0):
                currentPixel = (newPixelRow, newPixelCol)
                pixels.append(currentPixel)
                currentDirection = directionIndex
                break
    #makeNewImage(pixels)
    return pixels

def findCentroid(pixels):
    avgX = 0
    avgY = 0
    for pixel in pixels:
        pixelX, pixelY = pixel
        avgX += pixelX
        avgY += pixelY
    return (avgX//len(pixels), avgY//len(pixels))


def findPerimeter(image):
    pixelFound = False
    for rowIndex in range(len(image)//2, len(image)):
        for colIndex in range(len(image[0])):
            pixel = image[rowIndex][colIndex]
            if(pixel > 0):
                firstPixel = (rowIndex, colIndex)
                chain = findChain(firstPixel, image)
                pixelFound = True
            if(pixelFound):
                break
        if(pixelFound):
            break
    return chain


def computeData(image):
    blur = cv2.blur(image, (5,5))
    pixels = findPerimeter(blur)
    if(DEBUG):
        cv2.imshow('blurred_image', blur)
        cv2.waitKey(0)
    
    centroid = findCentroid(pixels)
    distances = findDistances(centroid, pixels)
    data = np.array(distances)[:,1]
    #data = data/(np.linalg.norm(data))
    data = data - np.mean(data)

    if(DEBUG):
        plt.plot(data)
        plt.show()
    return data


def findMinimum(d):
    smallest = None
    smallestIndex = None
    for key in d:
        if(smallest == None or d[key] < smallest):
            smallest = d[key]
            smallestIndex = key
    return smallestIndex

def knn(clusters, test, k):
    d = dict()
    classes = []
    for index in range(len(clusters)):
        d[index] = distance(clusters[index][0], clusters[index][1], test[0][0], test[0][1])
    for neighbor in range(k):
        index = findMinimum(d)
        classes.append(index//25)
        del d[index]
    return classes


def shiftData(signal, template):
    maximum = None
    maxshift = None
    if(DEBUG):
        #plt.plot(signal, 'r', template, 'g')
        plt.plot(correlations)
        plt.show()
    for shift in range(0, 360):
        shiftedSignal = np.roll(signal, shift)
        correlation = np.dot(template, shiftedSignal)
        if(maximum == None or correlation > maximum):
            maximum = correlation
            maxshift = shift 
    return (np.roll(signal, maxshift), correlation)

def getAccuracy():
    count = 0
    ballCount = 0
    blockCount = 0
    spikeCount = 0
    for file in os.listdir("segmented_data_test"):
        print(file)
        inp = "segmented_data_test" + os.sep + file
        if(count < 5 and main(inp) == "ball"):
            ballCount += 1
        elif(count < 10 and count >= 5 and main(inp) == "block"):
            blockCount += 1
        elif(count < 15 and count >= 10 and main(inp) == "spike"):
            spikeCount += 1
        count += 1
    print(ballCount, blockCount, spikeCount)


def main():
    D, C = 360, 3
    W = np.load("W.npy")
    VT = np.load("VT.npy")
    XLDA = np.load("XLDA.npy")
    means = np.load("means.npy")
    print("here")
    inp = sys.argv[1]
    im = cv2.imread(inp)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    X = computeData(gray_image)
    ball_template = np.load("ball1_template.npy")
    block_template = np.load("block3_template.npy")
    spike_template = np.load("spike4_template.npy")
    testBall, corball = shiftData(X, ball_template)
    testBlock, corblock = shiftData(X, block_template)
    testSpike, corspike= shiftData(X, spike_template)
    if(max(corball, corblock, corspike) == corball):
        print("correlation with ball")
        X = testBall
    elif(max(corball, corblock, corspike) == corblock):
        print("correlation with block")
        X = testBlock
    else:
        print("correlation with spike")
        X = testSpike
    X = np.array(X).reshape((D,1))
    XPCAi = np.dot(VT, X.reshape((D,1)) - means)
    XPCAi = XPCAi[:C, :]
    XLDAi = np.dot(XPCAi.T, W)
    class_mapper = {0: "ball", 1: "block", 2: "spike"}
    plt.plot(XLDA[0:25, 0:1], XLDA[0:25, 1:2], 'ro')
    plt.plot(XLDA[25:50, 0:1], XLDA[25:50, 1:2], 'bo')
    plt.plot(XLDA[50:75, 0:1], XLDA[50:75, 1:2], 'go')
    plt.plot(XLDAi[0][0], XLDAi[0][1], 'yo')

    #plt.show()
    k = 7
    classes = knn(XLDA, XLDAi, k)
    print(mode(classes))
    return class_mapper[mode(classes)[0]]


main()
#getAccuracy()
