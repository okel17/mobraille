import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt
import math

from sklearn import svm

import sys

DEBUG = False

def createX(path):
    X = []
    for file in os.listdir(path):
        im = cv2.imread(path + os.sep + file)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        X.append(gray_image)
    return np.array(X)

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

def makeNewImage(pixels):
    image = [[255] * 1000 for i in range(1000)]
    for row in range(1000):
        for col in range(1000):
            if((row, col) in pixels):
                image[row][col] = 0
    cv2.imwrite("test.jpg", np.array(image))
    img = cv2.imread("test.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('perimeter', gray)
    cv2.waitKey(0)

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

def getAmplitude(distances):
    return max(distances) - min(distances)

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
    return np.roll(signal, maxshift)
        

def makeTemplate(X, index, filename):
    showImage = True
    blur = cv2.blur(X[index], (5,5))
    pixels = findPerimeter(blur)

    if (showImage):
        cv2.imshow('blurred_image', blur)
        cv2.waitKey(0)
    
    centroid = findCentroid(pixels)
    distances = findDistances(centroid, pixels)
    data = np.array(distances)[:,1]
    plt.plot(data)
    plt.show()



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

def main():
    newData = []
    print(sys.argv)
    # if(sys.argc < 2):
    #     print("Not enough input arguments!!")
    X = createX('segmented_data_train')
    #im = cv2.imread(sys.argv[1])
    ball_template = np.load("ball1_template.npy")
    block_template = np.load("block3_template.npy")
    spike_template = np.load("spike4_template.npy")
    for image in range(len(X)):
        if(image < 25):
            template = np.load("ball1_template.npy")
        elif(image < 50):
            template = np.load("block3_template.npy")
        else:
            template = np.load("spike4_template.npy")
        template = template - np.mean(template)
        data = computeData(X[image])
        shiftedData = shiftData(data, template)
        newData.append(shiftedData)
        if(DEBUG):
            plt.plot(data, 'r')
            plt.plot(shiftedData, 'g')
            plt.plot(template, 'b')
            plt.show()
    newData = np.array(newData)
    np.save("distance_data.npy", newData)

    Y = np.array([i//25 for i in range(75)])
    clf = svm.LinearSVC(random_state=0, dual=False)
    model = clf.fit(newData, Y)

    print(sys.argv)
    # if(sys.argc < 2):
    #     print("Not enough input arguments!!")

    predicted_labels = []
    for file in os.listdir("segmented_data_test"):
        print(file)
        im = cv2.imread("segmented_data_test"+ os.sep + file)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        test = computeData(gray_image)
        testBall = shiftData(test, ball_template)
        testBlock = shiftData(test, block_template)
        testSpike = shiftData(test, spike_template)
        c1 = model.predict([testBall])[0]
        c2 = model.predict([testBlock])[0]
        c3 = model.predict([testSpike])[0]
        #normalize?
        balls = model.decision_function([testBall])
        blocks = model.decision_function([testBlock])
        spikes = model.decision_function([testSpike])
        confBall = balls[0][0]
        confBlock = blocks[0][1]
        confSpike = spikes[0][2]
        print(confBall,confBlock, confSpike)
        if(max(confBall,confBlock, confSpike) == confBall):
            print("ball")
        elif(max(confBall,confBlock, confSpike) == confBlock):
            print("block")
        else:
            print("spike")

        #accuracy for this attempt is 3/5 balls, 5/5 blocks, 2/5 spikes

        print(c1, c2, c3)
        print()

    return 0


main()



#the best comes from non-normalized with ball1 and variance under 200 and also cross correleation not normalized
