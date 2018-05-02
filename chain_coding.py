import numpy as np
import sys
import os
import cv2
#import matplotlib.pyplot as plt
import math

import sys


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
            if(image[newPixelRow][newPixelCol] > 0):
                currentPixel = (newPixelRow, newPixelCol)
                pixels.append(currentPixel)
                currentDirection = directionIndex
                break

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

def crossConvolution(signal, template):
    maximum = None
    maxshift = None
    correlations = []
    correlations = np.correlate(signal, template, "full")
    #plt.plot(signal, 'r', template, 'g')
    #plt.plot(correlations)
    #plt.show()
    # for shift in range(0, 360):
    #     shiftedTemplate = np.roll(template, shift)
    #     correlation = np.correlate(signal, shiftedTemplate)
    #     #print(correlation, np.dot(signal, shiftedTemplate))
    #     correlations.append(correlation[0])
    #     # if(shift % 60 == 0):
    #     #     plt.plot(signal, 'r', shiftedTemplate, 'g')
    #     #     plt.show()
    #     if(maximum == None or correlation[0] > maximum):
    #         maximum = correlation
    #         maxshift = shift
    # print(np.shape(correlations))
    # print(maxshift)
    # plt.plot(correlations)
    # plt.show()

    amplitude = getAmplitude(correlations)
    return max(abs(correlations))#amplitude#correlation


def makeTemplate(X, index, filename):
    showImage = False
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

    block_template = np.save(filename + ".npy", data)


def loadTemplates():
    spike_template = np.load("spike_template.npy")
    block_template = np.load("block3_template.npy")
    ball_template = np.load("ball_template.npy")
    spike_template = spike_template/(np.linalg.norm(spike_template))
    block_template = block_template/(np.linalg.norm(block_template))
    ball_template = ball_template/(np.linalg.norm(ball_template))
    return spike_template, block_template, ball_template

def classify(data):

    spike_template, block_template, ball_template = loadTemplates()

    amplitude = getAmplitude(data)
    variance = np.var(data)
    oldData = copy.deepcopy(data)
    ballCorrelation = crossConvolution(data, ball_template)
    blockCorrelation = crossConvolution(data, block_template)
    spikeCorrelation = crossConvolution(data, spike_template)
    maxCorrelation = max(ballCorrelation, blockCorrelation, spikeCorrelation)

    if(ballCorrelation == maxCorrelation):
        print("This is a ball!!!!")
        os.system("python ./braille/n_char.py " + '"ball"')
        return "ball"
    elif(blockCorrelation == maxCorrelation):
        print("This is a block!!!!")
        os.system("python ./braille/n_char.py " + '"block"')
        return "block"
    else:
        print("This is a spike!!!!")
        os.system("python ./braille/n_char.py " + '"spike"')
        return "spike"

def computeData(image):
    blur = cv2.blur(image, (5,5))
    pixels = findPerimeter(blur)

    # cv2.imshow('blurred_image', blur)
    # cv2.waitKey(0)

    centroid = findCentroid(pixels)
    distances = findDistances(centroid, pixels)
    data = np.array(distances)[:,1]
    data = data/(np.linalg.norm(data))
    return data

def main():
    print(sys.argv)
    # if(sys.argc < 2):
    #     print("Not enough input arguments!!")

    im = cv2.imread(sys.argv[1])
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    data = computeData(gray_image)
    return classify(data)

import random, copy
def getAccuracy():
    X = createX('segmented_data_train')
    dictionary = dict()

    spike_template = np.load("spike_template.npy")
    block_template = np.load("block3_template.npy")
    ball_template = np.load("ball_template.npy")
    spike_template = spike_template/(np.linalg.norm(spike_template))
    block_template = block_template/(np.linalg.norm(block_template))
    ball_template = ball_template/(np.linalg.norm(ball_template))

    for index in range(0, 75):
        print(index)
        try:
            blur = cv2.blur(X[index], (5,5))
            pixels = findPerimeter(blur)

            # cv2.imshow('blurred_image', blur)
            # cv2.waitKey(0)

            centroid = findCentroid(pixels)
            distances = findDistances(centroid, pixels)
            data = np.array(distances)[:,1]
            data = data/(np.linalg.norm(data))

            amplitude = getAmplitude(np.array(distances)[:,1])
            variance = np.var(data)
            oldData = copy.deepcopy(data)
            ballCorrelation = crossConvolution(data, ball_template)
            blockCorrelation = crossConvolution(data, block_template)
            spikeCorrelation = crossConvolution(data, spike_template)
            maxCorrelation = max(ballCorrelation, blockCorrelation, spikeCorrelation)
            print("ball", maxCorrelation, ballCorrelation)
            print("block", maxCorrelation, blockCorrelation)
            print("spike", maxCorrelation, spikeCorrelation)

            if(ballCorrelation == maxCorrelation):
                print("This is a ball!!!!")
                dictionary[index] = "ball"
            elif(blockCorrelation == maxCorrelation):
                print("This is a block!!!!")
                dictionary[index] = "block"
            else:
                print("This is a spike!!!!")
                dictionary[index] = "spike"
        except:
            continue

    print(dictionary)
    total = 0
    for key in dictionary:
        if(key < 25 and dictionary[key] == "ball"):
            total += 1
        elif(key < 50 and dictionary[key] == "block"):
            total += 1
        elif(key < 75 and dictionary[key] == "spike"):
            total += 1
    print('accuracy', total/len(dictionary))

    #makeNewImage(pixels)
    #cv2.imshow("blah", blur)

import time
start = time.time()
main()
end = time.time()
print(end-start)

# X = createX('segmented_data_train')
# makeTemplate(X, 27, "block3_template")

