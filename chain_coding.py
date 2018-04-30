import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt
import math


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
    print(len(perimeter))
    sample = len(perimeter)//360 + 1
    print(sample)
    distances = []
    for i in range(0, 360):
        pixelIndex = (sample*i)%len(perimeter)
        print(pixelIndex)
        pixelX, pixelY = perimeter[pixelIndex]
        pixDist = distance(centroidX, centroidY, pixelX, pixelY)
        distances.append((pixelIndex, pixDist))
    return distances

def getAmplitude(distances):
    return max(distances) - min(distances)


def crossConvolution(signal, template):
    maximum = None
    for shift in range(0, 360):
        shiftedTemplate = np.roll(signal, shift)
        correlation = np.correlate(signal, shiftedTemplate)
        if(maximum == None or correlation > maximum):
            maximum = correlation
    return correlation
        

import random
def main():
    X = createX('segmented_data_train')
    #index = random.randint(0, 74)
    index = 0
    blur = cv2.blur(X[index], (5,5))
    pixels = findPerimeter(blur)
    

    # cv2.imshow('blurred_image', blur)
    # cv2.waitKey(0)
    
    centroid = findCentroid(pixels)
    distances = findDistances(centroid, pixels)
    data = np.array(distances)[:,1]
    spike_template = np.load("spike_template.npy")
    block_template = np.load("block_template.npy")
    ball_template = np.load("ball_template.npy")
    plt.plot(np.array(distances)[:,1])
    #plt.show()
    
    print(len(data))
    amplitude = getAmplitude(np.array(distances)[:,1])
    variance = np.var(data)
    print("variance", variance)
    print("amplitude", amplitude)
    print(index)
    if(index < 25):
        print("ball")
    elif(index < 50):
        print("block")
    else:
        print("spike")

    if(variance < 100):
        print("we think this is a ball")

    print("ball", crossConvolution(data, ball_template))
    print("block", crossConvolution(data, block_template))
    print("spike", crossConvolution(data, spike_template))





    #makeNewImage(pixels)
    #cv2.imshow("blah", blur)

main()
