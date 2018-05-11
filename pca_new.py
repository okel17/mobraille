import numpy as np
import sys
import os
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
import math

INPUT_FILE = "./segmented_data_test/segspike27.jpg"
TEST_FOLDER = "./segmented_data_test"
DEBUG = False

def createX(path):
    X = []
    for file in os.listdir(path):
        im = cv2.imread(path + os.sep + file)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        arr = gray_image.flatten() 
        X.append(arr)
    return np.array(X)

#used this for guidance: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#mean_vec
#toy dataset from here: https://github.com/Horea94/Fruit-Images-Dataset/tree/master/Training/

def PCA(X, X_transpose):
    D, N = np.shape(X)
    assert(np.shape(X) == (D,N))
    gram_matrix = np.dot(X_transpose, X)
    #save some time getting gram matrix if the dataset is big
    #currently not using with toy data
    # if(not os.path.exists('transpose_segmented.out')):
    #     gram_matrix = np.dot(X_transpose, X)
    #     np.savetxt('transpose_segmented.out', newMatrix)
    # else:
    #     gram_matrix = np.loadtxt('transpose_s.out')
    print('here')
    #get the mean across all samples for each dimension
    means = np.mean(X, axis=1).reshape((D, 1))

    #get the eigenvalues and eigenvectors of the gram matrix
    eigenvalues, eigenvectors_ = np.linalg.eig(gram_matrix)
    sorted_pairs = sorted(zip(eigenvalues, eigenvectors_))
    sorted_pairs.reverse()
    #sort the eigenvectors by their corresponding eigenvalues
    eigenvectors_ = np.array([vec for _,vec in sorted_pairs])

    #transform the eigenvectors to their covariance matrix counterparts
    V = []
    for v_ in eigenvectors_.T:
        v = np.dot(X, v_)
        norm = np.linalg.norm(v)
        #normalize the vector
        v = np.true_divide(v,norm)
        V.append(v)
    V = np.array(V).transpose()
    V_transpose = V.transpose()
    assert(np.shape(V_transpose) == (N, D))

    #get the new dataset
    data = []
    
    for image_index in range(len(X_transpose)):
        #project each image onto the eigenvectors
        XPCA = np.dot(V_transpose, X_transpose[image_index].reshape((D,1)) - means)
        data.append(XPCA)
    return (np.squeeze(np.array(data)).T, V_transpose, means)

def labelData(X, X_transpose):
    classes = {0: X_transpose[0:25,:].T, 1: X_transpose[25:50, :].T, 2: X_transpose[50:75, :].T}
    return classes

def getS_W(X, class_data, class_means, features):
    C = len(class_data)

    S_W = np.zeros((features,features))
    for c in class_data:
        class_scatter = np.zeros((features,features))
        for sample in class_data[c].T:
            xi = sample.reshape((len(sample), 1))
            mi = class_means[c]
            class_scatter += (xi - mi).dot((xi - mi).T)
        S_W += class_scatter
    return S_W

def getS_B(mean, class_data, class_means, features):
    mean = mean.reshape((features, 1))
    S_B = np.zeros((features, features))
    for c in class_data:
        num_samples = len(class_data[c][0])
        mi = class_means[c]
        S_B += num_samples * (mi - mean).dot((mi - mean).T)
    return S_B

def LDA(XPCA):
    #get within class scatter matrix
    newD, newN = np.shape(XPCA)
    class_data = labelData(XPCA, XPCA.T)
    class_means = dict()
    for c in class_data:
        class_means[c] = np.mean(class_data[c], axis=1).reshape((newD,1))
    new_mean = np.mean(XPCA, axis=1)
    S_W = getS_W(XPCA, class_data, class_means, newD)
    S_B = getS_B(new_mean, class_data, class_means, newD)
    S_W_inv = np.linalg.inv(S_W)
    eigenvalues, eigenvectors = np.linalg.eig(S_W_inv.dot(S_B))
    sorted_pairs = sorted(zip(eigenvalues, eigenvectors))
    sorted_pairs.reverse()
    eigenvectors = np.array([vec for _,vec in sorted_pairs])
    W = np.hstack((eigenvectors[0].reshape(newD,1), eigenvectors[1].reshape(newD,1))) #eigenvectors[2].reshape(newD,1)))
    return W

def classify(model, VT, means, W, C):
    predicted_labels = []
    for file in os.listdir(TEST_FOLDER):
        im = cv2.imread(TEST_FOLDER + os.sep + file)
        gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        X = gray_image.flatten() 
        D = len(X)
        X = np.array(X).reshape((D,1))
        XPCAi = np.dot(VT, X.reshape((D,1)) - means)
        newD, newN = np.shape(XPCAi)
        XPCAi = XPCAi[:C, :]
        XLDAi = np.dot(XPCAi.T, W)
        class_mapper = {0: "ball", 1: "block", 2: "spike"}
        c = model.predict(XLDAi)[0]
        predicted_labels.append(c)
    return predicted_labels


def classifyDistance(model, VT, means, W, C):
    for file in os.listdir("segmented_data_test"):
            print(file)
            im = cv2.imread("segmented_data_test"+ os.sep + file)
            gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            test = computeData(gray_image)
            XPCAi = np.dot(VT, X.reshape((D,1)) - means)
            newD, newN = np.shape(XPCAi)
            XPCAi = XPCAi[:C, :]
            XLDAi = np.dot(XPCAi.T, W)
            class_mapper = {0: "ball", 1: "block", 2: "spike"}
            c = model.predict(XLDAi)[0]


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



def getClass(XLDA, W, means, VT, D, C):
    print("the D", D)
    im = cv2.imread(INPUT_FILE)
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
    print(corball, corblock, corspike)
    X = np.array(X).reshape((D,1))
    XPCAi = np.dot(VT, X.reshape((D,1)) - means)
    XPCAi = XPCAi[:C, :]
    XLDAi = np.dot(XPCAi.T, W)
    class_mapper = {0: "ball", 1: "block", 2: "spike"}
    print(XLDAi)
    plt.plot(XLDA[0:25, 0:1], XLDA[0:25, 1:2], 'ro')
    plt.plot(XLDA[25:50, 0:1], XLDA[25:50, 1:2], 'bo')
    plt.plot(XLDA[50:75, 0:1], XLDA[50:75, 1:2], 'go')
    plt.plot(XLDAi[0][0], XLDAi[0][1], 'yo')

    plt.show()
    k = 7
    classes = knn(XLDA, XLDAi, k)
    print(classes)
    # print(index)
    # c = index//25
    #print(XLDAi, XLDA[index])

def main():
    X_transpose = np.load("distance_data.npy")
    X = X_transpose.transpose()
    D, N = np.shape(X)
    print(D, N)
    print('hey')
    assert(np.shape(X) == (D,N))
    XPCA, VT, means = PCA(X, X_transpose)
    print('done')
    C = 3
    newD, newN = np.shape(XPCA)
    #only keep N - C - 1 dimensions from PCA
    #(newN - C - 1)
    XPCA = XPCA[:C, :]
    #W = LDA(XPCA[0:(newN - C - 1), :])
    Wprime = LDA(XPCA)
    print(Wprime)
    W = np.real(Wprime)
    XLDA = np.real(XPCA.T.dot(W))
    print("shape of LDA =", np.shape(XLDA))
    

    # Y = np.array([i//25 for i in range(np.shape(XLDA)[0])])
    # clf = svm.LinearSVC(random_state=0, dual=False)
    # model = clf.fit(XLDA, Y)
    # Y_true = np.array([i//5 for i in range(15)])
    # Y_test = classify(model, VT, means, W, C)
    # print(Y_true)
    # print(Y_test)
    # total = 0
    # for i in range(len(Y_test)):
    #     if(Y_true[i] == Y_test[i]):
    #         total += 1
    # print("Accuracy = ", total/15)

    getClass(XLDA, W, means, VT, D, C)
    np.save("XLDA.npy", XLDA)
    np.save("W.npy", W)
    np.save("means.npy", means)
    np.save("VT.npy", VT)


    

main()


