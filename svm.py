from sklearn import svm
import numpy as np
from sklearn.datasets import make_classification
import cv2
INPUT_FILE = "./test/segtest.jpg"

def main():
    XLDA = np.loadtxt('XLDA.out')
    print(np.shape(XLDA))
    Y = np.array([i//30 for i in range(np.shape(XLDA)[0])])
    clf = svm.LinearSVC()
    model = clf.fit(XLDA, Y)
    # W = np.loadtxt('W.out')
    # VT = np.loadtxt('VT.out')
    # mean = np.loadtxt('mean.out')
    im = cv2.imread(INPUT_FILE)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    X = gray_image.flatten() 
    D = len(X)
    X = np.array(X).reshape((D,1))
    model.predict([[1, 2]])
    # print(np.shape(X))
    # print(np.shape(VT))
    # XPCA = np.dot(VT, X.reshape((D,1)) - mean)


main()