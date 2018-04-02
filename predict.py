from sklearn import svm
import numpy as np

INPUT_FILE = "./test/segtest.jpg"

def main():
    W = np.loadtxt('W.out')
    VT = np.loadtxt('VT.out')
    mean = np.loadtxt('mean.out')
    im = cv2.imread(INPUT_FILE)
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    X = gray_image.flatten() 
    XLDA = np.loadtxt('XLDA.out')