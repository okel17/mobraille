from sklearn.metrics import confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from scipy import interp

y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
y_pred = np.array([2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0])
def makeConfusionMatrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    ball_cm = [[cm[0][0], cm[0][1]], [cm[1][0], np.sum([np.sum(row[1:]) for row in cm[1:]])]]
    return cm, ball_cm

def getTruePositives(y_test, y_pred, c):
    #y_score = y_score.reshape(-1, 1)
    tp = 0
    for i in range(np.size(y_pred)):
        if(y_test[i] == c and y_pred[i] == c):
            # positive value was correctly identified as positive
            tp += 1
    return tp

def getTruePositiveRate(y_true, y_score, threshold):
    return (getTruePositives(y_true, y_score, threshold)
            /np.sum(y_true))

def getFalsePositives(y_test, y_pred, c):
    fp = 0
    for i in range(np.size(y_pred)):
        if(y_test[i] != c and y_pred[i] == c):
            # negative value was incorrectly identified as positive
            fp += 1
    return fp

def getFalsePositiveRate(y_true, y_score, threshold):
    return (getFalsePositives(y_true, y_score, threshold)
            /(np.size(y_true) - np.sum(y_true)))

def getFalseNegatives(y_test, y_pred, c):
    fn = 0
    for i in range(np.size(y_pred)):
        if(y_test[i] == c and y_pred[i] != c):
            # positive value was incorrectly identified as negative
            fn += 1
    return fn

print(makeConfusionMatrix(y_test, y_pred))

