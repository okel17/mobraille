import numpy as np
import sys
import os
import cv2

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

def main():
    X_transpose = createX('toy_data')
    X = X_transpose.transpose()
    D, N = np.shape(X)
    assert(np.shape(X) == (D,N))
    #save some time getting gram matrix if the dataset is big
    #currently not using with toy data
    if(not os.path.exists('transpose_toy.out')):
        gram_matrix = np.dot(X_transpose, X)
        np.savetxt('transpose_toy.out', newMatrix)
    else:
        gram_matrix = np.loadtxt('transpose_toy.out')

    #get the mean across all samples for each dimension
    means = np.mean(X, axis=1).reshape((D, 1))

    #get the eigenvalues and eigenvectors of the gram matrix
    eigenvalues, eigenvectors_ = np.linalg.eig(np.dot(X_transpose, X))

    #sort the eigenvectors by their corresponding eigenvalues
    eigenvectors_ = np.array([vec for _,vec in sorted(zip(eigenvalues, eigenvectors_))])

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
    return (data, V_transpose, means)
