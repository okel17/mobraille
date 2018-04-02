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

def PCA(X, X_transpose):
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
    return (np.squeeze(np.array(data)).T, V_transpose, means)

def labelData(X, X_transpose):
    classes = {0: X_transpose[0:3,:].T, 1: X_transpose[3:6, :].T, 2: X_transpose[6:9, :].T}
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
        print(np.shape(np.mean(class_data[c], axis=1)))
        class_means[c] = np.mean(class_data[c], axis=1).reshape((newD,1))
    new_mean = np.mean(XPCA, axis=1)
    S_W = getS_W(XPCA, class_data, class_means, newD)
    S_B = getS_B(new_mean, class_data, class_means, newD)
    S_W_inv = np.linalg.inv(S_W)
    eigenvalues, eigenvectors = np.linalg.eig(S_W_inv.dot(S_B))
    print(eigenvectors)
    eigenvectors = np.array([vec for _,vec in sorted(zip(eigenvalues, eigenvectors))])


def main():
    X_transpose = createX('toy_data')
    X = X_transpose.transpose()
    D, N = np.shape(X)
    assert(np.shape(X) == (D,N))
    XPCA, VT, means = PCA(X, X_transpose)
    C = 3
    newD, newN = np.shape(XPCA)
    #LDA(XPCA[0:(newN - C - 1), :])


main()


