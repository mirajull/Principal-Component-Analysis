import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

layerDimensions = [3, 4]
labelNo = 0


featureTrainList = []
labelTrain = []
labelTrainList = []

meanVector = []
covVector = []

featureTestList = []
labelTest = []


def readTrain():
    file = open("online.txt", "r")
    trainDataSize = -1
    while True:
        trainDataSize += 1
        featureTrain = []
        data = file.readline()
        if data == "":
            break
        datanow = data.split()
        featureNo = len(datanow)
        for i in range(0, featureNo):
            featureTrain.append(float(datanow[i]))
        featureTrainList.append(featureTrain[:])


def pca():
    #covariance matrix
    s = np.cov(np.matrix(featureTrainList).T)
    eig_vals, eig_vecs = np.linalg.eig(s)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda tup: tup[0])
    eig_pairs.reverse()
    pca1 = eig_pairs[0][1]
    pca2 = eig_pairs[1][1]
    lowerDimensions = np.vstack([pca1, pca2])
    projection = np.dot(lowerDimensions,list(map(list, zip(*featureTrainList))))

    plt.figure(1)
    plt.scatter(projection[0], projection[1])
    plt.title("Principle Component Comparison")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    print('Principal Component Analysis is done!')
    print('See the figure!')
    plt.savefig('f:/pca.png')
    plt.show()

    data = list(map(list, zip(*projection)))
    return data


def clustering(data, k):
    weightVector = []
    # Initialization Step
    for i in range(0, k):
        weightVector.append(1/3)
    min1 = min(list(map(list, zip(*data)))[0])
    max1 = max(list(map(list, zip(*data)))[0])
    min2 = min(list(map(list, zip(*data)))[1])
    max2 = max(list(map(list, zip(*data)))[1])

    for i in range(0, k):
        meanVector.append([np.random.uniform(min1, max1, 1)[0], np.random.uniform(min2, max2, 1)[0]])

    paramValue = [[1/len(data)]*k]*len(data)

    covVector = []

    for i in range(0, k):
        covVector.append([[5.0, 0.0], [0.0, 5.0]])

    # log likelihood

    total1 = 0
    for i in range(0, len(data)):
        summ = 0
        for j in range(0, k):
            summ += weightVector[j]*multivariate_normal.pdf(data[i], meanVector[j], covVector[j])
        total1 += np.log(summ)

    turn = 0

    while True:
        # Expectation Step
        turn += 1

        for i in range(0, len(data)):
            sumParam = 0.0
            for j in range(0, k):
                paramValue[i][j] = weightVector[j]*multivariate_normal.pdf(data[i], meanVector[j], covVector[j])
                sumParam += paramValue[i][j]
            paramValue[i] = [x/sumParam for x in paramValue[i]]

        # Maximization Step

        # mean update
        for i in range(0, k):
            temp1 = [0, 0]
            temp2 = 0
            for j in range(0, len(data)):
                abc = [x * paramValue[j][i] for x in data[j]]
                temp1 = [x+y for x,y in zip(temp1, abc)]
                temp2 += paramValue[j][i]
            meanVector[i] = [x/temp2 for x in temp1]

        # covariance update

        for i in range(0, k):
            temp1 = [[0, 0],[0,0]]
            temp2 = 0
            for j in range(0, len(data)):
                mat1 = np.subtract(data[j], meanVector[i])
                mat2 = np.matmul(np.matrix(mat1).T, np.matrix(mat1)).tolist()
                abc = [[x*paramValue[j][i] for x in b] for b in mat2]
                temp1 = [[x[0]+y[0], x[1]+y[1]] for x,y in zip(temp1, abc)]
                temp2 += paramValue[j][i]
            covVector[i] = [[x/temp2 for x in b] for b in temp1]

        # weight update

        for i in range(0, k):
            temp = 0
            for j in range(0, len(data)):
                temp += paramValue[j][i]
            weightVector[i] = temp/len(data)

        # Evaluate the log likelihood and check for convergence

        total2 = 0
        for i in range(0, len(data)):
            summ = 0
            for j in range(0, k):
                eps = np.finfo(float).eps
                summ += weightVector[j] * multivariate_normal.pdf(data[i], meanVector[j], covVector[j])
            total2 += np.log(summ)

        # convergence check
        if total2 - total1 < 0.00005:
            break

        total1 = total2

    return paramValue, k, weightVector, meanVector


def main():
    readTrain()
    data = pca()
    paramValue, k, weight, mean = clustering(data, 4)
    plt.figure(2)
    c = ['r','g','b','y']
    a = 0
    for i in range(0, len(data)):
        mx = -100.0
        mxind = 0
        for j in range(0, k):
            if paramValue[i][j] > mx:
                mx = paramValue[i][j]
                mxind = j
        plt.scatter(data[i][0], data[i][1], color=c[mxind])
    print('\nEM Algorithm is done!\n')
    plt.title("Clusters by EM Algorithm")
    plt.xlabel("Projection 1")
    plt.ylabel("Projection 2")
    for i in range(0, k):
        print('Cluster ',i+1,' mean: ',mean[i][0],' , ', mean[i][1])
        print('Cluster ',i+1, 'mixing coefficient: ', weight[i])
        print('Cluster ', i+1, 'points number: ',weight[i]*len(data))
        plt.scatter(mean[i][0], mean[i][1], edgecolors='w' ,color='black')
        print()
    plt.savefig('f:/em.png')
    print('total points: ',len(data))
    plt.show()


main()