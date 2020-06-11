# minist dataset
# training 60000
# testing 10000

import numpy as np
import time

def loadData(file):
    data  = []
    label = []
    with open(file, 'r') as d:
        for line in d.readlines():
            ds = line.strip().split(',')
            if int(ds[0]) >= 5:
                label.append(1)
            else:
                label.append(-1)
            data.append([int(num)/255 for num in ds[1:]])
    return data, label


def perception(train_data, train_label):
    train_data  = np.mat(train_data)
    train_label = np.mat(train_label).T

    # parameter initialization
    m, n = train_data.shape
    w    = np.zeros((1, n))
    b    = 0
    lr   = 0.0001
    iters = 50
    for iter in range(iters):
        for i in range(m):
            x  = train_data[i]
            y = train_label[i]
            if -1*y*(w*x.T+b) >= 0:
                w = w + lr*y*x
                b = b + lr*y

        print('Iter {} training'.format(iter))
    return w, b


def test(w, b, test_data, test_label):
    test_data  = np.mat(test_data)
    test_label = np.mat(test_label).T
    m, n = test_data.shape
    errorCnt = 0
    for i in range(m):
        x = test_data[i]
        y = test_label[i]
        if -1*y*(w*x.T+b) >= 0:
            errorCnt += 1

    accRate = 1- (errorCnt/m)
    return accRate

if __name__ == '__main__':
    training_file = './mnist_train.csv'
    testing_file  = './mnist_test.csv'
    train_data, train_label = loadData(training_file)
    test_data, test_label   = loadData(testing_file)
    # print(len(train_data), len(train_label))
    # print(len(test_data), len(test_label))
    w, b = perception(train_data, train_label)
    acc = test(w, b, test_data, test_label)
    print('accuracy rate is:', acc)