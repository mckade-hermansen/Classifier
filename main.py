import matplotlib.pyplot as plt
import numpy as np
import math
import random as rand

def gendat2(c, N):
    m0 = [-0.132,  0.320, 1.672, 2.230,  1.217, -0.819, 3.629, 0.8210, 1.808, 0.1700, -0.711,
          -1.726, 0.139, 1.151, -0.373, -1.573, -0.243, -0.5220, -0.511, 0.5330]

    m1 = [-1.169, 0.813, -0.859, -0.608, -0.832, 2.015, 0.173, 1.432, 0.743, 1.0328, 2.065, 2.441,
          0.247, 1.806, 1.286, 0.928, 1.923, 0.1299, 1.847, -0.052]

    x = [[0,0]for i in range(N)]
    for i in range(N):
        idx = rand.randint(0, 19)
        if c == 0:
            m = m0[idx]
        elif c == 1:
            m = m1[idx]
        x[i][0] = m + np.random.normal()/math.sqrt(5)
        x[i][1] = m + np.random.normal()/math.sqrt(5)

    return x


def multMatricies(a,b):
    res = [[0 for i in range(len(b[0]))]for i in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for x in range(len(b)):
                val1 = float(a[i][x])
                val2 = float(b[x][j])
                res[i][j] += val1*val2

    return res

def printMatricy(a):
    for val in a:
        print(val)

def printTable(n, titles, data):
    print(" "*32, ":", "  ", "Error Rates", "  ")
    print("-"*50)
    print(" "*32, ":", "Training Test ")
    print("-" * 50)

    for i in range(n):
        print(" "*(30-len(titles[i])), titles[i], " :", data[i][0], " ", data[i][1])

def linearRegression(N0, N1, N, showGraph):
    returnVal = []

    # Create matricies
    X, Y = [], []

    # X
    X0Index, X1Index = 0, 0
    for i in range(N0):
        X.append([1, x0[i], y0[i]])
    for i in range(N1):
        X.append([1, x1[i], y1[i]])

    # Y
    for i in range(N0):
        Y.append([1, 0])
    for i in range(N1):
        Y.append([0, 1])

    XT = np.transpose(X)
    temp = np.matmul(XT, X)
    tempT = np.linalg.inv(temp)
    temp2 = multMatricies(temp, XT)
    Bhat = multMatricies(temp2, Y)
    Yhat = multMatricies(X, Bhat)
    Yhathard = Yhat
    printMatricy(Bhat)

    for i in range(len(Yhat)):
        for j in range(len(Yhat[0])):
            if Yhat[i][j] > 0.5:
                Yhathard[i][j] = 1
            else:
                Yhathard[i][j] = 0

    nerr = np.sum(abs(np.subtract(Yhathard, Y))) / 2
    errrate_linregress_train = nerr / N
    returnVal.append(errrate_linregress_train)

    Ntest0 = 5000
    Ntest1 = 5000
    nerr = 0
    xtest0 = gendat2(0, Ntest0)
    xtest1 = gendat2(0, Ntest1)

    for i in range(Ntest0):
        temp = np.transpose(xtest0[i])
        temp2 = []
        temp2.append(1)
        for val in temp:
            temp2.append(val)
        temp3 = np.array([temp2])
        Yhat = np.matmul(temp3, Bhat)
        if Yhat[0][1] > Yhat[0][0]:
            nerr += 1

    for i in range(Ntest1):
        temp = np.transpose(xtest1[i])
        temp2 = []
        temp2.append(1)
        for val in temp:
            temp2.append(val)
        temp3 = np.array([temp2])
        Yhat = np.matmul(temp3, Bhat)
        if Yhat[0][0] > Yhat[0][1]:
            nerr += 1

    errate_linregress_test = nerr / (Ntest0 + Ntest1)
    returnVal.append(errate_linregress_test)

    if showGraph:
        xtest0 = np.transpose(xtest0)
        xtest1 = np.transpose(xtest1)
        xmin = min(xtest0[0])
        xmax = max(xtest0[0])
        ymin = min(xtest0[1])
        ymax = max(xtest0[1])
        xpl = np.linspace(xmin, xmax, num=100)
        ypl = np.linspace(ymin, ymax, num=100)
        redx, redy, greenx, greeny = [], [], [], []

        for x in xpl:
            for y in ypl:
                temp = []
                temp.append(1)
                temp.append(x)
                temp.append(y)
                temp2 = np.array([temp])
                yhat = multMatricies(temp2, Bhat)
                if yhat[0][0] > yhat[0][1]:
                    redx.append(x)
                    redy.append(y)
                else:
                    greenx.append(x)
                    greeny.append(y)

        plt.plot(redx, redy, 'red', alpha=0.2)
        plt.plot(greenx, greeny, 'green', alpha=0.2)
        plt.show()

    return returnVal

def quadraticRegression(N0, N1, N, showGraph):
    returnVal = []

    # Create matricies
    X, Y = [], []

    # X
    X0Index, X1Index = 0, 0
    for i in range(N0):
        x0[i] = float(x0[i])
        y0[i] = float(y0[i])
        X.append([1, x0[i], y0[i], x0[i]**2, x0[i]*y0[i], y0[i]**2])
    for i in range(N1):
        x1[i] = float(x1[i])
        y1[i] = float(y1[i])
        X.append([1, x1[i], y1[i], x1[i] ** 2, x1[i] * y1[i], y1[i] ** 2])

    # Y
    for i in range(N0):
        Y.append([1, 0])
    for i in range(N1):
        Y.append([0, 1])

    XT = np.transpose(X)
    temp = multMatricies(XT, X)
    tempT = np.linalg.inv(temp)
    temp2 = multMatricies(temp, XT)
    Bhat = multMatricies(temp2, Y)
    Yhat = np.matmul(X, Bhat)
    Yhathard = Yhat

    for i in range(len(Yhat)):
        for j in range(len(Yhat[0])):
            if Yhat[i][j] > 0.5:
                Yhathard[i][j] = 1
            else:
                Yhathard[i][j] = 0

    nerr = np.sum(abs(np.subtract(Yhathard, Y))) / 2
    print(nerr)
    errrate_linregress_train = nerr / N
    returnVal.append(errrate_linregress_train)

    Ntest0 = 5000
    Ntest1 = 5000
    nerr = 0
    xtest0 = gendat2(0, Ntest0)
    xtest1 = gendat2(0, Ntest1)

    for i in range(len(xtest0)):
        temp = np.transpose(xtest0[i])
        temp3 = np.array([1, temp[0], temp[1], temp[0]**2, temp[0]*temp[1], temp[1]**2])
        Yhat = np.matmul(temp3, Bhat)
        if Yhat[1] > Yhat[0]:
            nerr += 1

    for i in range(len(xtest1)):
        temp = np.transpose(xtest1[i])
        temp3 = np.array([1, temp[0], temp[1], temp[0]**2, temp[0]*temp[1], temp[1]**2])
        Yhat = np.matmul(temp3, Bhat)
        if Yhat[0] > Yhat[1]:
            nerr += 1

    errate_linregress_test = nerr / (Ntest0 + Ntest1)
    returnVal.append(errate_linregress_test)

    if showGraph:
        xtest0 = np.transpose(xtest0)
        xtest1 = np.transpose(xtest1)
        xmin = min(xtest0[0])
        xmax = max(xtest0[0])
        ymin = min(xtest0[1])
        ymax = max(xtest0[1])
        xpl = np.linspace(xmin, xmax, num=100)
        ypl = np.linspace(ymin, ymax, num=100)
        redx, redy, greenx, greeny = [], [], [], []

        for x in xpl:
            for y in ypl:
                temp = []
                temp.append(1)
                temp.append(x)
                temp.append(y)
                temp2 = np.array([temp])
                yhat = multMatricies(temp2, Bhat)
                if yhat[0][0] > yhat[0][1]:
                    redx.append(x)
                    redy.append(y)
                else:
                    greenx.append(x)
                    greeny.append(y)

        plt.plot(redx, redy, 'red', alpha=0.2)
        plt.plot(greenx, greeny, 'green', alpha=0.2)
        plt.show()

    return returnVal

# read in the data
file = open("classasgntrain1.dat")
x0, x1, y0, y1 = [],[],[],[]
for line in file:
    curr = line.split()
    x0.append(curr[0])
    y0.append(curr[1])
    x1.append(curr[2])
    y1.append(curr[3])

N0 = len(x0)
N1 = len(x1)
N = N0 + N1

# plot the data
plt.figure(1)
plt.plot(x0, y0, 'ro')
plt.plot(x1, y1, 'go')
plt.axis([-5,5,-5,5])
#plt.show()

titles = ["Linear Regression", "Quadratic Regression"]
data = []
data.append(linearRegression(N0, N1, N, False))
data.append(quadraticRegression(N0, N1, N, False))
printTable(2, titles, data)







