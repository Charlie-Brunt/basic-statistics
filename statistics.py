"""
Module for simple statistical calculations including PMCC, linear regression and testing for normal distributions

Author: Charlie-Brunt

"""

import numpy as np
import random
import matplotlib.pyplot as plt


def arithMean(x):
    """Returns arithmetic mean of for dependent and independent variables"""
    xbar = sum(x)/len(x)
    return xbar


def geoMean(x, f):
    """Returns geometric mean of x with frequencies f"""
    fx = np.array(x)*np.array(f)
    xbar = sum(fx) / sum(f)
    return xbar


def variance(x):
    """Returns standard deviation"""
    x2 = np.array(x)**2
    varx = arithMean(x2) - arithMean(x)**2
    return varx 


def stdDev(x):
    """Returns standard deviation"""
    return np.sqrt(variance(x))


def PMCC(x, y):
    """Returns PMCC for x and y lists"""
    if len(x) != len(y):
        raise "Dimension mismatch for x and y"
    x = np.array(x)
    y = np.array(y)
    Sxy = sum(x*y) - sum(x)*sum(y)/len(x)
    Sxx = sum(x**2) - (sum(x)**2)/len(x)
    Syy = sum(y**2) - (sum(y)**2)/len(x)
    r = Sxy/np.sqrt(Sxx*Syy)
    return r


def linearRegression(x, y):
    """Returns parameters alpha and beta to minimise squared error"""
    r = PMCC(x, y)
    sy = stdDev(y)
    sx = stdDev(x)
    beta = r*sy/sx
    alpha = arithMean(y) - beta*arithMean(x)
    return alpha, beta


def generateData(n=1000):
    """Linear data with noise"""
    c = random.random()
    k = random.random()*0.6 + 0.2
    x = np.linspace(0, 1.0, n)
    y = [i*k + c + (0.3*random.random()-0.15) for i in x]
    return x, y


def isNormal(x):
    """Return True if data follows normal distribution, only works for large data sets"""
    s = stdDev(x)
    mean = arithMean(x)
    n = len(x)
    a = 0
    for i in x:
        if i > (mean - s) and i < (mean + s):
            a += 1
    b = 0
    for i in x:
        if i > (mean - 2*s) and i < (mean + 2*s):
            b += 1
    c = 0
    for i in x:
        if i > (mean - 3*s) and i < (mean + 3*s):
            c += 1
    
    if  (0.66 < a/n < 0.70) and (0.93 < b/n < 0.97) and (0.98 < c/n < 1):
        return True
    else:
        return False


def generateNormal(mu, sigma, n=1000):
    """Generate normal variate data list"""
    values = [random.normalvariate(mu, sigma) for _ in range(n)]
    return values, mu, sigma


def normalPlot(data):
    values = data[0]
    mu = data[1]
    sigma = data[2]
    xlist = np.arange(mu - int(3*sigma), mu + int(3*sigma), 1)
    ylist = np.zeros(len(xlist))

    for i in range(len(xlist)):
        for val in values:
            if xlist[i]-0.5 <= val < xlist[i]+0.5:
                ylist[i] += 1
    
    return xlist, ylist


def main():
    n = 10
    x, y = generateData(n)

    alpha, beta = linearRegression(x, y)

    r = PMCC(x, y)
    print(f"r = {r}")

    xpts = np.linspace(0, 1, n) 
    ypts = alpha + beta*xpts

    plt.plot(x, y, "x")
    plt.plot(xpts, ypts ,"-r")
    plt.show()

    normalData = generateNormal(100, 15, 10000)
    xlist, ylist = normalPlot(normalData) 
    print(f"Generated data has normal distribution: {isNormal(normalData[0])}")
    plt.bar(xlist, ylist)
    plt.show()


if __name__ == '__main__':
    main()
