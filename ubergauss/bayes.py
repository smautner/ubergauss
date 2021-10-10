from lmz import *
import numpy as np
import matplotlib.pyplot as plt


'''
1d bayes, interpolating to best expected values between points
'''

def logistic(x):
    x/=2
    return np.exp(x)/(np.exp(x)+1)

def getavgslope(myx, ydata):
    slopes=[]
    for i in Range(1,myx):
        x1 = myx[i-1]
        x2 = myx[i]
        y1 = ydata[i-1]
        y2 = ydata[i]
        slope= abs(y1-y2)/abs(x1-x2)
        slopes.append(slope)
    meanslope = np.mean(slopes)
    return meanslope

def suggest(aruX, aruY,bravery =1 , plot = False):

    aruX, aruY = Transpose(sorted(zip(aruX,aruY)))

    meanslope = getavgslope(aruX, aruY)

    xspan = max(aruX) - min(aruX)
    yspan = max(aruY) - min(aruY)

    newyx = []
    for i in Range(1,myx):
        x1 = aruX[i-1]
        x2 = aruX[i]
        y1 = aruY[i-1]
        y2 = aruY[i]

        slope= (y2-y1)/abs(x1-x2)
        nuxx = x2-x1
        nuxx = logistic(slope/meanslope)*nuxx+x1

        nuyy = max(y1,y2) + bravery*logistic(-abs(slope)/meanslope)*((x2-x1)/(xspan))*yspan

        newyx.append((nuyy,nuxx))

    newyx.sort(reverse = True)
    y,x = Transpose(newyx)
    if plot:
        plt.scatter(aruX, aruY, label = 'given')
        plt.scatter(x,y, label='suggested')
        plt.legend()
        plt.show()
    return x,y
