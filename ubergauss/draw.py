from matplotlib import pyplot as plt
from lmz import *
import numpy as np
from scipy.optimize import curve_fit
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


###############
# regression:
##################
def f(x, a, b,c):
    return a * np.log(x+c) +b

def regr(d):
    popt, pcov = curve_fit(f, Range(d) , d)
    return popt



###################
# READ THE DATA
###################
def readfile(fname):
    # format: 1 row contains all y for one x
    r=[]
    for line in open(fname,'r').read().split('\n'):
        values=line.split()
        r.append(np.array(Map(float,values)))
    return r


#################
# fit 
###############
def ex(d, out):
    # return [ourliers],sigma1, median, sigma2 , [outliers]
    #med  = np.median(d)
    #a = [v for v in d if v >=med]
    #b = [v for v in d if v <=med]
    a,med,b = np.percentile(d, [15.9, 50, 84.1])

    #_,sa = stats.halfnorm.fit(a)
    #_,sb = stats.halfnorm.fit(-np.array(b))
    oa = [v for v in d if v < med-out*(med-a)]
    ob = [v for v in d if v > med+out*(b-med)]

    return oa,a,med,b,ob



#################
# PUT STUFF TOGETHER 
###############

def draw(data,smooth):
    # data is [[ally for first x],[all y for second x],...]


    stuff = [ex(rr,2.576) for rr in data]
    LR,LOW,MED,HI,HR = Transpose(stuff)
    X= Range(MED)


    p1,p2,p3 = [regr(x) for x in [LOW,MED,HI]]
    M2 =[f(x,*p2) for x in X]
    L2 =[f(x,*p1) for x in X]
    H2 =[f(x,*p3) for x in X]


    lo,med,hi = LOW, MED, HI 
    if smooth:
        lo,med,hi = L2,M2,H2

    # plot fillbetween
    plt.fill_between(X,lo,hi, color='green', alpha=0.4, label='median+-34.1th percentile')

    # plot median 
    plt.plot(X,med,label='median')

    # plot outlayers
    for i,(lr,hr) in enumerate(zip(LR,HR)):
        z=lr+hr
        if i==0:
            plt.scatter([i]*len(z), z,marker='x',c='b',alpha=.4,label='p<.99')
        else:
            plt.scatter([i]*len(z), z,marker='x',c='b',alpha=.4)

    plt.xticks(X)
    plt.legend()

    
    

if __name__ == "__main__":
    data = readfile("../test/counts.txt")
    draw(data,False)
    plt.ylim((40,150))
    plt.show()
