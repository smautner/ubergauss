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

def ex(d, out=2.576):
    """
        d: list of values(y) at a single x position

        return: [[outlier, low], 15.9 percentile, 50% percentile, 84.1 percentile, [outlier high]]
    """
    a,med,b = np.percentile(d, [15.9, 50, 84.1])
    oa = [v for v in d if v < med-out*(med-a)]
    ob = [v for v in d if v > med+out*(b-med)]

    return oa,a,med,b,ob



#################
# PUT STUFF TOGETHER
###############

def draw(data,smooth = False):
    """
        data: list of list of numbers :)

        does: plot median line, areafill+-sigma, outliers as +
    """
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





def boxplot(d,x =1):
    # how wide is the box?
    width = min(.5,.15*(max(d)-min(d)))

    # how tall is the box?
    oa,a,med,b,ob = ex(d)
    hight = b-a

    # draw the box
    rectangle = plt.Rectangle((x-.5*width,a), width, hight,ec="black",fill=False)
    plt.gca().add_patch(rectangle)
    # draw the median
    line = plt.Line2D((x-.5*width, x+.5*width), (med, med), lw=1.5, color='orange')
    plt.gca().add_line(line)
    # draw the outliers
    plt.scatter([x]*len(oa+ob), oa+ob, edgecolors='black', c='white')




def plotsmoothline(datax, datay):
    popt, pcov = curve_fit(f, datax ,datay)
    X = np.linspace(min(datax), max(datax), 100)
    plt.plot(X, map(lambda x:f(x,*popt), X))



def scatter(X,y):
    sns.scatterplot(x= X[:,0], y= X[:,1], hue = y )

if __name__ == "__main__":

    # import matplotlib
    # matplotlib.use("module://matplotlib-sixel")
    # import matplotlib.pyplot as plt

    data = readfile("../test/counts.txt")
    draw(data,False)
    plt.ylim((40,150))
    plt.show()

    boxplot([1,2,3,4,5,6,7,7,100])
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def binned_pairplot(df, score = 'score', bins= 15):

    g = sns.PairGrid(df)

    # Function to bin and average hue
    def binned_avg_hue(x, y, hue, bins=bins, **kwargs):
        if len(x) < 2: return

        # 2D histogram binning
        df = pd.DataFrame({'x': x, 'y': y, 'hue': hue})
        xbins = np.linspace(x.min(), x.max(), bins)
        ybins = np.linspace(y.min(), y.max(), bins)

        df['x_bin'] = np.digitize(df['x'], xbins)
        df['y_bin'] = np.digitize(df['y'], ybins)

        grouped = df.groupby(['x_bin', 'y_bin'])['hue'].mean().reset_index()

        # Convert bin indexes to bin centers
        grouped['x'] = xbins[grouped['x_bin'] - 1]
        grouped['y'] = ybins[grouped['y_bin'] - 1]

        # Normalize hue to [0,1] for colormap
        norm = Normalize(vmin=hue.min(), vmax=hue.max())
        plt.scatter(grouped['x'], grouped['y'], c=grouped['hue'], cmap='viridis', norm=norm, s=50)

    # Apply to lower triangle
    g.map_lower(lambda x, y, **kwargs: binned_avg_hue(x, y, df[score], **kwargs))

    # Optional: diagonal and upper
    g.map_diag(sns.histplot)
    g.map_upper(sns.scatterplot, hue=df[score])
    plt.colorbar()
    plt.show()





