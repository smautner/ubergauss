import random
from scipy.special import softmax
import numpy as np
import multiprocessing as mp
from matplotlib.patches import Ellipse
from lmz import *
import forest

'''
ok TPE is terrible so lets  go to bayesian...

- we have a function and a space to optimize...
- fit the trees with variance estimation
- predict...
- draw
- sample
- repeat
'''

def mpmap(func, iterable, chunksize=1, poolsize=5):
    """pmap."""
    pool = mp.Pool(poolsize)
    result = pool.map(func, iterable, chunksize=chunksize)
    pool.close()
    pool.join()
    return list(result)


class BAY:
    '''
    we should discuss how the space is defined:
    space = [(mi,ma),...],{a:(mi,ma)}
    '''
    def __init__(self,f, space, n_init =10):

        self.f = f
        self.space = space
        self.params = []
        self.values = []

        if n_init:
            pt  = [self.randomsample() for i in range(n_init)]
            scores = mpmap(f,pt)
            self.register (pt,scores)



    def register(self,params:list, values: list):
        self.params+=params
        self.values+= values

    def suggest(self, n= 5):
        pass

    def randomsample(self):
        # returns 1 random parameter set
        return [np.random.uniform(a,b) for a,b in self.space]


    def minimize(self,n_iter=20, n_jobs = 5):
        while n_iter > 0:
            args = self.suggest(n_jobs)
            values = mpmap(self.f,args)
            self.register(args, values)
            n_iter -= n_jobs

    ################
    # TPE model
    ##################
    def fit(self, sample=0, draw=False):
        assert len(self.params) > 1, 'do some random guessing first'

        if 'model'  in self.__dict__:
            #self.space= [(x.min(),x.max()) for x in self.model[0].T]
            pass

        f = forest.RandomForestRegressor(n_estimators = 50)
        f.fit(np.array(self.params), self.values)

        '''
        get prediction for the whole space
        '''
        X = [np.linspace(a,b,100) for a,b in self.space]
        X = cartesian_product(X)
        Y,S = f.predict(X, return_std= True)

        ie = Y-1*S # expected valies

        cut = np.median(ie)
        XNU = X[ie < cut]
        ieNU = ie[ie < cut]-cut
        ieNU = -ieNU
        ieNU/=max(ieNU)
        ieNU = ieNU*ieNU.T
        self.model = XNU, ieNU

        probe_here = 0
        if sample:
            probe_here = self.sample(sample)

        if draw:
            plt.figure(figsize=(9,6))

            plt.subplot(231)
            d2plot(np.array(self.params),self.values, title = 'sampled data')
            plt.subplot(232)
            d2plot(X,np.log2(Y), title = 'log predict')
            plt.subplot(233)
            d2plot(X,S, title = 'std')
            plt.subplot(234)
            d2plot(X,np.log(ie), title = 'log prediction - std')

            plt.subplot(235)
            d2plot(XNU,ieNU, title = 'sampleweight')

            if sample:
                plt.subplot(236)
                d2plot(np.array(probe_here),y='r', title = 'new samples')
            plt.show()
            plt.close()
        return probe_here


    def sample(self, num):
        # so we sample,,, ie are the weights...
        #return random.choices(self.model[0], self.model[1])[0]
        X,w = self.model
        sample = np.random.choice(Range(X),num,replace=False, p=w/sum(w))
        return [X[r] for r in sample]


    def bestsuggestion(self):
        # so we sample,,, ie are the weights...
        #return random.choices(self.model[0], self.model[1])[0]
        X,w = self.model
        sample = np.argmax(w)
        return X[sample]


def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


import matplotlib
matplotlib.use("module://matplotlib-sixel")
import matplotlib.pyplot as plt

def d2plot(X,y, done = False, title = None):
    pl=plt.scatter(X[:,1],X[:,0], c= y,s=9)
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.colorbar(pl)
    if title:
        plt.title(title)
    if done:
        plt.show()
        plt.close()

def myf(args):
    return abs(args[0]**2)+abs(args[1]**2)




if __name__ == "__main__":
    opti = BAY(myf,[(-100,100),(-100,100)],n_init = 5)


    if True:
        for i in range(5):
            pt = opti.fit(draw=True, sample = 3)
            score = [myf(x) for x in pt]
            opti.register(pt,score)

    if False:
        for i in range(10):
            pt  = opti.bestsuggestion()
            score = [myf(pt)]
            opti.register([pt],score)
            opti.fit(draw=True)
