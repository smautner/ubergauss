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
    def fit(self, draw=False):
        assert len(self.params) > 1, 'do some random guessing first'

        if 'model'  in self.__dict__:
            #self.space= [(x.min(),x.max()) for x in self.model[0].T]
            pass

        f = forest.RandomForestRegressor(n_estimators = 500)
        f.fit(np.array(self.params), self.values)

        '''
        get prediction for the whole space
        '''
        X = [np.linspace(a,b,100) for a,b in self.space]
        X = cartesian_product(X)
        Y,S = f.predict(X, return_std= True)

        '''

        '''
        ie = Y-S # expected valies

        XNU = X[ie < min(self.values)]
        ieNU = ie[ie < min(self.values)]-min(self.values)
        ieNU = -ieNU
        ieNU/=max(ieNU)
        ieNU = ieNU*ieNU.T
        #ieNU = ieNU*ieNU.T

        # softmax
        #ieNU = softmax(ieNU)
        #ieNU= np.log2(ieNU)



        plt.figure(figsize=(9,3))
        plt.subplot(131)
        d2plot(np.array(self.params),self.values, title = 'sampled data')
        plt.subplot(132)
        d2plot(X,ie, title = 'prediction - 2*std')
        plt.subplot(133)
        d2plot(XNU,ieNU, title = 'sampleweight')
        plt.show()
        plt.close()

        self.model = XNU, ieNU

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
    opti.fit(draw=True)


    if False:
        for i in range(5):
            pt  = opti.sample(3)
            score = [myf(x) for x in pt]
            opti.register(pt,score)
            opti.fit(draw=True)

    if True:
        for i in range(10):
            pt  = opti.bestsuggestion()
            score = [myf(pt)]
            opti.register([pt],score)
            opti.fit(draw=True)
