import random
from scipy.special import softmax
import numpy as np
import multiprocessing as mp
from lmz import *
from ubergauss.tools import cartesian_product, xmap, maxnum, binarize
from sklearn import ensemble

'''
We use BORE but make it nicer...
'''





class Minimizer:
    '''
    we should discuss how the space is defined:
    space = [(min,max),...]


    -infer space
    -
    '''


    def __init__(self,f, space = None):
        self.f = f
        self.params = []
        self.values = []
        self.space = space
        self.model= None

    def infer_space(self, data):
        self.space = [(min(data[:, i]), max(data[:, i])) for i in range(data.shape[1])]

    def init_random(self,n_init):
            pt  = [self.randomsample() for i in range(n_init)]
            scores = xmap(self.f,pt)
            self.register(pt,scores)

    def register(self,params:list, values: list):
        self.params+=params
        self.values+= values

    def randomsample(self):
        return [np.random.uniform(a,b) for a,b in self.space]






    ################
    # Fitting..
    ##################
    def fit(self):
        assert len(self.params) > 1, 'use .register() or init_random() so we have some points to learn from'

        fpos = ensemble.RandomForestClassifier(n_estimators=50)
        X = np.array(self.params)
        Y = binarize(self.values, .25)
        fpos.fit(X, Y)
        self.model = fpos

    def get_grid(self, space, points=100):
        X = [np.linspace(a, b, points) for a, b in space]
        X = cartesian_product(X)
        return X

    def make_sample_terain(self, points_per_dim= 100):
        '''
        get prediction for the whole space
        '''
        X = self.get_grid(self.space, points_per_dim)
        pos = get1index(self.model)
        YALL = self.model.predict_proba(X)
        Y1, Y2 = YALL[:,pos], YALL[:, 0 if pos == 1 else 0]
        # TODO  Y2 being zero is a problem..
        #  with obvious solutions y2==0 -> picj definitely from there
        ie = Y1.flatten()/Y2.flatten()
        ie = np.where(ie != np.inf, ie, Y1f * maxnum(ie)*2)
        ie*=ie.T # squaring to shift the probabilities around
        cut = np.median(ie)
        X = X[ie > cut]
        ie = ie[ie > cut]
        self.sample_terain = X,ie

    def sample(self, num, draw= False):
        X, w = self.sample_terain
        sample = np.random.choice(Range(X), num, replace=False, p=w / sum(w))
        samples  =  [X[r] for r in sample]

        if draw:
            self.draw_terain_and_samples(self, samples)
        return samples

    def draw_terain_and_samples(self, samples):

            xylim = self.space
            plt.figure(figsize=(9,6))
            plt.subplot(231)
            d2plot(np.array(self.params),self.values, title = 'sampled data', xylim = xylim)
            plt.subplot(232)
            d2plot(X,Y1, title = 'POS', xylim = xylim)
            plt.subplot(233)
            d2plot(X,Y2, title = 'NEG', xylim = xylim)
            plt.subplot(234)
            d2plot(X,ie, title = 'pos/neg', xylim = xylim)
            plt.subplot(235)
            d2plot(XNU,ieNU, title = 'sampleweight', xylim = xylim)
            if sample:
                plt.subplot(236)
                d2plot(np.array(probe_here),y='r', title = 'new samples', xylim = xylim)
            plt.show()
            plt.close()


    def minimize(self,n_iter=20, n_jobs = 5):
        while n_iter > 0:
            args = self.suggest(n_jobs)
            values = xmap(self.f,args)
            self.register(args, values)
            n_iter -= n_jobs



    def bestsuggestion(self):
        # so we sample,,, ie are the weights...
        #return random.choices(self.model[0], self.model[1])[0]
        X,w = self.model
        sample = np.argmax(w)
        return X[sample]



def get1index(clf):
    if clf.classes_[0] == 1:
        return 0
    else:
        return 1

import matplotlib
matplotlib.use("module://matplotlib-sixel")
import matplotlib.pyplot as plt

def d2plot(X,y, done = False, title = None,xylim=None):
    pl=plt.scatter(X[:,0],X[:,1], c= y,s=9)
    plt.xlim(*xylim[0])
    plt.ylim(*xylim[1])
    plt.colorbar(pl)
    if title:
        plt.title(title)
    if done:
        plt.show()
        plt.close()




def myf(args):
    #return (args[0]**2)+(args[1]**2)
    return min(((args[0]+30)**2)+((args[1]+90)**2), ((args[0]-30)**2)+((args[1]-60)**2))




if __name__ == "__main__":
    opti = BAY(myf,[(-100,100),(-100,100)],n_init = 10)


    X = get_grid(opti.space)
    plt.scatter(X[:,0],X[:,1], c=Map(myf, X), s=9)
    plt.show();plt.close()

    if True:
        for i in range(6):
            pt = opti.fit(draw=True, sample = 5)
            score = [myf(x) for x in pt]
            opti.register(pt,score)

    if False:
        for i in range(10):
            pt  = opti.bestsuggestion()
            score = [myf(pt)]
            opti.register([pt],score)
            opti.fit(draw=True)
