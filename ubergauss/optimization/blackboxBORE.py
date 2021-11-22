import random
from scipy.special import softmax
import numpy as np
import multiprocessing as mp
from matplotlib.patches import Ellipse
from lmz import *
from ubergauss.tools import cartesian_product, xmap, maxnum
from sklearn import ensemble

'''
BORE optimizes by just using prediction rations for the pos/negative classes on a classifier that can
output probabilities...
'''


def get_grid(space):
        X = [np.linspace(a,b,100) for a,b in space]
        X = cartesian_product(X)
        return X

def get1index(clf):
    if clf.classes_[0] == 1:
        return 0
    else:
        return 1



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
            scores = xmap(f,pt)
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
            values = xmap(self.f,args)
            self.register(args, values)
            n_iter -= n_jobs

    ################
    # Fitting..
    ##################
    def fit(self, sample=0, draw=False):
        assert len(self.params) > 1, 'do some random guessing first'

        if 'model'  in self.__dict__:
            #self.space= [(x.min(),x.max()) for x in self.model[0].T]
            pass

        fpos = ensemble.RandomForestClassifier(n_estimators=50)
        fneg = ensemble.RandomForestClassifier(n_estimators=50)

        #fpos = ensemble.RandomForestRegressor(n_estimators=5)
        #fneg = ensemble.RandomForestRegressor(n_estimators=5)

        values = np.array(self.values)
        param = np.array(self.params)


        argsrt = np.argsort(self.values) # smallest to highest
        cut = max(int(len(self.values)/4),1)
        values = np.zeros(len(self.values))
        values[argsrt[:cut]] = 1

        #tools.binarize(self.values,.25) -> i  should be using this :)
        vinf = (values+1)%2
        fpos.fit(param, values)
        #fneg.fit(param, vinf)


        #for p,v in zip(param,values): print(f"{p} {myf(p)} {v}")

        '''
        get prediction for the whole space
        '''
        X = get_grid(self.space)


        #Y1,Y2 = fpos.predict_proba(X)[:,get1index(fpos)], fneg.predict_proba(X)[:,get1index(fneg)]
        pos = get1index(fpos)
        Y1,Y2 = fpos.predict_proba(X)[:,pos], fpos.predict_proba(X)[:, 0 if pos == 1 else 0]



        #Y1,Y2 = fpos.predict(X), fneg.predict(X)
        # TODO  Y2 being zero is a problem..
        #  with obvious solutions y2==0 -> picj definitely from there
        Y1f = Y1.flatten()
        Y2f = Y2.flatten()
        ie = Y1f/Y2f
        ie = np.where(ie != np.inf, ie, Y1f * maxnum(ie)*2)
        #ie = np.nan_to_num(ie,posinf=i, nan = i, neginf = i)
        ieNU = ie
        XNU = X

        ieNU = ieNU*ieNU.T

        # ieNU = np.log(ie)
        # XNU = X[ieNU > 0]
        # ieNU = ieNU[ieNU > 0]

        cut = np.median(ieNU)
        XNU = X[ieNU > cut]
        ieNU = ieNU[ieNU > cut]
        '''
        #ieNU = -ieNU
        #ieNU/=max(ieNU)
        #ieNU = ieNU*ieNU.T
        '''

        self.model = XNU, ieNU  # coordinates, pribabilities

        probe_here = 0
        if sample:
            probe_here = self.sample(sample)

        if draw:
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
