
import numpy as np
from sklearn.neighbors import KernelDensity
import multiprocessing as mp
from matplotlib.patches import Ellipse
from lmz import *


def mpmap(func, iterable, chunksize=1, poolsize=5):
    """pmap."""
    pool = mp.Pool(poolsize)
    result = pool.map(func, iterable, chunksize=chunksize)
    pool.close()
    pool.join()
    return list(result)


class TPE:
    '''
    we should discuss how the space is defined:
    space = [(mi,ma),...],{a:(mi,ma)}
    '''
    def __init__(self,f, space, n_init =10, kde_top = 5):

        self.f = f
        self.space = space
        self.n_init = n_init
        self.params = []
        self.values = []
        self.kde_top = kde_top

    def register(self,params:list, values: list):
        self.params+=params
        self.values+= values

    def suggest(self, n= 5):
        if self.n_init > len(self.params):
            return [self.randomsample() for i in range(n)]
        else:
            self.TPEfit()
            return [self.TPEsample() for i in range(n)]

    def randomsample(self):
        # returns 1 random parameter set
        return [np.random.uniform(a,b) for a,b in self.space[0]],\
                   {k:np.random.uniform(a,b) for k,(a,b) in self.space[1].items()}


    def minimize(self,n_iter=20, n_jobs = 5):
        while n_iter > 0:
            args = self.suggest(n_jobs)
            values = mpmap(self.f,args)
            self.register(args, values)
            n_iter -= n_jobs

    ################
    # TPE model
    ##################
    def TPEfit(self):
        assert len(self.params) > 1, 'do some random guessing first'
        #num_v = int(len(self.values)/2)# or self.kde_top
        num_v = self.kde_top
        goodparams = [self.params[i] for i in np.argsort(self.values)[:num_v] ]
        self.tpemodels = [ self._fitkd([l[i] for l,_ in goodparams]) for i in range(len(self.space[0]))]
        #exp = [kd.bandwidth for kd in self.tpemodels]
        #for l,_ in goodparams: plt.Circle(l,exp[0])

    def _fitkd(self, values):
        # n**(-1./(d+4))   heuristic to determine bandwith; n= num_sampled, d = dimensions
        bandwidth = np.std(values)/5 #len(values)**(-1/5)
        print(f"{ bandwidth = }")
        kd = KernelDensity(kernel='gaussian', bandwidth = bandwidth)
        kd.fit(np.array(values).reshape(-1,1))
        return kd

    def TPEsample(self):
        return [kd.sample()[0,0] for kd in self.tpemodels],{}

    ################
    # BAYES model ..
    ##################
    def Bayesfit(self):
        assert len(self.params) > 1, 'do some random guessing first'
        self.baymodels = [ self._fitbayes(s,[(l[i],v) for s,v,(l,_) in zip(self.values,self.params)]) for i,s in enumerate(self.space[0])]

    def _fitbayes(self, space, xyL):
        X,y = Transpose(xyL)
        bayes().fit(space, X,y)

    def Bayessample(self):
        return [bay.sample() for kd in self.baymodels],{}




def myf(args):
    x,y = args[0]
    return abs(x)+abs(y)

if __name__ == "__main__":
    opti = TPE(myf,([(-100,100),(-100,100)],{}),n_init = 10)
    n_iter = 30
    n_jobs = 5
    import matplotlib
    matplotlib.use("module://matplotlib-sixel")
    import matplotlib.pyplot as plt
    while n_iter > 0:
        args = opti.suggest(n_jobs)
        values = mpmap(opti.f,args)
        opti.register(args, values)
        tmp = np.array([a for a,b in args])
        n_iter -= n_jobs
        plt.scatter(tmp[:,0], tmp[:,1])
        plt.show()
        plt.close()
        print(np.mean(values))
    #opti.minimize()
    print ( [p for p,d in opti.params] )
    print (opti.values)
