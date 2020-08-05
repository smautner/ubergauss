import numpy as np
import functools
from sklearn.mixture import GaussianMixture as gmm 
import multiprocessing as mp                                                
import math

###
# knee point detection 1: 
# choose value between gaussians
###

def gmm2(values):
    A=np.array(values).reshape(-1, 1)
    model = gmm(n_components=2, n_init=10).fit(A)
    return model.predict_proba(A), model

def only_between(cluster_probs,Y,gmm_means):
    '''removes outliers, i.e. sets 0 fields of z where a not between means'''
    mi,ma = gmm_means.min(),gmm_means.max()
    filtr = [ aa < mi or aa > ma for aa in Y ]
    cluster_probs[filtr]=0 
    return cluster_probs

def between_gaussians(values):
    cluster_probs, model = gmm2(values)
    cprobs = cluster_probs.sum(axis=1)
    cprobs = only_between(cprobs,values, model.means_)
    return np.argmax(cprobs)



####
#  max distance to line 
#  kneed implementaiton is annoying, 
######

def diag_maxdist(values): 
    points = [ (x,y) for x,y in enumerate(values)  ]
    x1,y1 = points[0]
    x2,y2 = points[-1]
    def dist(p):
        x0,y0 = p
        return  ( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 -y2*x1) / math.sqrt(  (y2-y1)**2 + (x2-x1)**2 ) 


    res  =list(map(dist,points))
    if not (all ( map( lambda x: x>=0, res)) or all(map(lambda x: x<=0 ,res))):
        print ("ubergauss.diag_maxdist encountered strange data: dist to diagonal", res)
    return np.argmax(np.abs(res))






###
# multiprocessing
###
def mpmap(func, iterable, chunksize=1, poolsize=2):                            
    pool = mp.Pool(poolsize)                                                    
    result = pool.map(func, iterable, chunksize=chunksize)                      
    pool.close()                                                                
    pool.join()                                                                 
    return list(result) 


def traingmm(n_comp, X=None,n_init=10,**kwargs):
   return gmm(n_init = n_init,
     n_components=n_comp, **kwargs).fit(X) # covarianve_type full is default


####
# get a model
####

def get_model(X, poolsize = -1, nclust_min = 4, nclust_max = 20, n_init = 20,covariance_type = 'tied',**kwargs):

    # trivial case:
    if nclust_min == nclust_max: 
        return traingmm(nclust_min,X=X,n_init=n_init,**kwargs)

    # train models
    train = functools.partial(traingmm,X=X,n_init=n_init,**kwargs)
    models = mpmap( train , range(nclust_min,nclust_max), poolsize= poolsize)

    # kneepoint
    bics = [m.bic(X) for m in models]
    best = diag_maxdist(bics)
    return models[best]




'''
####
# OLD MAN, using KNEED to find kneepoints... 
####
def  get_model(X, poolsize = 4, nclust_min = 4, nclust_max = 20, n_init = 20,**kwargs):
    import kneed
    # trivial case:
    if nclust_min == nclust_max: 
        return traingmm(nclust_min,X=X,n_init=n_init,**kwargs)

    # train models
    train = functools.partial(traingmm,X=X,n_init=n_init,**kwargs)
    models = mpmap( train , range(nclust_min,nclust_max), poolsize= poolsize)

    # kneepoint
    bics = [-m.bic(X) for m in models]
    print ("bics:", bics)
    #best = last_of_variate_gaussian(bics)
    kneedler = kneed.KneeLocator(list(range(len(bics))),bics, S=1.0, curve='concave', direction='increasing')
    best = kneedler.knee
    if not isinstance(best,int):
        best = 3
    return models[best]
'''

'''
#####
# knee point detection 2:
# choose last of the distribution that is more varying
########
def last_of_variate_gaussian(bics):
    predictions, model = gmm2(bics)
    zz = predictions[:,np.argmax(model.covariances_)]
    for i,v in enumerate(zz):
        if v < .95: 
            return i-1
    return -1
'''


