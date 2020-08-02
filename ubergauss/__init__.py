import numpy as np
import functools
from sklearn.mixture import GaussianMixture as gmm 


###
# multiprocessing
###
import multiprocessing as mp                                                
def mpmap(func, iterable, chunksize=1, poolsize=2):                            
    pool = mp.Pool(poolsize)                                                    
    result = pool.map(func, iterable, chunksize=chunksize)                      
    pool.close()                                                                
    pool.join()                                                                 
    return list(result) 


def traingmm(n_comp, X=None,n_init=10,**kwargs):
   return gmm(n_init = n_init,
     n_components=n_comp, **kwargs).fit(X) # covarianve_type full is default



###
# knee point detection 1: 
# choose value between gaussians 
###
def only_between(z,a,means):
    '''removes outliers, i.e. sets 0 fields of z where a not between means'''
    mi,ma = means.min(),means.max()
    filter = [ aa < mi or aa > ma for aa in a ]
    z[filter,0]=0 # one column is enough because we use the min later
    return z

def value_between_gaussians(bics):
    cluster_probs, model = kneemixedmodel(bics)
    cluster_probs = only_between( cluster_probs ,bics, model.means_)
    return np.argmax(cluster_probs.min(axis=1))

#####
# knee point detection 2:
# choose last of the distribution that is more varying
########
def last_of_variate_gaussian(bics):
    predictions, model = kneemixedmodel(bics)
    zz = predictions[:,np.argmax(model.covariances_)]
    for i,v in enumerate(zz):
        if v < .95: 
            return i-1
    return -1



####
# MAIN 
####
def kneemixedmodel(bics):
    A=np.array(bics).reshape(-1, 1)
    model = gmm(n_components=2).fit(A)
    return model.predict_proba(A), model

import kneed
def  get_model(X, poolsize = 4, nclust_min = 4, nclust_max = 20, n_init = 20,**kwargs):
        
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


