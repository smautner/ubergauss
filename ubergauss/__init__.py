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


def traingmm(n_comp, X=None,n_init=10):
   return gmm(n_init = n_init,
     n_components=n_comp, covariance_type='full').fit(X)


def only_between(z,a,means):
    '''removes outliers, i.e. sets 0 fields of z where a not between means'''
    mi,ma = means.min(),means.max()
    filter = [ aa < mi or aa > ma for aa in a ]
    z[filter,0]=0 # one column is enough because we use the min later
    return z

####
# MAIN 
####
def  get_model(X, poolsize = 4, nclust_min = 4, nclust_max = 20, n_init = 20):

    # train models
    train = functools.partial(traingmm,X=X,n_init=n_init)
    models = mpmap( train , range(nclust_min,nclust_max), poolsize= poolsize)

    # get bic scores 
    bics = [m.bic(X) for m in models]
    print("bics:",bics)



    # select best 
    A=np.array(bics).reshape(-1, 1)
    model = gmm(n_components=2).fit(A)
    cluster_probs=model.predict_proba(A)
    cluster_probs = only_between( cluster_probs ,bics, model.means_)
    best = np.argmax(cluster_probs.min(axis=1))

    # return best
    return models[best]


