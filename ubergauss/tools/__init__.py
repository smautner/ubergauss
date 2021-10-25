from multiprocessing import Pool
_func = None
def worker_init(func):
  global _func
  _func = func
def worker(x):
  return _func(x)
def xmap(func, iterable, processes=None):
  with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
    return p.map(worker, iterable)

import numpy as np
from scipy.stats import spearmanr
from scipy import sparse

def spearman(x,y):
    spear = lambda ft: np.abs(spearmanr(ft.T,y)[0])
    x = zehidense(x)
    re = list(map(spear,x.T))
    return np.array(re)

def zehidense(X):
    if sparse.issparse(X):
        return X.todense()
    return X
