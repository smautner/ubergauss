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
import dill
import json

def spearman(x,y):
    spear = lambda ft: np.abs(spearmanr(ft.T,y)[0])
    x = zehidense(x)
    re = list(map(spear,x.T))
    return np.array(re)

def zehidense(X):
    if sparse.issparse(X):
        return X.todense()
    return X

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)



def np_bool_select(numpi, bools):
    return np.array([x for x,y in zip(numpi,bools) if y  ])

dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "wb"))
loadfile = lambda filename: dill.load(open(filename, "rb"))

jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
jloadfile = lambda filename:  json.loads(open(filename,'r').read())



def maxnum(X):
    return np.nanmax( np.where(np.isinf(X),-np.Inf,X) )
def minnum(X):
    return np.nanmin( np.where(np.isinf(X),np.Inf,X) )
