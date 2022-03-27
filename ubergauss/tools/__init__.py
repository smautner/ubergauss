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
    if type(X) == np.matrix:
        X = X.A
    return X

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)



def np_bool_select(numpi, bools):
    return np.array([x for x,y in zip(zehidense(numpi),bools) if y  ])



def maxnum(X):
    return np.nanmax( np.where(np.isinf(X),-np.Inf,X) )

def minnum(X):
    return np.nanmin( np.where(np.isinf(X),np.Inf,X) )



dumpfile = lambda thing, filename: dill.dump(thing, open(filename, "wb"))
loadfile = lambda filename: dill.load(open(filename, "rb"))

jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
jloadfile = lambda filename:  json.loads(open(filename,'r').read())


def ndumpfile(thing,filename):
    if type(thing) == list:
        d= { chr(i+98):e  for i,e in enumerate(thing)}
        d['a'] = len(thing)
        np.savez_compressed(filename, **d)
    else:
        np.savez_compressed(filename,a=0,b=thing)

def nloadfile(filename):
    if filename[-4]!= '.':
        print('adding .npz to filename')
        filename += '.npz'

    if filename.endswith('.npy'):
        return np.load(filename)

    # load .npz
    z = np.load(filename, allow_pickle=True)
    num = int(z['a'])
    if num == 0:
        return z['b']
    else:
        return [z[chr(i+98)] for i in range(num) ]


sdumpfile = lambda thing, filename:  sparse.save_npz(filename, thing)
sloadfile = lambda filename:  sparse.load_npz(filename+'.npz')


if __name__ == "__main__":
    a = np.array([0,1,2])
    ndumpfile(a,'adump')
    ndumpfile([a,a],'aadump')
    print(nloadfile('adump'))
    print(nloadfile('aadump'))
    sdumpfile(sparse.csr_matrix(a),'sdump')
    print(sloadfile('sdump'))



def binarize(X,posratio):
    '''
    lowest posratio -> 1 ;; rest 0
    '''
    argsrt = np.argsort(X)
    cut = max(int(len(X)*posratio),1)
    values = np.zeros(len(X))
    values[argsrt[:cut]] = 1
    return values



class spacemap():
    # mapping items to integers...
    def __init__(self, items):
        self.itemlist = items
        self.integerlist = list(range(len(items)))
        self.len = len(items)
        self.getitem = { i:k for i,k in enumerate(items)}
        self.getint = { k:i for i,k in enumerate(items)}
