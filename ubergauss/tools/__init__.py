from lmz import Map,Zip,Filter,Grouper,Range,Transpose
import os

from multiprocessing import Pool, current_process
_func = None
def worker_init(func):
  global _func
  _func = func
def worker(x):
  return _func(x)

from functools import partial

import tqdm

# def xmap(func, iterable, n_jobs=None, tasksperchild = 1, **kwargs):
#   func = partial(func, **kwargs)
#   with Pool(n_jobs, initializer=worker_init, initargs=(func,),maxtasksperchild = tasksperchild) as p:
#     return p.map(worker, iterable)

def xmap(func, iterable, n_jobs=None, tasksperchild = 1, **kwargs):
    func = partial(func, **kwargs)
    result_list_tqdm = []
    with Pool(n_jobs, initializer=worker_init, initargs=(func,),maxtasksperchild = tasksperchild) as p:
        for result in tqdm.tqdm(p.imap(worker, iterable), total=len(iterable)):
            result_list_tqdm.append(result)
    return result_list_tqdm


def xxmap(func, iterable, n_jobs=None, tasksperchild = 1, **kwargs):
    '''if in a subprocess we do sequencial map else do distributed map'''

    if current_process().name == 'MainProcess' and n_jobs != 1:
        return xmap(func, iterable, n_jobs=n_jobs, tasksperchild = tasksperchild, **kwargs)

    return Map(func,iterable,**kwargs)


def test_xmap():
    def f(x,y=0):
        return x+y
    assert xmap(f,[1,1,2], y= 3) == [4,4,5]
    def f(x):
        return x+3
    assert xmap(f,[1,1,2]) == [4,4,5]

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
        X =  X.todense()
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


def fixpath(path):
    if path[0]  == f'~':
        return  os.environ[f'HOME'] + path[1:]
    return path

dumpfile = lambda thing, filename: dill.dump(thing, open(fixpath(filename), "wb"))
loadfile = lambda filename: dill.load(open(fixpath(filename), "rb"))

jdumpfile = lambda thing, filename:  open(fixpath(filename),'w').write(json.dumps(thing))
jloadfile = lambda filename:  json.loads(open(fixpath(filename)  ,'r').read())


def ndumpfile(thing,filename):
    filename = fixpath(filename)
    if type(thing) == list:
        d= { chr(i+98):e  for i,e in enumerate(thing)}
        d['a'] = len(thing)
        np.savez_compressed(filename, **d)
    else:
        np.savez_compressed(filename,a=0,b=thing)

def nloadfile(filename):
    filename = fixpath(filename)
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


sdumpfile = lambda thing, filename:  sparse.save_npz(fixpath(filename), thing)
sloadfile = lambda filename:  sparse.load_npz(fixpath(filename)+'.npz')



class spacemap():
    # mapping items to integers...
    def __init__(self, uniqueitems):
        self.itemlist = uniqueitems
        self.integerlist = list(range(len(uniqueitems)))
        self.len = len(uniqueitems)
        self.getitem = {i:k for i,k in enumerate(uniqueitems)}
        self.getint = {k:i for i,k in enumerate(uniqueitems)}

    def encode(self, stuff, fallback = False):
        return [self.getint.get(x, fallback or x) for x in stuff]


    def decode(self, stuff,fallback = False):
        return [self.getitem.get(x,fallback or -1) for x in stuff]


def labelsToIntList(items):
    sm = spacemap(np.unique(items))
    return [sm.getint[i] for i in items], sm

def binarize(X,posratio):
    '''
    lowest posratio -> 1 ;; rest 0
    '''
    argsrt = np.argsort(X)
    if 0< posratio < 1:
        cut = max(int(len(X)*(1-posratio)),1)
    elif len(X) > posratio and isinstance(posratio,int):
        cut = len(X) - posratio
    else:
        assert False, f'{0<posratio<1= }  {len(X)>posratio= } {posratio=} {len(X)=}'

    values = np.ones(len(X), dtype = np.int32)
    values[argsrt[:cut]] = 0
    return values



if __name__ == "__main__":
    a = np.array([0,1,2])
    ndumpfile(a,'adump')
    ndumpfile([a,a],'aadump')
    print(nloadfile('adump'))
    print(nloadfile('aadump'))
    sdumpfile(sparse.csr_matrix(a),'sdump')
    print(sloadfile('sdump'))

    a = np.random.rand(6)
    print(a)
    print(binarize(a,2))





def stack_arrays(arrays, axis=0):
    """
    Stack numpy arrays and sparse scipy arrays horizontally and vertically.

    Args:
        arrays: list of numpy arrays and/or sparse scipy arrays
            The input arrays to stack.
        axis: int, optional (default=0)
            The axis along which to stack the arrays. If 0, the arrays are stacked vertically.
            If 1, the arrays are stacked horizontally.

    Returns:
        numpy array or sparse scipy array
            The stacked array.
    """
    assert axis in [0, 1],"Invalid axis value. Must be 0 or 1."
    assert isinstance(arrays, list), "Input must be a list of arrays."
    assert len(arrays) > 0, "Input list must not be empty."

    # Determine the data type of the input arrays
    if  sparse.issparse(arrays[0]):
        # If all arrays are sparse, use scipy.sparse.vstack or scipy.sparse.hstack
        if axis == 0:
            return sparse.vstack(arrays)
        else:
            return sparse.hstack(arrays)
    else:
        # If there is at least one numpy array, use numpy.vstack or numpy.hstack
        if axis == 0:
            return np.vstack(arrays)
        else:
            return np.hstack(arrays)

def hstack(arrays):
    return stack_arrays(arrays, axis = 1)
def vstack(arrays):
    return stack_arrays(arrays, axis = 0)



def cache(fname, makedata):
    if os.path.isfile(fname):
        return loadfile(fname)
    else:
        data = makedata()
        dumpfile(data,fname)
        return data
