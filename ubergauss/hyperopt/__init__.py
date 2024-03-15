# tools for hyperopt
import pandas as pd
import numpy as np
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, Trials
import ubergauss.tools as ug
import hyperopt.rand as hyrand

def trial2df(trials):

    # get the loss
    res  = np.array([trial['result']['loss'] for trial in trials])

    # variables that we optimized
    keys = trials._trials[0]['misc']['vals'].keys()
    def get_param_list(d):
        return [d[kk][0] for kk in keys]


    # get each line, then traospose
    vals = np.array([ get_param_list(trial['misc']['vals']) for trial in trials]).T

    data = {k:v for k,v in zip(keys,vals)}
    data['loss'] = res
    return pd.DataFrame(data)

from hyperopt.pyll.stochastic import sample

class spaceship():
    def __init__(self,name_range):
        name_range = name_range.split('\n')
        name_range = [x.strip() for x in name_range]
        name_range = [ x.split() for x in name_range if x]
        name_range = [[l[0],l[1:]] for l in name_range]
        self.space = {}
        self.nr = dict(name_range)
        for name, value_range in name_range:
            if '[' in value_range[0]:
                self.space[name] = hp.choice('bla',eval(''.join(value_range)))
            else:
                self.space[name] = scope.int(hp.quniform(name,*map(int, value_range))) if len(value_range) == 3  else hp.uniform(name,*map(float,value_range))

    def translate(self,best):
        def lol(k,v):
            return k, int(v) if len(self.nr[k],[]) == 2 else v
        return dict(map(lol,best.items()))

    def sample(self):
        return sample(self.space)

if __name__ == f"__main__":
    s= '''
    Z 0 1
    A [False, True]
    '''
    print(spaceship(s).sample())

from hyperopt import  trials_from_docs


def run(x , f= None, trials = None, space = None, max_evals = None, algo = tpe.suggest):
    if trials == None:
        trials = Trials()

    fn=lambda y: f(x=x, **y)

    best = fmin(fn,
                algo=algo,
                trials = trials,
                space = space,
                max_evals=max_evals)
    return trials

def concattrials(trials):
    trialdicts = [d  for a in trials for d in a._trials]
    return trials_from_docs(trialdicts)

def fffmin(fun, items=[0], probing_parallel = 2, probing_evals = 10, after_evals = 10, space= None):
    eva = lambda x: run( x, f = fun,space=space, max_evals = probing_evals)
    trialslist = ug.xmap(eva, items*probing_parallel)
    print(f"first round fin")
    print(trialslist)
    merged_trials = [concattrials(trialslist[i::len(items)]) for i in items]
    eva = lambda x: run(x[0],trials = x[1], f = fun,space=space, max_evals = probing_parallel*probing_evals+ after_evals)
    print(f"start second round")
    trialslist = ug.xmap(eva, zip(items, merged_trials))
    return trialslist


def fffmin2(fun,fun2, probing_parallel = 2, probing_evals = 10, after_evals = 10, space= None):
    # ffmin2 optimizes many targets at open
    # here we just have one target... so we parallelize in the beginnig to have many seeds...
    # then we merge and optimize...  :)
    eva = lambda x: run( None, f = fun,space=space, max_evals = probing_evals, algo = hyrand.suggest)

    if not probing_evals:
            merged_trials = Trials()
    else:
        trialslist = ug.xmap(eva, range(probing_parallel))
        print(f"first round fin")
        merged_trials = concattrials(trialslist)

    print(f"start second round")
    fmin(fun2,
          algo=tpe.suggest,
                trials = merged_trials,
                space = space,
                max_evals=probing_parallel*probing_evals+after_evals)
    return merged_trials
