# tools for hyperopt
import pandas as pd
import numpy as np
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, Trials


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


class spaceship():
    def __init__(self,name_range):
        name_range = [ x.split() for x in name_range]
        name_range = [[l[0],l[1:]] for l in name_range]
        self.space = {}
        self.nr = dict(name_range)
        for name, range in name_range:
            self.space[name] = scope.int(hp.quniform(name,*map(int, range))) if len(range) == 3  else hp.uniform(name,*map(float,range))

    def translate(self,best):
        def lol(k,v):
            return k, int(v) if len(self.nr[k]) == 2 else v
        return dict(map(lol,best.items()))


from hyperopt import  trials_from_docs


def run(x , f= None, trials = None, space = None, max_evals = None):
    if trials = None:
        trials = Trials()
    fn=lambda y: f(x=x, **y),
    best = fmin(fn,
                algo=tpe.suggest,
                trials = trials,
                space = space,
                max_evals=1)
    return trials

def fffmin(fun, items=[0], probing_parallel = 2, probing_trials = 10, after_evals = 10, space= None):
    eva = lambda x: run( x, f = fun,space=space, max_evals = probing_trials)
    trialslist = ug.xmap(eva, items*probing_parallel)
    merged_trials = [trials_from_docs(trialslist[i::len(items)]) for i in items]
    eva = lambda x: run(x[0],trials = x[1], f = fun,space=space, max_evals = after_evals)
    trialslist = ug.xmap(eva, zip(items, merged_trials))


