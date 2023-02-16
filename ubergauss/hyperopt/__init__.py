
# tools for hyperopt


import pandas as pd
import numpy as np
from hyperopt.pyll import scope
from hyperopt import fmin, tpe, hp, Trials


def trial2df(trials):

    # get the loss
    res  = np.array([trial['result']['loss'] for trial in trials])

    # variables that we optimized
    keys = trials[0]['misc']['vals'].keys()
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


