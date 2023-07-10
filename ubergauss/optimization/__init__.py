import pandas as pd
from itertools import product
from ubergauss import tools as ut
import numpy as np
from sklearn.model_selection import  BaseCrossValidator



def maketasks(param_dict):
    return [dict(zip(param_dict.keys(), values)) for values in product(*param_dict.values())]


def gridsearch(func, param_dict, data, score = 'score', df = True):

    tasks = maketasks(param_dict)

    func2 = lambda t: func(*data,**t)
    res = ut.xmap(func2, tasks)

    for t,r in zip(tasks, res):
        t[score] = r
    if df:
        r= pd.DataFrame(tasks)
        r.dropna(thresh=2, axis=1)
        return r
    return tasks

def print(grid_df, score='score', showall=True):
    grid_df = grid_df.sort_values(by = score)

    if showall:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print (grid_df)
    else:
        print(grid_df)



def contains(iter, element):
    for e in iter:
        if e == element:
            return True
    return False

class groupedCV(BaseCrossValidator):
    def __init__(self, n_splits):
        self.n_splits = n_splits
    def get_n_splits(self, X= None, y= None, groups = None):
        return self.n_splits

    # def arin(self,groupindex, testgrps):
    #     return np.array([ contains(testgrps, a) for a in groupindex])

    def arin_index(self,groupindex, testgrps):
        return np.array([i for i,a in enumerate(groupindex) if contains(testgrps,a)])

    def _iter_test_indices(self, X,y,groups):
        groups = np.array(groups)
        z = np.unique(groups)
        np.random.shuffle(z)
        if self.n_splits > 1:
            for testgroups in np.array_split(z, self.n_splits):
                res =  self.arin_index(groups, testgroups)
                yield res
        else:
            test = np.array_split(z, 3)[0]
            yield self.arin_index(groups, test)
