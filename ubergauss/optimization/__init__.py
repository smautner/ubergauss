import pandas as pd
from itertools import product
from ubergauss import tools as ut
import numpy as np
from sklearn.model_selection import  BaseCrossValidator



def maketasks(param_dict):
    return [dict(zip(param_dict.keys(), values)) for values in product(*param_dict.values())]


import time
def gridsearch(func, param_dict, data, score = 'score', df = True,timevar=f'time'):

    tasks = maketasks(param_dict)

    def func2(t):
        start = time.time()
        try:
            res = func(*data,**t)
        except Exception as e:
            print(f"EXCEPTION:")
            print(e)
            print(f"PARAMS:")
            print(t)
            print(f"EXCEPTION END")
            exit()

        return res, time.time()-start

    res = ut.xmap(func2, tasks)
    # res = list(map(func2, tasks))

    for t,(r,sek) in zip(tasks, res):
        t[score] = r
        t[timevar] = sek
    if df:
        r= pd.DataFrame(tasks)
        r.dropna(thresh=2, axis=1)
        return r
    return tasks

def df_remove_duplicates(grid_df, return_unique=False):
    unique = grid_df.nunique(dropna=False)
    grid_df = grid_df.loc[:,unique!=1]
    if return_unique:
        return grid_df, unique
    return grid_df

def dfprint(grid_df: pd.DataFrame, score:str ='score', showall:bool =True):

    grid_df = grid_df.sort_values(by = score)
    grid_df, unique = df_remove_duplicates(grid_df, return_unique = True)

    if showall:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print (grid_df)
    else:
        print(grid_df)

    if sum(unique!=1) > 0:
        print(f'\n\nremoved column(s) that had only 1 value:')
        print(f'{unique[unique==1]}')


def test_grid_optimizer():
    from sklearn.datasets import make_classification
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score as ari

    data = make_classification()
    grid = {'n_clusters': [2,3,4]}
    def run(X,y,**params):
        clf = KMeans(**params)
        yh = clf.fit_predict(X)
        return ari(y,yh)
    df = gridsearch(run, grid, data)
    # print(df.corr(method=f'spearman'))
    dfprint(df)




def contains(iter, element):
    for e in iter:
        if e == element:
            return True
    return False

class groupedCV(BaseCrossValidator):
    def __init__(self, n_splits, randseed=None):
        self.n_splits = n_splits
        self.randomseed = randseed

    def get_n_splits(self, X= None, y= None, groups = None):
        return self.n_splits

    # def arin(self,groupindex, testgrps):
    #     return np.array([ contains(testgrps, a) for a in groupindex])

    def arin_index(self,groupindex, testgrps):
        return np.array([i for i,a in enumerate(groupindex) if contains(testgrps,a)])

    def _iter_test_indices(self, X,y,groups):
        groups = np.array(groups)
        z = np.unique(groups)
        np.random.seed(seed = self.randomseed)
        np.random.shuffle(z)
        if self.n_splits > 1:
            for testgroups in np.array_split(z, self.n_splits):
                res =  self.arin_index(groups, testgroups)
                yield res
        else:
            test = np.array_split(z, 3)[0]
            yield self.arin_index(groups, test)
