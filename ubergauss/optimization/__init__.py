import pandas as pd
from itertools import product
from ubergauss import tools as ut


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
