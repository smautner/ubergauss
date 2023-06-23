import pandas as pd
from itertools import product
from ubergauss import tools as ut


def maketasks(dict_):
    return [dict(zip(dict_.keys(), values)) for values in product(*dict_.values())]


def gridsearch(func, dict_,data, score = 'score', df = True):

    tasks = maketasks(dict_)

    func2 = lambda t: func(*data,**t)
    res = ut.xmap(func2, tasks)

    for t,r in zip(tasks, res):
        t[score] = r
    if df:
        r= pd.DataFrame(tasks)
        r.dropna(thresh=2, axis=1)
        return r
    return tasks
