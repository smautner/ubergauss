import pandas as pd
from itertools import product
from ubergauss import tools as ut


def maketasks(dict_):
    return (dict_(zip(dict_.keys(), values)) for values in product(*dict_.values()))


def gridsearch(func, dict_,data, score = 'score', df = True):

    tasks = maketasks(dict_)

    func2 = lambda t: func(*data,**t)
    res = ut.xmap(func2, tasks)

    for t,r in zip(tasks, res):
        r[score] = r

    if df:
        return pd.DataFrame(tasks)
    return tasks
