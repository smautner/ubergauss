

import pandas as pd
from itertools import product
from ubergauss import tools as ut


def maketasks(dict):
    return (dict(zip(dict.keys(), values)) for values in product(*dict.values()))


def gridsearch(func, dict, score = 'score', df = True):

    tasks = maketasks(dict)
    res = ut.xmap(func, tasks)

    for t,r in zip(tasks, res):
        r[score] = r

    if df:
        return pd.DataFrame(tasks)
    return tasks
