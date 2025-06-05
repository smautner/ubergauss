from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import pandas as pd
from itertools import product
from ubergauss import tools as ut
import numpy as np
import traceback
from sklearn.model_selection import  BaseCrossValidator



def maketasks(param_dict):
    return [dict(zip(param_dict.keys(), values)) for values in product(*param_dict.values())]


def getvalues(val):
    try:
        a, b, c = val.split()
        r=  np.linspace(float(a), float(b), int(c))
        return r
    except:
        return eval(val)

def string_to_param_dict(text):
    '''
    this is not a spaceship, its just a helper to make a paramdict from text
    '''
    result = {}
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        key, values = line.strip().split(maxsplit=1)
        result[key] = getvalues(values)
    return result

import time
def gridsearch(func,data_list = None, *, param_dict = False, tasks = False, taskfilter =None,
               score = 'score',mp = True,  df = True, param_string = False , timevar=f'time', **kwargs):
    '''
    ways to provide tasks:
        # data
        - data_list [tupple being passed as *,..]

        # tasks
        - tasks: ive me a task list of dictionaries
        - param_dict: a dict that defines valid options {paramname: [1,2,3]}
        - param_string: either valid options param:['option1','option2']
                                or linspace param: 1 1.5 11
        - you could also use hyperopt.spaceship(string).sample() to sample tasks


    '''

    ############
    # setting up tasks
    ##########
    assert sum( [type(x) == bool for x in [param_string, param_dict, tasks]] )  == 2, 'we expect 2 to be false'
    assert len(data_list) > 0
    if param_string:
        param_dict = string_to_param_dict(param_string)
    if param_dict:
        tasks = maketasks(param_dict)
    if taskfilter:
        tasks = list(filter(taskfilter ,tasks))

    def func2(t):
        start = time.time()
        try:
            t.update(kwargs)
            data = t.pop('datafield')
            res = func(*data_list[data],**t)
        except Exception as e:
            print(f"EXCEPTION:")
            traceback.print_exc()
            print(f"PARAMS:")
            print(t)
            print(f"EXCEPTION END")
            res = None
        return res, time.time()-start

    def mktask(d,e):
        e=e.copy()
        e['datafield'] = d
        return e

    tasks = [ mktask(d,e) for d in range(len(data_list)) for e in tasks ]

    if mp:
        res = ut.xxmap(func2, tasks)
    else:
        res = Map(func2, tasks)

    #  removes failed tasks
    t_r = filter(lambda r: r[1][0] is not None,zip(tasks,res))

    # this is updating the tasks
    for t,(r,sek) in t_r:
        if type(r) != dict:
            t[score] = r
        else:
            t.update(r)
        t[timevar] = sek

    if df:
        r= pd.DataFrame(tasks)
        # r.dropna(thresh=2, axis=1) <- this shouldnt work anyaway as inplace is false per dfefault
        # dropped_columns = df.columns.difference(df_cleaned.columns)

        return r
    return tasks

def df_remove_duplicates(grid_df, return_unique=False):
    unique = grid_df.nunique(dropna=False)
    grid_df = grid_df.loc[:,unique!=1]
    if return_unique:
        return grid_df, unique
    return grid_df

def dfprint(grid_df: pd.DataFrame, score:str ='score', showall:bool =True):

    if score in grid_df:
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


def get_best(df, column= 'score'):
    max_index = df[column].idxmax()
    max_row = df.loc[max_index].to_dict()
    print(max_row)


def test_grid_optimizer():
    from sklearn.datasets import make_classification
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score as ari

    data = make_classification()
    grid = {'n_clusters': [2,3,4]}
    def run(X,y,**params):
        return 0


    df = gridsearch(run, param_dict= grid, data = data)
    # print(df.corr(method=f'spearman'))
    dfprint(df)

    st = '''
    ert 0 1 11
    zomg ["1","2"]
    '''
    df = gridsearch(run,data, param_string=st)
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





def pareto_scores(df, score='score', data='dataset', scoretype = 'target', method = ['method']):
    '''
    so we expect a df in input,

    method identifies the data sources that we are comparing
    score is the score column
    scoretype we use to distinguish between the different attributes that we compare

    data is an index for the scores so we can match the scoretypes
    '''
    # collect Rs
    names, runs = split_dataframe(df,method)
    rids = Range(runs)

    # now we collect scores for the Rs
    fail  = np.zeros_like(rids)
    for i in rids:
        for j in rids:
            if i != j:
                fail[i] += dominated(runs[i],runs[j], score, scoretype, data)

    return Zip(names, fail)

def dominated(r1,r2, score, scoretype, data):
    # how often does r1 get dominated?

    # print(r1)
    # print(f"{score=}")
    # print(f"{scoretype=}")
    # print(f"{data=}")

    if data != '':
        r1 = r1.pivot_table(index=data, columns=scoretype, values=score).to_numpy()
        r2 = r2.pivot_table(index=data, columns=scoretype, values=score).to_numpy()
    else:
        r1 = r1.pivot_table(index=None, columns=scoretype, values=score).to_numpy()
        r2 = r2.pivot_table(index=None, columns=scoretype, values=score).to_numpy()
    count = 0
    for row in r1:
        count+= np.sum(np.all(r2 > row, axis = 1))
    return count

def split_dataframe(df, column_names):
    groupname_df = df.groupby(column_names)
    return Transpose(groupname_df)


# def split_dataframe(df, column_names):
#     grouped = df.groupby(column_names)
#     split_dataframes = []
#     for group_name, group_df in grouped:
#         split_dataframes.append(group_df.copy())
#     return split_dataframes

if __name__ == "__main__":
    test_grid_optimizer()
