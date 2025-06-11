from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from hyperopt.pyll.stochastic import sample as hypersample
from pprint import pprint
'''
genetic optimizer
'''
import random
from ubergauss.optimization import baseoptimizer

class nutype(baseoptimizer.base):

    def __init__(self, space, f, data, numsample = 16,hyperband=[], floatavg =0):
        super().__init__( space, f, data, numsample = numsample, hyperband = hyperband )
        self.floatavg = floatavg
        self.seen = set()
        self.keyorder = list(self.params[0].keys())

    # def hashconfig(self,p):
    #     return hash(tuple(p[k] for k in self.keyorder))
    # def register(self,p):
    #     for e in p:
    #         self.seen.add(self.hashconfig(e))


    def nuParams(self):
        select  = int(self.numsample*.5)
        pool, weights = new_pool_soft(self.runs, select, maxold=.4)
        # pool, weights = tournament_usual(self.runs, select, 10)
        # pool, weights = tournament(self.runs, select, 5)
        # pool, weights = elitist_pool(self.runs, select)
        # pool, weights = rando_pool(self.runs, select, self.space)
        # pool, weights = self.loose_pool()
        weights = weights / np.sum(weights)
        # recombine
        new_params= []
        while len(new_params) < self.numsample:
            # x,y = np.random.choice(np.arange(len(pool)), size=2, replace=False, p=weights)
            x,y = np.random.choice(np.arange(len(pool)), size=2, replace=False)
            candidate  =  combine(pool[x],pool[y], self.space)
            # candidate = combine_aiming(pool[x],pool[y], weights[x] > weights[y], self.space)
            # candidate =  combine_dependant(pool[x],pool[y],self.paramgroups, self.space)
            # candidate =  combine_classic(pool[x],pool[y], self.space)
            new_params.append(candidate)
            # if self.hashconfig(candidate) not in self.seen:
            #     self.register([candidate])
            #     new_params.append(candidate)
            # else:
            #     print(self.seen)
            #     print(self.hashconfig(candidate))
            #     print(pool[x])
            #     print(pool[y])
            #     print(candidate)

        self.params = self.mutate(new_params)

    def mutate(self, params):
        return [ self.mutate_params(p,1/(len(self.keyorder)+1)) for p in params]

    def mutate_params(self, p, proba):
        for k in list(p.keys()):
            if random.random() < proba:# + proba*isinstance(p[k], int): # double mutation rate for categoricals
                # Mutate by sampling a new value from the original search space
                p[k] = hypersample(self.space.hoSpace[k])
        return p

def df_to_params(dfdf):
    # scores -= sorted.iloc[-5].score
    print(dfdf)
    scores =  dfdf.score
    dfdf = dfdf.drop(columns=['time', 'score','config_id'])
    pool = dfdf.to_dict(orient='records')
    weights= np.argsort(np.array(scores)) + 3
    return pool,weights # scores.tolist()

def avg_noise(a,b,key,space):
    typ = space.space[key][0]
    a,b = a[key], b[key]
    new = np.mean([a,b])
    std = abs(a-b)*.7
    if typ == 'int':
        std = max(std, .5)
    new = np.random.normal(new, std)
    low, high = space.space[key][1][:2]
    new = max(new,low)
    new = min(high, new)
    if typ == 'int':
        new = int(new+.5)
    return new

def combine( a, b, space=None):
    new_params = {}
    for k in a.keys():
        val_a = a[k]
        val_b = b[k]
        typ = space.space[k][0]
        if typ == 'cat':
            new_params[k] = random.choice([val_b, val_a])
            continue
        new_params[k] = avg_noise(a,b,k,space)
    return new_params

def combine_classic(a, b, space=None):
    new_params = {}
    keys = list(a.keys())
    num_keys = len(keys)
    crossover_point = random.randint(0, num_keys)
    for i, key in enumerate(keys):
        if i < crossover_point:
            new_params[key] = a[key]
        else:
            new_params[key] = b[key]
    return new_params




def new_pool_soft(runs, numselect, maxold = .66):
    # hmm i dont even need to check the len runs :0 nice

    n_combo = int(numselect*maxold)
    combo = pd.concat(runs)
    combo = combo.sort_values(by='score', ascending=False).head(n_combo)

    final = pd.concat([combo, runs[-1]])
    final = final.sort_values(by='score', ascending=False)
    final = final.drop_duplicates().head(numselect)
    return df_to_params(final)



def elitist_pool(runs, numselect):
    # SELECT THE BEST
    dfdf = pd.concat(runs)
    # dfdf = runs[-1]
    dfdf = dfdf[dfdf.score > 0]
    sorted = dfdf.sort_values(by='score', ascending=False)
    dfdf = sorted.head(numselect)
    # scores -= sorted.iloc[-5].score
    return df_to_params(dfdf)



def tournament(runs, numselect, bestof = 8):

    dfdf = pd.concat(runs).sort_values(by='score', ascending=False).head(len(runs[-1])*2)
    dfdf = dfdf[dfdf.score > 0]

    items = Zip(dfdf.score, Range(dfdf))
    def select():
        random.shuffle(items)
        item = max(items[:8])
        items.remove(item)
        return item[1]
    indices = [select() for i in range(numselect)]
    dfdf = dfdf.iloc[indices]
    return df_to_params(dfdf)

def tournament_usual(runs, numselect, bestof = 8):
    dfdf = pd.concat(runs)
    dfdf = dfdf[dfdf.score > 0]
    items = Zip(dfdf.score, Range(dfdf))
    def select():
        item = max(random.sample(items, bestof))
        items.remove(item)
        return item[1]
    indices = [select() for i in range(numselect)]
    dfdf = dfdf.iloc[indices]
    return df_to_params(dfdf)


def combine_aiming( a, b, agb=False, space=None):
    new_params = {}
    for k in a.keys():
        val_a = a[k]
        val_b = b[k]
        typ = space.space[k][0]
        better, good = (val_a, val_b) if agb else (val_b,val_a)
        if typ == 'cat':
            new_params[k] = random.choice([better,good])
            continue
        #new =  (better+good)/2
        new = np.random.normal(better, abs(better-good))
        low, high = space.space[k][1][:2]
        new = max(new,low)
        new = min(high, new)
        if typ == 'int':
            new = int(new+.5)
        new_params[k] = new
    return new_params


def combine_dependant(a, b, paramgroups, space):
    new_params = a.copy()
    for  keys_in_group in paramgroups:
        if random.random() < 0.5: # 50% chance to inherit from 'b'
            for k in keys_in_group:
                if k in new_params and k in b:
                     new_params[k] = b[k]
        k=keys_in_group[0]
        if len(keys_in_group) == 1 and space.space[k][0] != 'cat':
            new_params[k] = avg_noise(a,b,k,space)
        elif len(keys_in_group) == 2 and a[k]==b[k]:
            k2 = keys_in_group[1]
            new_params[k2] = avg_noise(a,b,k2,space)
    return new_params
