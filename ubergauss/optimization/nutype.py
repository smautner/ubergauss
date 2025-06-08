from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from functools import partial
import pandas as pd
import numpy as np
from hyperopt.pyll.stochastic import sample as hypersample
from ubergauss.optimization import baseoptimizer
'''
for ints -> build a model cumsum(occcurance good / occurance bad) ,  then sample accoridngly
for floats -> gaussian sample from the top 50% of the scores
'''



class nutype(baseoptimizer.base):

    def nuParams(self):
        '''
        first the non dependants can be done as usual,
        then for the dependants, they need subsamplers , so a sampler returns a dict v:k... in the end i can combine all the dicts...
        '''
        if not hasattr(self, 'samplers'):
            self.samplers = [Sampler(self.space,p) for p in self.paramgroups]
        if not hasattr(self, 'carry'):
            self.carry = pd.DataFrame()
        # data = pd.concat((self.carry,self.df))
        data = pd.concat((self.carry,self.df))
        for s in self.samplers:
            do, self.key_log[s.name] = check_col(self, s.name)
            s.learn(data)

        self.params = [self.sample() for _ in range(self.numsample)]

        c = int(self.numsample*.4)
        data = data.sort_values(by='score', ascending=False)
        self.carry = data.head(c).copy()


    def sample(self):
        d={}
        for s in self.samplers:
            d.update(s.sample())
        return d




def Sampler(space, keys):
    if len(keys) ==1:
        return  Simple(space, keys[0])
    else:
        return  Samplerr(space, keys)

class Samplerr():
    def __init__(self, space, keys):
        self.name = keys[0]
        self.mainsampler = Simple(space,keys[0])
        self.sub = {cat:Simple(space,keys[1]) for cat in space.space[keys[0]][1] }
        # orig = partial(hypersample, spaceship.hoSpace[k])

    def learn(self, df):
        self.mainsampler.learn(df)
        for e in self.sub:
            df2 = df[df[self.name] == e]
            if len(df2)>2:
                self.sub[e].learn(df2)

    def sample(self):
        r = self.mainsampler.sample()

        r.update(self.sub[list(r.values())[0]].sample())
        return r

class Simple():
    def __init__(self, space,key):
        self.name=key
        self.par = space.space[key]
        self.sample_f = partial(hypersample, space.hoSpace[key])

    def sample(self):
        return {self.name:self.sample_f()}

    def learn(self,df):
        self.sample_f = mksampler(self.name,df,self.par)

def mksampler(name,df,par):
    if par[0] == 'cat':
        return learn_cat_sampler(df.score,df[name])

    df = df.tail(6)
    if par[0] == 'float':
        return learn_float_sampler(df.score,df[name])

    if par[0] == 'int':
        return learn_int_sampler(df.score,df[name],par[1])

def learn_int_sampler(scores, values,par):
    float_sampler = learn_float_sampler(scores,values)
    def int_sampler():
        while True:
            v = int(float_sampler()+.5)
            if par[0]<= v <= par[1]:
                return v
    return int_sampler


def learn_cat_sampler(scores, values):
    # take top 40% and bottom 40%
    scores = np.array(scores)
    values = np.array(values)
    sorted_indices = np.argsort(scores)[::-1]
    top_40 = sorted_indices[:int(len(scores) * 0.4)]
    bottom_40 = sorted_indices[int(len(scores) * 0.6):]

    # calc probability for each integer:
    # allints = unique(top40)
    # freqscore = [ score(int) for allints]
    # score is the occurance in top40 / occ in bottom +1
    allints = np.unique(values[top_40])
    def getscore(i):
        top_count = np.sum(values[top_40] == i)
        bottom_count = np.sum(values[bottom_40] == i) + 1
        scr =  top_count / bottom_count
        return scr
    scores = np.array([getscore(i) for i in allints])
    # print(dict(zip(allints, scores)))

    # now we can make a cumsum of the scores, scale up a random.random and choose one of the scores

    cum_scores = np.cumsum(scores)
    total_score = cum_scores[-1]
    def sample():
        r = np.random.uniform(0, total_score)
        chosen_index = np.searchsorted(cum_scores, r)
        return allints[chosen_index]
    return sample

def need_sampler(scores,values):
    # sort score and values by scores
    # throw away the middle 20%
    # then label topscores 1, bottomscores 0  -> y
    #return sum(y == np.roll(y,-1))-2/len(y)
    sv = Zip(scores,values)
    scoresort = sorted(sv, key = lambda x:x[0])
    p40 = int(len(sv)*.4)
    score1 = [ (0,v) for s,v in scoresort[:p40] ]
    score1+= [ (1,v) for s,v in scoresort[-p40:]]
    valsort = sorted(score1, key = lambda x:x[1])
    y = np.array([ii for ii,_ in valsort])
    score = (np.sum(y == np.roll(y, -1)) -2) / (len(y)-2) # 2 misses are allowed :)
    # 0 -> all the same  -> i shoudl resample
    # 1 -> all different -> dont resample
    return score

def learn_float_sampler_old(scores,values):
        scores = np.array(scores)
        values = np.array(values)

        sorted_indices = np.argsort(scores)[::-1]
        topat = int(len(scores) * 0.4)
        top_half = sorted_indices[:topat]
        top_scores = scores[top_half]
        top_values = values[top_half]

        min_score = top_scores.min()
        max_score = top_scores.max()
        if max_score == min_score:
            scaled_scores = np.full_like(top_scores, 100.0)
        else:
            scaled_scores = 100 * (top_scores - min_score) / (max_score - min_score)
        flattened = [v for s, v in zip(scaled_scores, top_values) for _ in range(int(s))]

        # flattened = top_values
        m,s = np.mean(flattened),np.std(flattened)
        # print(f"{m=} {s=} {values=}")
        samples = lambda: np.random.normal(m,s)
        # print mean and std
        return samples


def learn_float_sampler(scores,values):

        scores = np.array(scores)
        values = np.array(values)
        weights = np.argsort(scores)

        flattened = [v for s, v in zip(weights, values) for _ in range(int(s))]

        m,s = np.mean(flattened),np.std(flattened)
        # print(f"{values,m,s=}")

        # print(f"{m=} {s=} {values=}")
        samples = lambda: np.random.normal(m,s)
        # print mean and std
        return samples





def check_col(optimizer, key):
    # for the key, we either cal shouldlearn
    # for floats and ints log will just be the normalized value,
    # for cat maybe the proba for each category


    values = optimizer.df[key]
    scores = optimizer.df.score
    param_type = optimizer.space.space[key][0]

    should = True
    log = None

    if param_type in ['float', 'int']:
        values_reshaped = values.values.reshape(-1, 1)
        log = should_learn_float(values_reshaped, scores.values)
    elif param_type == 'cat':
        log = should_learn_cat(values, scores)

    return should, log

from sklearn.neighbors import NearestNeighbors

def should_learn_float(x, y, n_neighbors=1):
    """
    Predicts y using average of y-values of nearest neighbors in x,
    and returns the normalized prediction error.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Fit Nearest Neighbors model (excluding the point itself)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto')
    nn.fit(x)
    distances, indices = nn.kneighbors(x)
    pred_y = np.array([
        np.mean(y[indices[i][1:]]) for i in range(len(y))
    ])

    # Compute Mean Squared Error
    mse = np.mean((y - pred_y) ** 2)

    # Expected squared difference between any two y values (normalizer)
    diffs = y[:, None] - y[None, :]
    expected_squared_diff = np.mean(diffs ** 2)

    # Normalized error
    normalized_error = mse / expected_squared_diff if expected_squared_diff != 0 else np.nan

    return normalized_error

from scipy.stats import binom

def should_learn_cat(x, y, percentile=50, direction='high'):
    """
    Evaluate how each category in x performs in predicting high/low values of y.

    Parameters:
    - x: array-like (categorical)
    - y: array-like (numeric target)
    - percentile: int (e.g., 90 means top 10% of y are 'successes')
    - direction: 'low' or 'high' — detect underperformance or overperformance

    Returns:
    - DataFrame with columns: [category, count, successes, expected_successes, p_value]
    """

    # Define success threshold
    threshold = np.percentile(y, percentile)
    if direction == 'high':
        success = y >= threshold
    else:  # 'low'
        success = y <= threshold

    df = pd.DataFrame({'x': x, 'success': success})

    # Group by category
    grouped = df.groupby('x')['success'].agg(['count', 'sum']).reset_index()
    grouped.columns = ['category', 'n', 'k']
    print(f"{ grouped=}")
    grouped['p'] = (success.sum() / len(success))  # empirical success probability
    grouped['expected'] = grouped['n'] * grouped['p']

    # Compute binomial p-value (one-sided)
    def compute_pval(row):
        if direction == 'high':
            # Is k unusually high?
            return binom.sf(row.k - 1, row.n, row.p)
        else:
            # Is k unusually low?
            return binom.cdf(row.k, row.n, row.p)

    grouped['p_value'] = grouped.apply(compute_pval, axis=1)

    grouped.pop('p')
    return grouped.sort_values('p_value')

