from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from functools import partial
import pandas as pd
import numpy as np
from hyperopt.pyll.stochastic import sample as hypersample
from ubergauss.optimization import baseoptimizer
from sklearn.neighbors import NearestNeighbors
from scipy.stats import binom
from sklearn import feature_selection
import random
'''
basically TPE style optimization per parameter vs score.
distributions are updated after a test is passed:
    - i remove categoricals base on p-values
    - mutual info threshold for continuous variables
'''



class nutype(baseoptimizer.base):

    def nuParams(self):
        '''
        first the non dependants can be done as usual,
        then for the dependants, they need subsamplers , so a sampler returns a dict v:k... in the end i can combine all the dicts...
        '''

        if not hasattr(self, 'samplers'):
            self.samplers = [mks(self.space,p[0]) for p in self.paramgroups]

        # if not hasattr(self, 'carry'):
        #     self.carry = pd.DataFrame()
        # data = pd.concat((self.carry,self.df))
        # data = pd.concat((self.carry,self.df))
        # data = data.sort_values(by='score', ascending=False)

        print(self.df[:8])
        for s in self.samplers:
            self.key_log[s.name] = s.update(self.runs)
            print( self.key_log[s.name])
        self.params = [self.sample() for _ in range(self.numsample)]

        # c = int(self.numsample*.4)
        # self.carry = data.head(c).copy()


    def sample(self):
        d={}
        for s in self.samplers:
            d.update(s.sample())
        return d


def mks(space,key):
    if space.space[key][0]=='cat':
        return CS(space,key)
    if space.space[key][0]=='float':
        return FS(space,key)
    if space.space[key][0]=='int':
        return IS(space,key)
class Simple():
    def __init__(self, space,key):
        self.name=key
        self.par = space.space[key]
        self.sample_f = partial(hypersample, space.hoSpace[key])
    def sample(self):
        return {self.name:self.sample_f()}





class CS(Simple):
    def update(self,runs):
        df = runs[-1]
        comment = p_values(df[self.name], df.score)
        self.sample_f = partial(random.choice, [k for k,v in comment.items() if v < .85])
        return comment

def p_values(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    success = y >= np.median(y)
    overall_p = .5

    # Compute p-value for each category
    results = {}
    for category in np.unique(x):
        category_success = success[x == category]
        n = len(category_success)
        k = np.sum(category_success)
        p_value = binom.sf(k, n, overall_p)
        results[category] = p_value
    return results



class FS(Simple):
    def update(self,runs):
        df = runs[-1]
        mutual_info = should_learn_float(df[self.name], df.score)
        if mutual_info > .2:
            top = .5
            if mutual_info > .3: top -= .1
            if mutual_info > .4: top -= .1
            if mutual_info > .5: top -= .1
            self.sample_f = learn_float_sampler(df.score,df[self.name], learntop = top)
        return mutual_info

def should_learn_float(values, score):
    scores = score.to_numpy()
    values = values.values.reshape(-1, 1)
    log = feature_selection.mutual_info_regression(values, scores, n_neighbors=2, discrete_features=False)
    return log[0]

def learn_float_sampler(scores,values,learntop=.4):
        scores = np.array(scores)
        values = np.array(values)
        weights = np.argsort(scores)
        vals = [values[i] for i in weights[int(len(scores)*learntop):]]

        # return partial(np.random.uniform,np.min(vals)*.9,np.max(vals)*1.1)
        return partial(np.random.normal,np.mean(vals),np.std(vals))
        # return partial(np.random.normal,np.mean(vals),np.std(vals))
        # flattened = [v for s, v in zip(weights, values) for _ in range(int(s))]
        # m,s = np.mean(flattened),np.std(flattened)*.5

class IS(Simple):
    def update(self,runs):
        '''
        1. check if we need to update the sampler.
        2. this will generate a comment that we might want to return
        3. then we update the sampler if necessary
        '''
        df = runs[-1]
        mutInfo = should_learn_float(df[self.name], df.score)
        if mutInfo > .2:
            sample_f = learn_float_sampler(df.score,df[self.name])

            def int_sampler():
                while True:
                    v = int(sample_f()+.5)
                    par = self.par[1]
                    if par[0]<= v <= par[1]:
                        return v
            self.sample_f = int_sampler
        return mutInfo
























def learn_cat_sampler(scores, values):
    '''
    this actually works nicely. looks at the worst and best instances and samples accordingly
    '''
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






def should_learn_float_old(x, y, n_neighbors=1):
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


def need_sampler(scores,values):
    '''
    this was my ring checker
    '''
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



def Sampler(space, keys):
    if len(keys) ==1:
        return  Simple(space, keys[0])
    else:
        return  Samplerr(space, keys)
