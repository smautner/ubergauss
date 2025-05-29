from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from ubergauss import hyperopt as ho
from ubergauss import optimization as op
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pprint import pprint
import structout as so
from hyperopt.pyll.stochastic import sample as hypersample
'''
for ints -> build a model cumsum(occcurance good / occurance bad) ,  then sample accoridngly
for floats -> gaussian sample from the top 50% of the scores
'''

class nutype:

    def __init__(self, space, f, data, numsample = 16):
        self.f = f
        self.data = data
        self.numsample = numsample
        space  = ho.spaceship(space)
        self.params =  [space.sample() for x in range(self.numsample) ]

        self.space = space
        self.scores = []
        self.carry = pd.DataFrame()
        self.runs = []
        self.paramgroups = self.getgroups()

        self.samplers = [Sampler(space,p) for p in self.paramgroups]

    def getgroups(self):
        keys = {k:[k] for k in self.params[0].keys()}
        for slave, master in self.space.dependencies.items():
            keys.pop(slave)
            keys[master].append(slave)
        r =  list(keys.values())
        return r



    def opti(self):
        # get new data
        self.df = op.gridsearch(self.f, data_list = self.data,tasks = self.params)
        self.df = fix(self.df)
        self.df = self.df.fillna(0)
        self.df = self.df.sort_values(by='score', ascending=True)

        # save the run and get the next parameters
        self.runs.append(self.df)
        self.scores+=self.df.score.tolist()
        self.nuParams()
        return self
        # self.print()


    def nuParams(self):
        '''
        first the non dependants can be done as usual,
        then for the dependants, they need subsamplers , so a sampler returns a dict v:k... in the end i can combine all the dicts...
        '''
        # data = pd.concat((self.carry,self.df))
        data = pd.concat((self.carry,self.df))
        for s in self.samplers:
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

    def getmax(self):
        dfdf = pd.concat(self.runs)
        return dfdf.sort_values(by='score', ascending=False).iloc[0]['score']

    def print(self):
        scr = pd.concat(self.runs)
        so.lprint(scr.score)
        best_run = scr.sort_values(by='score', ascending=False).iloc[0]
        print('Best params:\n', best_run)

        plt.plot(scr.score.cummax().tolist())

        plt.show()
        plot_params_with_hist(self.params, self.df,self.samplers)

        # print best params


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
    print(dict(zip(allints, scores)))




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
        print(f"{values,m,s=}")

        # print(f"{m=} {s=} {values=}")
        samples = lambda: np.random.normal(m,s)
        # print mean and std
        return samples




















def fix(df):
    # Identify columns that define a unique parameter combination (all except 'score' and 'datafield')
    param_cols = [col for col in df.columns if col not in ['score', 'datafield' ,'time']]

    # Calculate the average score for each parameter combination across all datafields
    avg_scores = df.groupby(param_cols)['score'].mean().reset_index()

    # Rename the calculated average score column to 'score'
    avg_scores = avg_scores.rename(columns={'score': 'average_score'})

    # Filter the original DataFrame to keep only rows where datafield is 0
    df_filtered = df[df['datafield'] == 0].copy()

    # Merge the calculated average scores back into the filtered DataFrame
    # The 'score' column in df_filtered will be replaced by the 'average_score' from avg_scores
    df_fixed = pd.merge(df_filtered.drop(columns='score'), avg_scores, on=param_cols, how='left')

    # Rename the 'average_score' column back to 'score'
    df_fixed = df_fixed.rename(columns={'average_score': 'score'})


    # Return the DataFrame with scores fixed and filtered to datafield == 0
    return df_fixed

def plot_params_with_hist(params, df, samp):
    params = pd.DataFrame(params)
    for col in params.columns:
        if col == "score":
            continue  # Skip the score column itself

        fig, ax1 = plt.subplots(figsize=(8, 4))

        # Lineplot: param vs score
        sns.scatterplot(x=col, y="score", data=df, ax=ax1, color='blue', label='Score')
        ax1.set_ylabel("Score", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Histogram: distribution of values in df
        ax2 = ax1.twinx()
        sns.histplot(params[col], ax=ax2, color='gray', alpha=0.3, bins=20, label='Distribution')
        ax2.set_ylabel("Frequency", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')

        # Titles and layout
        plt.title(f"{col} ")
        fig.tight_layout()
        plt.show()


